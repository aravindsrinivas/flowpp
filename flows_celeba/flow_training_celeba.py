import os
import time
from collections import deque
from pprint import pprint

import numpy as np
import tensorflow as tf

from tensorflow.distributions import Normal
from flows_imagenet.utils import iterbatches, seed_all, TensorBoardOutput, tile_imgs

from tensorflow.python.client import timeline
import cv2
from tqdm import tqdm 
import pickle

def gaussian_logp(y, logd):
    assert y.shape[0] == logd.shape[0] and y.dtype == logd.dtype and logd.shape.ndims == 1
    orig_dtype = y.dtype
    y, logd = map(tf.to_float, [y, logd])
    return tf.cast(
        tf.reduce_sum(Normal(0., 1.).log_prob(y)) / int(y.shape[0]) + tf.reduce_mean(logd),
        dtype=orig_dtype
    )

def build_forward(*, x, dequant_flow, flow, flow_kwargs):
    bs = int(x.shape[0])
    dequant_x, dequant_logd = dequant_flow.forward(x, **flow_kwargs)
    y, logd = flow.forward(dequant_x, **flow_kwargs)
    assert dequant_logd.shape == logd.shape == [bs]
    total_logd = dequant_logd + logd
    loss = -gaussian_logp(y, total_logd)
    return dequant_x, y, loss

def train(
        *,
        flow_constructor,
        logdir,
        lr_schedule,
        dropout_p,
        seed,
        init_bs,
        total_bs,
        ema_decay,
        steps_per_log,
        max_grad_norm,
        dtype=tf.float32,
        scale_loss=None,
        dataset='celeba64_5bit',
        steps_per_samples=1000,
        steps_per_dump=5000,
        n_epochs=2,
        restore_checkpoint=None,
        save_jpg=False,
):

    import horovod.tensorflow as hvd

    # Initialize Horovod
    hvd.init()
    # Verify that MPI multi-threading is supported.
    assert hvd.mpi_threads_supported()

    from mpi4py import MPI

    assert hvd.size() == MPI.COMM_WORLD.Get_size()

    is_root = hvd.rank() == 0

    def mpi_average(local_list):
        local_list = list(map(float, local_list))
        sums = MPI.COMM_WORLD.gather(sum(local_list), root=0)
        counts = MPI.COMM_WORLD.gather(len(local_list), root=0)
        sum_counts = sum(counts) if is_root else None
        avg = (sum(sums) / sum_counts) if is_root else None
        return avg, sum_counts


    # Seeding and logging setup
    seed_all(hvd.rank() + hvd.size() * seed)
    assert total_bs % hvd.size() == 0
    local_bs = total_bs // hvd.size()

    logger = None
    logdir = '{}_mpi{}_{}'.format(os.path.expanduser(logdir), hvd.size(), time.time())
    checkpointdir = os.path.join(logdir, 'checkpoints')
    profiledir = os.path.join(logdir, 'profiling')
    if is_root:
        print('Floating point format:', dtype)
        pprint(locals())
        os.makedirs(logdir)
        os.makedirs(checkpointdir)
        os.makedirs(profiledir)
        logger = TensorBoardOutput(logdir)

    # Load data
    
    if is_root:
        print('Loading data')
    MPI.COMM_WORLD.Barrier()
    assert dataset in ['celeba64_3bit', 'celeba64_5bit', 'celeba128_5bit']
    if dataset == 'celeba64_3bit':
        data_train = np.load('../celeba_full_64x64_3bit.npy')
        assert np.max(data_train) <= 7
        assert np.min(data_train) >= 0
        assert data_train.dtype == 'uint8'
        assert list(data_train.shape[1:]) == [64, 64, 3]    
    elif dataset == 'celeba64_5bit':
        data_train = np.load('../celeba_full_64x64_5bit.npy')
        assert np.max(data_train) <= 31
        assert np.min(data_train) >= 0
        assert data_train.dtype == 'uint8'
        assert list(data_train.shape[1:]) == [64, 64, 3]    
    elif dataset == 'celeba128_5bit':
        data_train = np.load('../celeba_128pixels_5bit_full.npy')
        assert np.max(data_train) <= 31
        assert np.min(data_train) >= 0
        assert data_train.dtype == 'uint8'
        assert list(data_train.shape[1:]) == [128, 128, 3]    
    data_train = data_train.astype(dtype.as_numpy_dtype)
    img_shp = list(data_train.shape[1:])
    if is_root:
        print('Training data: {}'.format(data_train.shape[0]))
        print('Image shape:', img_shp)
    bpd_scale_factor = 1. / (np.log(2) * np.prod(img_shp))

    # Build graph
    if is_root: print('Building graph')
    dequant_flow, flow = flow_constructor()
    # Data-dependent init
    if is_root: print('===== Init graph =====')
    x_init_sym = tf.placeholder(dtype, [init_bs] + img_shp)
    init_syms = build_forward(
        x=x_init_sym, dequant_flow=dequant_flow, flow=flow,
        flow_kwargs=dict(init=True, dropout_p=dropout_p, verbose=is_root)
    )
    # Training
    if is_root: print('===== Training graph =====')
    x_sym = tf.placeholder(dtype, [local_bs] + img_shp)
    _, y_sym, loss_sym = build_forward(
        x=x_sym, dequant_flow=dequant_flow, flow=flow,
        flow_kwargs=dict(dropout_p=dropout_p, verbose=is_root)
    )

    # EMA
    params = tf.trainable_variables()
    if is_root:
        print('Parameters', sum(np.prod(p.get_shape().as_list()) for p in params))
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    maintain_averages_op = tf.group(ema.apply(params))
    # Op for setting the ema params to the current non-ema params (for use after data-dependent init)
    name2var = {v.name: v for v in tf.global_variables()}
    copy_params_to_ema = tf.group([
        name2var[p.name.replace(':0', '') + '/ExponentialMovingAverage:0'].assign(p) for p in params
    ])

    if is_root: print('===== Sampling graph =====')
    samples_sym, _ = flow.inverse(tf.random_normal(y_sym.shape.as_list(), dtype=dtype), dropout_p=0, ema=ema,
                                  verbose=is_root)
    allgathered_samples_sym = hvd.allgather(tf.to_float(samples_sym))

    assert len(tf.trainable_variables()) == len(params)


    def run_sampling(sess, i_step, *, prefix=dataset, save_jpg=save_jpg):
        samples = sess.run(allgathered_samples_sym)
        if is_root: 
            print('samples gathered from the session')
            np.save('samples_' + prefix + '_' + str(i_step) + '.npy', samples)
            if save_jpg:
                if dataset == 'celeba64_3bit':
                    viz_samples = np.clip(samples, 0, 7)
                    viz_samples = np.floor(viz_samples)
                    viz_samples = viz_samples.astype('uint8')
                    viz_samples = viz_samples * 32
                    viz_samples = tile_imgs(viz_samples)
                    cv2.imwrite('samples_' + prefix + '_' + str(i_step) + '.jpg', viz_samples)
                else:
                    viz_samples = np.clip(samples, 0, 31)
                    viz_samples = np.floor(viz_samples)
                    viz_samples = viz_samples.astype('uint8')
                    viz_samples = viz_samples * 8
                    viz_samples = tile_imgs(viz_samples)
                    cv2.imwrite('samples_' + prefix + '_' + str(i_step) + '.jpg', viz_samples)

    # Optimization
    lr_sym = tf.placeholder(dtype, [], 'lr')
    optimizer = hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr_sym))
    if scale_loss is None:
        grads_and_vars = optimizer.compute_gradients(loss_sym, var_list=params)
    else:
        grads_and_vars = [
            (g / scale_loss, v) for (g, v) in optimizer.compute_gradients(loss_sym * scale_loss, var_list=params)
        ]
    if max_grad_norm is not None:
        clipped_grads, grad_norm_sym = tf.clip_by_global_norm([g for (g, _) in grads_and_vars], max_grad_norm)
        grads_and_vars = [(cg, v) for (cg, (_, v)) in zip(clipped_grads, grads_and_vars)]
    else:
        grad_norm_sym = tf.constant(0.)
    opt_sym = tf.group(optimizer.apply_gradients(grads_and_vars), maintain_averages_op)

    def loop(sess: tf.Session):
        i_step = 0
        i_step_lr = 0
        if is_root: print('Initializing')
        sess.run(tf.global_variables_initializer())

        if restore_checkpoint is not None:
            """If restoring from an existing checkpoint whose path is specified in the launcher"""
            restore_step = int(restore_checkpoint.split('-')[-1])
            if is_root:
                saver = tf.train.Saver()
                print('Restoring checkpoint:', restore_checkpoint)
                print('Restoring from step:', restore_step)
                saver.restore(sess, restore_checkpoint)
                print('Loaded checkpoint')
            else:
                saver = None
            i_step = restore_step
            """You could re-start with the warm-up or start from wherever the checkpoint stopped depending on what is needed.
               If the session had to be stopped due to NaN/Inf, warm-up from a most recent working checkpoint is recommended.
               If it was because of Horovod Crash / Machine Shut down, re-starting from the same LR can be done in which case
               you need to uncomment the blow line. By default, it warms up."""
            #i_step_lr = restore_step 
        else:
            if is_root: print('Data dependent init')
            sess.run(init_syms, {x_init_sym: data_train[np.random.randint(0, data_train.shape[0], init_bs)]})
            sess.run(copy_params_to_ema)
            saver = tf.train.Saver() if is_root else None
        if is_root: print('Broadcasting initial parameters')
        sess.run(hvd.broadcast_global_variables(0))
        sess.graph.finalize()

        if is_root: print('Training')

        loss_hist = deque(maxlen=steps_per_log)
        gnorm_hist = deque(maxlen=steps_per_log)
        for i_epoch in range(n_epochs):

            epoch_start_t = time.time()
            for i_epoch_step, (batch,) in enumerate(iterbatches(  # non-sharded: each gpu goes through the whole dataset
                    [data_train], batch_size=local_bs, include_final_partial_batch=False,
            )):
                lr = lr_schedule(i_step_lr)
                loss, gnorm, _ = sess.run(
                    [loss_sym, grad_norm_sym, opt_sym], {x_sym: batch, lr_sym: lr},
                )
                loss_hist.append(loss)
                gnorm_hist.append(gnorm)

                if i_epoch == i_epoch_step == 0:
                    epoch_start_t = time.time()

                if i_step % steps_per_log == 0:
                    loss_hist_means = MPI.COMM_WORLD.gather(float(np.mean(loss_hist)), root=0)
                    gnorm_hist_means = MPI.COMM_WORLD.gather(float(np.mean(gnorm_hist)), root=0)
                    steps_per_sec = (i_epoch_step + 1) / (time.time() - epoch_start_t)
                    if is_root:
                        kvs = [
                            ('iter', i_step),
                            ('epoch', i_epoch + i_epoch_step * local_bs / data_train.shape[0]),  # epoch for this gpu
                            ('bpd', float(np.mean(loss_hist_means) * bpd_scale_factor)),
                            ('gnorm', float(np.mean(gnorm_hist_means))),
                            ('lr', float(lr)),
                            ('fps', steps_per_sec * total_bs),  # fps calculated over all gpus (this epoch)
                            ('sps', steps_per_sec),
                        ]
                        logger.writekvs(kvs, i_step)

                if i_step > 0 and i_step % steps_per_samples == 0 and i_step_lr > 0:
                    run_sampling(sess, i_step=i_step)
 
                if i_step % steps_per_dump == 0 and i_step > 0 and i_step_lr > 0:
                    if saver is not None:
                        saver.save(sess, os.path.join(checkpointdir, 'model'), global_step=i_step)
                    
                i_step += 1
                i_step_lr += 1
            # End of epoch

    # Train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())  # Pin GPU to local rank (one GPU per process)
    with tf.Session(config=config) as sess:
        loop(sess)
