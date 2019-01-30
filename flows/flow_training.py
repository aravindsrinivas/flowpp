import os
import time
from collections import deque
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal
from tqdm import trange

from .flows import VarConfig, sumflat
from .utils import iterbatches, seed_all, TensorBoardOutput, tile_imgs


def build_forward(*, x, dequant_flow, flow, flow_kwargs):
    dequant_x, dequant_logd = dequant_flow.forward(x, **flow_kwargs)
    y, main_logd = flow.forward(dequant_x, **flow_kwargs)
    logp = sumflat(Normal(0., 1.).log_prob(y))
    assert dequant_logd.shape == main_logd.shape == logp.shape == [y.shape[0]] == [dequant_x.shape[0]] == [x.shape[0]]
    total_logp = dequant_logd + main_logd + logp
    loss = -tf.reduce_mean(total_logp)
    return dequant_x, y, loss, total_logp


def load_data(*, dataset, dtype):
    if dataset == 'cifar10':
        (data_train, _), (data_val, _) = tf.keras.datasets.cifar10.load_data()
        assert data_train.shape[1:] == data_val.shape[1:] == (32, 32, 3)
    elif dataset == 'mnist':
        (data_train, _), (data_val, _) = tf.keras.datasets.mnist.load_data()
        data_train = data_train[..., None]
        data_val = data_val[..., None]
        assert data_train.shape[1:] == data_val.shape[1:] == (28, 28, 1)
    else:
        raise NotImplementedError(dataset)
    return data_train.astype(dtype), data_val.astype(dtype)


def setup_horovod():
    import horovod.tensorflow as hvd

    # Initialize Horovod
    hvd.init()
    # Verify that MPI multi-threading is supported.
    assert hvd.mpi_threads_supported()

    from mpi4py import MPI

    assert hvd.size() == MPI.COMM_WORLD.Get_size()

    is_root = hvd.rank() == 0

    def mpi_average(local_list):
        # _local_list_orig = local_list
        local_list = list(map(float, local_list))
        # print('RANK {} AVERAGING {} -> {}'.format(hvd.rank(), _local_list_orig, local_list))
        sums = MPI.COMM_WORLD.gather(sum(local_list), root=0)
        counts = MPI.COMM_WORLD.gather(len(local_list), root=0)
        sum_counts = sum(counts) if is_root else None
        avg = (sum(sums) / sum_counts) if is_root else None
        return avg, sum_counts

    return hvd, MPI, is_root, mpi_average


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
        epochs_per_val,
        max_grad_norm,
        dtype=tf.float32,
        scale_loss=None,
        restore_checkpoint=None,
        scale_grad=None,
        dataset='cifar10',
        steps_per_extra_samples=None
):
    hvd, MPI, is_root, mpi_average = setup_horovod()

    # Seeding and logging setup
    seed_all(hvd.rank() + hvd.size() * seed)
    assert total_bs % hvd.size() == 0
    local_bs = total_bs // hvd.size()

    logger = None
    logdir = '{}_mpi{}_{}'.format(os.path.expanduser(logdir), hvd.size(), time.time())
    checkpointdir = os.path.join(logdir, 'checkpoints')
    if is_root:
        print('Floating point format:', dtype)
        pprint(locals())
        os.makedirs(logdir)
        os.makedirs(checkpointdir)
        logger = TensorBoardOutput(logdir)

    # Load data
    if is_root:
        # Load once on root first to prevent downloading conflicts
        print('Loading data')
        load_data(dataset=dataset, dtype=dtype.as_numpy_dtype)
    MPI.COMM_WORLD.Barrier()
    data_train, data_val = load_data(dataset=dataset, dtype=dtype.as_numpy_dtype)
    img_shp = list(data_train.shape[1:])
    if is_root:
        print('Training data: {}, Validation data: {}'.format(data_train.shape[0], data_val.shape[0]))
        print('Image shape:', img_shp)
    bpd_scale_factor = 1. / (np.log(2) * np.prod(img_shp))

    # Build graph
    if is_root: print('Building graph')
    dequant_flow, flow = flow_constructor()
    # Data-dependent init
    if is_root: print('===== Init graph =====')
    x_init_sym = tf.placeholder(dtype, [init_bs] + img_shp)
    _, _, init_loss_sym, _ = build_forward(
        x=x_init_sym, dequant_flow=dequant_flow, flow=flow,
        flow_kwargs=dict(vcfg=VarConfig(init=True, ema=None, dtype=dtype), dropout_p=dropout_p, verbose=is_root)
    )
    # Training
    if is_root: print('===== Training graph =====')
    x_sym = tf.placeholder(dtype, [local_bs] + img_shp)
    _, y_sym, loss_sym, _ = build_forward(
        x=x_sym, dequant_flow=dequant_flow, flow=flow,
        flow_kwargs=dict(vcfg=VarConfig(init=False, ema=None, dtype=dtype), dropout_p=dropout_p, verbose=is_root)
    )

    # EMA
    params = tf.trainable_variables()
    if is_root:
        # for p in params:
        #     print(p.name, p.shape)
        print('Parameters', sum(np.prod(p.get_shape().as_list()) for p in params))
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    maintain_averages_op = tf.group(ema.apply(params))
    # Op for setting the ema params to the current non-ema params (for use after data-dependent init)
    name2var = {v.name: v for v in tf.global_variables()}
    copy_params_to_ema = tf.group([
        name2var[p.name.replace(':0', '') + '/ExponentialMovingAverage:0'].assign(p) for p in params
    ])

    # Validation and sampling (with EMA)
    if is_root: print('===== Validation graph =====')
    val_flow_kwargs = dict(vcfg=VarConfig(init=False, ema=ema, dtype=dtype), dropout_p=0, verbose=is_root)
    val_dequant_x_sym, val_y_sym, val_loss_sym, _ = build_forward(
        x=x_sym, dequant_flow=dequant_flow, flow=flow, flow_kwargs=val_flow_kwargs
    )
    # for debugging invertibility
    val_inverr_sym = tf.reduce_max(tf.abs(val_dequant_x_sym - flow.inverse(val_y_sym, **val_flow_kwargs)[0]))

    if is_root: print('===== Sampling graph =====')
    samples_sym, _ = flow.inverse(tf.random_normal(y_sym.shape.as_list(), dtype=dtype), **val_flow_kwargs)
    allgathered_samples_sym = hvd.allgather(tf.to_float(samples_sym))
    assert len(tf.trainable_variables()) == len(params)

    def run_validation(sess, i_step):
        data_val_shard = np.array_split(data_val, hvd.size(), axis=0)[hvd.rank()]
        shard_losses, shard_inverrs = zip(*[
            sess.run([val_loss_sym, val_inverr_sym], {x_sym: val_batch}) for val_batch, in
            iterbatches([data_val_shard], batch_size=local_bs, include_final_partial_batch=False)
        ])
        val_loss, total_count = mpi_average(shard_losses)
        inv_err, _ = mpi_average(shard_inverrs)
        samples = sess.run(allgathered_samples_sym)
        if is_root:
            logger.writekvs(
                [
                    ('val_bpd', bpd_scale_factor * val_loss),
                    ('val_inverr', inv_err),
                    ('num_val_examples', total_count * local_bs),
                    ('samples', tile_imgs(np.clip(samples, 0, 255).astype(np.uint8)))
                ],
                i_step
            )

    def run_sampling_only(sess, i_step):
        samples = sess.run(allgathered_samples_sym)
        if is_root:
            logger.writekvs(
                [
                    ('samples', tile_imgs(np.clip(samples, 0, 255).astype(np.uint8)))
                ],
                i_step
            )

    # Optimization
    lr_sym = tf.placeholder(dtype, [], 'lr')
    optimizer = hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr_sym))

    if scale_loss is None:
        grads_and_vars = optimizer.compute_gradients(loss_sym, var_list=params)
    else:
        grads_and_vars = [
            (g / scale_loss, v) for (g, v) in optimizer.compute_gradients(loss_sym * scale_loss, var_list=params)
        ]

    if scale_grad is not None:
        grads_and_vars = [
            (g / scale_grad, v) for (g, v) in grads_and_vars
        ]
    if max_grad_norm is not None:
        clipped_grads, grad_norm_sym = tf.clip_by_global_norm([g for (g, _) in grads_and_vars], max_grad_norm)
        grads_and_vars = [(cg, v) for (cg, (_, v)) in zip(clipped_grads, grads_and_vars)]
    else:
        grad_norm_sym = tf.constant(0.)
    opt_sym = tf.group(optimizer.apply_gradients(grads_and_vars), maintain_averages_op)

    def loop(sess: tf.Session):
        i_step = 0

        if is_root: print('Initializing')
        sess.run(tf.global_variables_initializer())
        if restore_checkpoint is not None:
            # Restore from checkpoint
            if is_root:
                saver = tf.train.Saver()
                print('Restoring checkpoint:', restore_checkpoint)
                restore_step = int(restore_checkpoint.split('-')[-1])
                print('Restoring from step:', restore_step)
                saver.restore(sess, restore_checkpoint)
                i_step = restore_step
            else:
                saver = None
        else:
            # No checkpoint: perform data dependent initialization
            if is_root: print('Data dependent init')
            init_loss = sess.run(init_loss_sym,
                                 {x_init_sym: data_train[np.random.randint(0, data_train.shape[0], init_bs)]})
            if is_root: print('Init loss:', init_loss * bpd_scale_factor)
            sess.run(copy_params_to_ema)
            saver = tf.train.Saver() if is_root else None
        if is_root: print('Broadcasting initial parameters')
        sess.run(hvd.broadcast_global_variables(0))
        sess.graph.finalize()

        if is_root:
            print('Training')

        loss_hist = deque(maxlen=steps_per_log)
        gnorm_hist = deque(maxlen=steps_per_log)
        for i_epoch in range(99999999999):
            if i_epoch % epochs_per_val == 0:
                run_validation(sess, i_step=i_step)
                if saver is not None:
                    saver.save(sess, os.path.join(checkpointdir, 'model'), global_step=i_step)

            epoch_start_t = time.time()
            for i_epoch_step, (batch,) in enumerate(iterbatches(  # non-sharded: each gpu goes through the whole dataset
                    [data_train], batch_size=local_bs, include_final_partial_batch=False,
            )):

                if steps_per_extra_samples is not None and i_step % steps_per_extra_samples == 0:
                    run_sampling_only(sess, i_step)

                lr = lr_schedule(i_step)
                loss, gnorm, _ = sess.run([loss_sym, grad_norm_sym, opt_sym], {x_sym: batch, lr_sym: lr})
                loss_hist.append(loss)
                gnorm_hist.append(gnorm)

                # Skip timing the very first step, which will be unusually slow due to TF initialization
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
                i_step += 1
            # End of epoch

    # Train
    config = tf.ConfigProto()
    # config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())  # Pin GPU to local rank (one GPU per process)
    with tf.Session(config=config) as sess:
        loop(sess)


def evaluate(
        *,
        flow_constructor,
        seed,
        restore_checkpoint,
        total_bs=1024,
        iw_samples=4096,
        dtype=tf.float32,
        dataset='cifar10',
        samples_filename=None
):
    hvd, MPI, is_root, mpi_average = setup_horovod()

    restore_checkpoint = os.path.expanduser(restore_checkpoint)

    # Seeding and logging setup
    seed_all(hvd.rank() + hvd.size() * seed)
    assert total_bs % hvd.size() == 0
    local_bs = total_bs // hvd.size()
    assert iw_samples % total_bs == 0

    if is_root: print('===== EVALUATING {} ({} IW samples) ====='.format(restore_checkpoint, iw_samples))

    # Load data
    if is_root:
        # Load once on root first to prevent downloading conflicts
        print('Loading data')
        load_data(dataset=dataset, dtype=dtype.as_numpy_dtype)
    MPI.COMM_WORLD.Barrier()
    data_train, data_val = load_data(dataset=dataset, dtype=dtype.as_numpy_dtype)
    img_shp = list(data_train.shape[1:])
    if is_root:
        print('Training data: {}, Validation data: {}'.format(data_train.shape[0], data_val.shape[0]))
        print('Image shape:', img_shp)
    bpd_scale_factor = 1. / (np.log(2) * np.prod(img_shp))

    # Build graph
    if is_root: print('Building graph')
    dequant_flow, flow = flow_constructor()
    x_sym = tf.placeholder(dtype, [local_bs] + img_shp)
    # This is a fake training graph. Just used to mimic flow_training, so we can load from the saver
    build_forward(
        x=x_sym, dequant_flow=dequant_flow, flow=flow,
        flow_kwargs=dict(vcfg=VarConfig(init=False, ema=None, dtype=dtype), dropout_p=0, verbose=is_root)
        # note dropout is 0: it doesn't matter
    )

    # EMA
    params = tf.trainable_variables()
    if is_root: print('Parameters', sum(np.prod(p.get_shape().as_list()) for p in params))
    ema = tf.train.ExponentialMovingAverage(decay=0.9999999999999)  # ema turned off
    maintain_averages_op = tf.group(ema.apply(params))

    # Validation and sampling (with EMA)
    if is_root: print('===== Validation graph =====')
    val_flow_kwargs = dict(vcfg=VarConfig(init=False, ema=ema, dtype=dtype), dropout_p=0, verbose=is_root)
    val_dequant_x_sym, val_y_sym, val_loss_sym, val_logratio_sym = build_forward(
        x=x_sym, dequant_flow=dequant_flow, flow=flow,
        flow_kwargs=val_flow_kwargs
    )
    allgathered_val_logratios_sym = hvd.allgather(val_logratio_sym)
    # for debugging invertibility
    val_inverr_sym = tf.reduce_max(tf.abs(val_dequant_x_sym - flow.inverse(val_y_sym, **val_flow_kwargs)[0]))

    if is_root: print('===== Sampling graph =====')
    samples_sym, _ = flow.inverse(tf.random_normal(val_y_sym.shape.as_list(), dtype=dtype), **val_flow_kwargs)
    allgathered_samples_sym = hvd.allgather(tf.to_float(samples_sym))
    assert len(tf.trainable_variables()) == len(params)

    def run_iw_eval(sess):
        if is_root:
            print('Running IW eval with {} samples...'.format(iw_samples))
        # Go through one example at a time
        all_val_losses = []
        for i_example in (trange if is_root else range)(len(data_val)):
            # take this single example and tile it
            batch_x = np.tile(data_val[i_example, None, ...], (local_bs, 1, 1, 1))
            # repeatedly evaluate logd for the IWAE bound
            batch_logratios = np.concatenate(
                [sess.run(allgathered_val_logratios_sym, {x_sym: batch_x}) for _ in range(iw_samples // total_bs)]
            ).astype(np.float64)
            assert batch_logratios.shape == (iw_samples,)
            # log [1/n \sum_i exp(r_i)] = log [exp(-b) 1/n \sum_i exp(r_i + b)] = -b + log [1/n \sum_i exp(r_i + b)]
            shift = batch_logratios.max()
            all_val_losses.append(-bpd_scale_factor * (shift + np.log(np.mean(np.exp(batch_logratios - shift)))))
            if i_example % 100 == 0 and is_root:
                print(i_example, np.mean(all_val_losses))
        if is_root:
            print(f'Final ({len(data_val)}):', np.mean(all_val_losses))

    def run_standard_eval(sess):
        if is_root:
            print('Running standard eval...')
        # Standard validation (single sample)
        data_val_shard = np.array_split(data_val, hvd.size(), axis=0)[hvd.rank()]
        shard_losses, shard_inverrs = zip(*[
            sess.run([val_loss_sym, val_inverr_sym], {x_sym: val_batch}) for val_batch, in
            iterbatches([data_val_shard], batch_size=local_bs, include_final_partial_batch=False)
        ])
        val_loss, total_count = mpi_average(shard_losses)
        inv_err, _ = mpi_average(shard_inverrs)
        if is_root:
            for k, v in [
                ('val_bpd', bpd_scale_factor * val_loss),
                ('val_inverr', inv_err),
                ('num_val_examples', total_count * local_bs),
            ]:
                print(k, v)

    def run_sampling_only(sess):
        samples = sess.run(allgathered_samples_sym)
        # # warmup a few times
        # for _ in range(10):
        #     sess.run(allgathered_samples_sym)
        # # start timing
        # trials = 100
        # tstart = time.time()
        # for _ in range(trials):
        #     samples = sess.run(allgathered_samples_sym)
        # sample_time = (time.time() - tstart) / trials

        if is_root:
            from PIL import Image
            Image.fromarray(tile_imgs(np.clip(samples, 0, 255).astype(np.uint8))).save(samples_filename)
            print('Saved {} samples to {}'.format(len(samples), samples_filename))
            # print('Sampled in {} seconds'.format(sample_time))

    # Run
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())  # Pin GPU to local rank (one GPU per process)
    with tf.Session(config=config) as sess:
        if is_root: print('Initializing')
        sess.run(tf.global_variables_initializer())
        # Restore from checkpoint
        if is_root:
            print('Restoring checkpoint:', restore_checkpoint)
            saver = tf.train.Saver()
            saver.restore(sess, restore_checkpoint)
            print('Broadcasting initial parameters')
        sess.run(hvd.broadcast_global_variables(0))
        sess.graph.finalize()

        if samples_filename:
            run_sampling_only(sess)

        # Make sure data is the same on all MPI processes
        tmp_inds = [0, 183, 3, 6, 20, 88]
        check_batch = np.ascontiguousarray(data_val[tmp_inds])
        gathered_batches = np.zeros((hvd.size(), *check_batch.shape), check_batch.dtype) if is_root else None
        MPI.COMM_WORLD.Gather(check_batch, gathered_batches, root=0)
        if is_root:
            assert all(np.allclose(check_batch, b) for b in gathered_batches), 'data must be in the same order!'
            print('data ordering ok')

        # Run validation
        run_standard_eval(sess)
        run_iw_eval(sess)
