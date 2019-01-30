"""
Ablation: no attention (replaced with a pointwise MLP, with the same number of parameters)

Params: 31,443,440

Dropout 0.2
"""

import tensorflow as tf

from flows.flow_training import train, evaluate
from flows.flows import (
    conv2d, gated_conv, gaussian_sample_logp, VarConfig, get_var, layernorm, nin, gate,
    Flow, Compose, Inverse, ImgProc, Sigmoid, MixLogisticCDF, ElemwiseAffine,
    TupleFlip, CheckerboardSplit, ChannelSplit, SpaceToDepth, Norm, Pointwise
)


def gated_nin(x, *, name, pos_emb, dropout_p, vcfg: VarConfig):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        c = nin(c, name='proj1', num_units=3 * ch, vcfg=vcfg)
        assert c.shape == [bs, height, width, 3 * ch]
        c = tf.reshape(c, [bs, height, width, ch, 3])
        c1 = tf.reduce_max(c, axis=4)
        assert c1.shape == [bs, height, width, ch]
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, vcfg=vcfg)
        return x + gate(c2, axis=3)


class MixLogisticCoupling(Flow):
    def __init__(self, filters, blocks, components, init_scale=0.1):
        def f(x, *, vcfg: VarConfig, context=None, dropout_p=0., verbose=True):
            if vcfg.init and verbose:
                # debug stuff
                xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.shape))))
                x = tf.Print(
                    x, [tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x)],
                    message='{} (shape/mean/std/min/max) '.format(self.template.variable_scope.name), summarize=10
                )
            B, H, W, C = x.shape.as_list()
            pos_emb = get_var('pos_emb', shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01),
                              vcfg=vcfg)
            x = conv2d(x, name='proj_in', num_units=filters, vcfg=vcfg)
            for i_block in range(blocks):
                with tf.variable_scope(f'block{i_block}'):
                    x = gated_conv(x, name='conv', a=context, use_nin=True, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln1', vcfg=vcfg)
                    x = gated_nin(x, name='attn', pos_emb=pos_emb, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln2', vcfg=vcfg)
            x = conv2d(x, name='proj_out', num_units=C * (2 + 3 * components), init_scale=init_scale, vcfg=vcfg)
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            ml_logits, ml_means, ml_logscales = tf.split(x[:, :, :, :, 2:], 3, axis=4)
            assert s.shape == t.shape == [B, H, W, C]
            assert ml_logits.shape == ml_means.shape == ml_logscales.shape == [B, H, W, C, components]

            return Compose([
                MixLogisticCDF(logits=ml_logits, means=ml_means, logscales=ml_logscales),
                Inverse(Sigmoid()),
                ElemwiseAffine(scales=tf.exp(s), logscales=s, biases=t),
            ])

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        flow = self.template(cf, **kwargs)
        out, logd = flow.forward(ef)
        assert out.shape == ef.shape == cf.shape
        return (cf, out), logd

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        flow = self.template(cf, **kwargs)
        out, logd = flow.inverse(ef)
        assert out.shape == ef.shape == cf.shape
        return (cf, out), logd


def construct(*, filters, dequant_filters, components, blocks):
    # see MixLogisticAttnCoupling constructor
    dequant_coupling_kwargs = dict(filters=dequant_filters, blocks=2, components=components)
    coupling_kwargs = dict(filters=filters, blocks=blocks, components=components)

    class Dequant(Flow):
        def __init__(self):
            def shallow_processor(x, *, dropout_p, vcfg):
                x = x / 256.0 - 0.5
                (this, that), _ = CheckerboardSplit().forward(x)
                x = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, vcfg=vcfg)
                for i in range(3):
                    x = gated_conv(x, name=f'c{i}', vcfg=vcfg, dropout_p=dropout_p, use_nin=False, a=None)
                return x

            self.context_proc = tf.make_template("context_proc", shallow_processor)

            self.dequant_flow = Compose([
                CheckerboardSplit(),
                Norm(), Pointwise(), MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),
                Sigmoid(),
            ])

        def forward(self, x, *, vcfg, dropout_p=0., verbose=True, context=None):
            assert context is None
            eps, eps_logp = gaussian_sample_logp(x.shape.as_list())
            xd, logd = self.dequant_flow.forward(
                eps,
                context=self.context_proc(x, dropout_p=dropout_p, vcfg=vcfg),
                dropout_p=dropout_p, verbose=verbose, vcfg=vcfg
            )
            assert eps.shape == x.shape and logd.shape == eps_logp.shape == [x.shape[0]]
            return x + xd, logd - eps_logp

    dequant_flow = Dequant()
    flow = Compose([
        ImgProc(),

        CheckerboardSplit(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        SpaceToDepth(),

        ChannelSplit(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(ChannelSplit()),

        CheckerboardSplit(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),
    ])
    return dequant_flow, flow


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None)
    args = parser.parse_args()

    max_lr = 3e-4
    warmup_steps = 2000
    lr_decay = 1

    def lr_schedule(step):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        return max_lr * (lr_decay ** (step - warmup_steps))

    dropout_p = 0.2
    components = 32  # logistic mixture components
    blocks = 10
    filters = dequant_filters = 96
    ema_decay = 0.999

    def flow_constructor():
        return construct(filters=filters, dequant_filters=dequant_filters, components=components, blocks=blocks)

    if args.eval_checkpoint:
        evaluate(flow_constructor=flow_constructor, seed=0, restore_checkpoint=args.eval_checkpoint)
        return

    train(
        flow_constructor=flow_constructor,
        logdir=f'~/logs/abl_noattn_fbdq{dequant_filters}_mixlog{components}_blocks{blocks}_f{filters}_lr{max_lr}_drop{dropout_p}',
        lr_schedule=lr_schedule,
        dropout_p=dropout_p,
        seed=0,
        init_bs=128,
        total_bs=64,
        ema_decay=ema_decay,
        steps_per_log=100,
        epochs_per_val=1,
        max_grad_norm=1.,
    )


if __name__ == '__main__':
    main()
