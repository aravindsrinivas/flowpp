"""
Ablation: no mixture of logistics

Filters: 108 (to compensate for parameter count)
Params: 32,045,708

Dropout 0.2
"""

import tensorflow as tf

from flows.flow_training import train, evaluate
from flows.flows import (
    Flow, Compose, Inverse, ImgProc, Sigmoid,
    TupleFlip, CheckerboardSplit, ChannelSplit, SpaceToDepth, Norm, Pointwise, ElemwiseAffine,
    conv2d, gated_conv, gated_attn, layernorm, VarConfig, get_var, gaussian_sample_logp
)


class AffineAttnCoupling(Flow):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, heads=4, init_scale=0.1):
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
                    x = gated_attn(x, name='attn', pos_emb=pos_emb, heads=heads, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln2', vcfg=vcfg)
            components = 0  # no mixture of logistics
            x = conv2d(x, name='proj_out', num_units=C * (2 + 3 * components), init_scale=init_scale, vcfg=vcfg)
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])
            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            assert s.shape == t.shape == [B, H, W, C]
            return ElemwiseAffine(scales=tf.exp(s), logscales=s, biases=t)

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


def construct(*, filters, dequant_filters, blocks):
    dequant_coupling_kwargs = dict(filters=dequant_filters, blocks=2)
    coupling_kwargs = dict(filters=filters, blocks=blocks)

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
                Norm(), Pointwise(), AffineAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), AffineAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), AffineAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), AffineAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
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
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        SpaceToDepth(),

        ChannelSplit(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(ChannelSplit()),

        CheckerboardSplit(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), AffineAttnCoupling(**coupling_kwargs), TupleFlip(),
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
    blocks = 10
    filters = dequant_filters = 108
    ema_decay = 0.999

    def flow_constructor():
        return construct(filters=filters, dequant_filters=dequant_filters, blocks=blocks)

    if args.eval_checkpoint:
        evaluate(flow_constructor=flow_constructor, seed=0, restore_checkpoint=args.eval_checkpoint)
        return

    train(
        flow_constructor=flow_constructor,
        logdir=f'~/logs/abl_nomixlog_fbdq{dequant_filters}_blocks{blocks}_f{filters}_lr{max_lr}_drop{dropout_p}',
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
