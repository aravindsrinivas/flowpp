"""
CIFAR10 experiment

Params: 31,443,440
Dropout 0.2
"""

import tensorflow as tf

from .flow_training import train, evaluate
from .flows import (
    conv2d, gated_conv, gaussian_sample_logp,
    Flow, Compose, Inverse, ImgProc, Sigmoid,
    TupleFlip, CheckerboardSplit, ChannelSplit, SpaceToDepth, Norm, Pointwise, MixLogisticAttnCoupling
)


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
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
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
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        SpaceToDepth(),

        ChannelSplit(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(ChannelSplit()),

        CheckerboardSplit(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),
    ])
    return dequant_flow, flow


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None)
    parser.add_argument('--save_samples', type=str, default=None)
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
        evaluate(
            flow_constructor=flow_constructor, seed=0, restore_checkpoint=args.eval_checkpoint,
            samples_filename=args.save_samples
        )
        return

    train(
        flow_constructor=flow_constructor,
        logdir=f'~/logs/cifar_fbdq{dequant_filters}_mixlog{components}_blocks{blocks}_f{filters}_lr{max_lr}_drop{dropout_p}',
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
