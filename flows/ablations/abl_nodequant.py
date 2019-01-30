"""
Ablation: uniform dequantization

Filters: 104 (to compensate for parameter count)
Params: 32,324,408

Dropout 0.2
"""

import tensorflow as tf

from flows.flow_training import train, evaluate
from flows.flows import (
    Flow, Compose, Inverse, ImgProc,
    TupleFlip, CheckerboardSplit, ChannelSplit, SpaceToDepth, Norm, Pointwise, MixLogisticAttnCoupling
)


def construct(*, filters, components, blocks):
    # see MixLogisticAttnCoupling constructor
    coupling_kwargs = dict(filters=filters, blocks=blocks, components=components)

    class UnifDequant(Flow):
        def forward(self, x, **kwargs):
            return x + tf.random_uniform(x.shape.as_list()), tf.zeros([int(x.shape[0])])

    dequant_flow = UnifDequant()
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
    filters = 104
    ema_decay = 0.999

    def flow_constructor():
        return construct(filters=filters, components=components, blocks=blocks)

    if args.eval_checkpoint:
        evaluate(flow_constructor=flow_constructor, seed=0, restore_checkpoint=args.eval_checkpoint)
        return

    train(
        flow_constructor=flow_constructor,
        logdir=f'~/logs/abl_nodequant_mixlog{components}_blocks{blocks}_f{filters}_lr{max_lr}_drop{dropout_p}',
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
