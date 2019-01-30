"""
Num params: 73489960
Run it as: mpiexec -n {num_processes} python3.6 -m flows_imagenet.launchers.imagenet64_5bit_official from the flows master directory of the git repo. 
           num_processes=5 was used for this launcher on a 8-GPU (1080 Ti) machine with 40 GB RAM. 
If you want to use python3.5, remove the f string in the logdir.
"""

import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal
from tqdm import tqdm
from flows_imagenet.logistic import mixlogistic_logpdf, mixlogistic_logcdf, mixlogistic_invcdf
from flows_imagenet import flow_training_imagenet

DEFAULT_FLOATX = tf.float32
STORAGE_FLOATX = tf.float32

def to_default_floatx(x):
    return tf.cast(x, DEFAULT_FLOATX)

def at_least_float32(x):
    assert x.dtype in [tf.float16, tf.float32, tf.float64]
    if x.dtype == tf.float16:
        return tf.cast(x, tf.float32)
    return x

def get_var(var_name, *, ema, initializer, trainable=True, **kwargs):
    """forced storage dtype"""
    assert 'dtype' not in kwargs
    if isinstance(initializer, np.ndarray):
        initializer = initializer.astype(STORAGE_FLOATX.as_numpy_dtype)
    v = tf.get_variable(var_name, dtype=STORAGE_FLOATX, initializer=initializer, trainable=trainable, **kwargs)
    if ema is not None:
        assert isinstance(ema, tf.train.ExponentialMovingAverage)
        v = ema.average(v)
    return v

def _norm(x, *, axis, g, b, e=1e-5):
    assert x.shape.ndims == g.shape.ndims == b.shape.ndims
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.squared_difference(x, u), axis=axis, keepdims=True)
    x = (x - u) * tf.rsqrt(s + e)
    return x * g + b

def norm(x, *, name, ema):
    """Layer norm over last axis"""
    with tf.variable_scope(name):
        dim = int(x.shape[-1])
        _g = get_var('g', ema=ema, shape=[dim], initializer=tf.constant_initializer(1))
        _b = get_var('b', ema=ema, shape=[dim], initializer=tf.constant_initializer(0))
        g, b = map(to_default_floatx, [_g, _b])
        bcast_shape = [1] * (x.shape.ndims - 1) + [dim]
        return _norm(x, g=tf.reshape(g, bcast_shape), b=tf.reshape(b, bcast_shape), axis=-1)

def int_shape(x):
    return list(map(int, x.shape.as_list()))

def sumflat(x):
    return tf.reduce_sum(tf.reshape(x, [x.shape[0], -1]), axis=1)

def inverse_sigmoid(x):
    return -tf.log(tf.reciprocal(x) - 1.)

def init_normalization(x, *, name, init_scale=1., init, ema):
    with tf.variable_scope(name):
        g = get_var('g', shape=x.shape[1:], initializer=tf.constant_initializer(1.), ema=ema)
        b = get_var('b', shape=x.shape[1:], initializer=tf.constant_initializer(0.), ema=ema)
        if init:
            # data based normalization
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            assert m_init.shape == v_init.shape == scale_init.shape == g.shape == b.shape
            with tf.control_dependencies([
                g.assign(scale_init),
                b.assign(-m_init * scale_init)
            ]):
                g, b = tf.identity_n([g, b])
        return g, b

def dense(x, *, name, num_units, init_scale=1., init, ema):
    with tf.variable_scope(name):
        _, in_dim = x.shape
        W = get_var('W', shape=[in_dim, num_units], initializer=tf.random_normal_initializer(0, 0.05), ema=ema)
        b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), ema=ema)

        if init:
            y = tf.matmul(x, W)
            m_init, v_init = tf.nn.moments(y, [0])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            with tf.control_dependencies([
                W.assign(W * scale_init[None, :]),
                b.assign(-m_init * scale_init),
            ]):
                x = tf.identity(x)

        return tf.nn.bias_add(tf.matmul(x, W), b)

def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=(1, 1), pad='SAME', init_scale=1., init, ema):
    with tf.variable_scope(name):
        assert x.shape.ndims == 4
        W = get_var('W', shape=[*filter_size, int(x.shape[-1]), num_units],
                    initializer=tf.random_normal_initializer(0, 0.05), ema=ema)
        b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), ema=ema)

        if init:
            y = tf.nn.conv2d(x, W, [1, *stride, 1], pad)
            m_init, v_init = tf.nn.moments(y, [0, 1, 2])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            with tf.control_dependencies([
                W.assign(W * scale_init[None, None, None, :]),
                b.assign(-m_init * scale_init),
            ]):
                x = tf.identity(x)

        return tf.nn.bias_add(tf.nn.conv2d(x, W, [1, *stride, 1], pad), b)

def nin(x, *, num_units, **kwargs):
    assert 'num_units' not in kwargs
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units=num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])

def matmul_last_axis(x, w):
    _, out_dim = w.shape
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = tf.matmul(x, w)
    return tf.reshape(x, s[:-1] + [out_dim])

def concat_elu(x, *, axis=-1):
    return tf.nn.elu(tf.concat([x, -x], axis=axis))

def gate(x, *, axis):
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.sigmoid(b)

def gated_resnet(x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, init, ema, dropout_p):
    with tf.variable_scope(name):
        num_filters = int(x.shape[-1])

        c1 = conv(nonlinearity(x), name='c1', num_units=num_filters, init=init, ema=ema)
        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), name='a_proj', num_units=num_filters, init=init, ema=ema)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)

        c2 = (nin if use_nin else conv)(c1, name='c2', num_units=num_filters * 2, init_scale=0.1, init=init, ema=ema)
        return x + gate(c2, axis=3)

def attn(x, *, name, pos_emb, heads, init, ema, dropout_p):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        assert ch % heads == 0
        timesteps = height * width
        dim = ch // heads
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        # b, h, t, d == batch, num heads, num timesteps, per-head dim (C // heads)
        c = nin(c, name='proj1', num_units=3 * ch, init=init, ema=ema)
        assert c.shape == [bs, height, width, 3 * ch]
        # Split into heads / Q / K / V
        c = tf.reshape(c, [bs, timesteps, 3, heads, dim])  # b, t, 3, h, d
        c = tf.transpose(c, [2, 0, 3, 1, 4])  # 3, b, h, t, d
        q_bhtd, k_bhtd, v_bhtd = tf.unstack(c, axis=0)
        assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == [bs, heads, timesteps, dim]
        # Attention
        w_bhtt = tf.matmul(q_bhtd, k_bhtd, transpose_b=True) / np.sqrt(float(dim))
        w_bhtt = tf.cast(tf.nn.softmax(at_least_float32(w_bhtt)), dtype=x.dtype)
        assert w_bhtt.shape == [bs, heads, timesteps, timesteps]
        a_bhtd = tf.matmul(w_bhtt, v_bhtd)
        # Merge heads
        a_bthd = tf.transpose(a_bhtd, [0, 2, 1, 3])
        assert a_bthd.shape == [bs, timesteps, heads, dim]
        a_btc = tf.reshape(a_bthd, [bs, timesteps, ch])
        # Project
        c1 = tf.reshape(a_btc, [bs, height, width, ch])
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, init=init, ema=ema)
        return x + gate(c2, axis=3)

class Flow:
    def forward(self, x, **kwargs):
        raise NotImplementedError
    def backward(self, y, **kwargs):
        raise NotImplementedError

class Inverse(Flow):
    def __init__(self, base_flow):
        self.base_flow = base_flow

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x, **kwargs)
    
    def inverse(self, y, **kwargs):
        return self.base_flow.forward(y, **kwargs)

class Compose(Flow):
    def __init__(self, flows):
        self.flows = flows

    def _maybe_tqdm(self, iterable, desc, verbose):
        return tqdm(iterable, desc=desc) if verbose else iterable

    def forward(self, x, **kwargs):
        bs = int((x[0] if isinstance(x, tuple) else x).shape[0])
        logd_terms = []
        for i, f in enumerate(self._maybe_tqdm(self.flows, desc='forward {}'.format(kwargs),
                                               verbose=kwargs.get('verbose'))):
            assert isinstance(f, Flow)
            x, l = f.forward(x, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return x, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)

    def inverse(self, y, **kwargs):
        bs = int((y[0] if isinstance(y, tuple) else y).shape[0])
        logd_terms = []
        for i, f in enumerate(
                self._maybe_tqdm(self.flows[::-1], desc='inverse {}'.format(kwargs), verbose=kwargs.get('verbose'))):
            assert isinstance(f, Flow)
            y, l = f.inverse(y, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return y, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)

class ImgProc(Flow):
    def forward(self, x, **kwargs):
        x = x * (.9 / 32) + .05  # [0, 32] -> [.05, .95]
        x = -tf.log(1. / x - 1.)  # inverse sigmoid
        logd = np.log(.9 / 32) + tf.nn.softplus(x) + tf.nn.softplus(-x)
        logd = tf.reduce_sum(tf.reshape(logd, [int_shape(logd)[0], -1]), 1)
        return x, logd

    def inverse(self, y, **kwargs):
        y = tf.sigmoid(y)
        logd = tf.log(y) + tf.log(1. - y)
        y = (y - .05) / (.9 / 32)  # [.05, .95] -> [0, 32]
        logd -= np.log(.9 / 32)
        logd = tf.reduce_sum(tf.reshape(logd, [int_shape(logd)[0], -1]), 1)
        return y, logd

class TupleFlip(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tuple)
        a, b = x
        return (b, a), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        return (b, a), None

class SpaceToDepth(Flow):
    def __init__(self, block_size=2):
        self.block_size = block_size

    def forward(self, x, **kwargs):
        return tf.space_to_depth(x, self.block_size), None

    def inverse(self, y, **kwargs):
        return tf.depth_to_space(y, self.block_size), None

class CheckerboardSplit(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tf.Tensor)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H, W // 2, 2, C])
        a, b = tf.unstack(x, axis=3)
        assert a.shape == b.shape == [B, H, W // 2, C]
        return (a, b), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        assert a.shape == b.shape
        B, H, W_half, C = a.shape
        x = tf.stack([a, b], axis=3)
        assert x.shape == [B, H, W_half, 2, C]
        return tf.reshape(x, [B, H, W_half * 2, C]), None

class ChannelSplit(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tf.Tensor)
        assert len(x.shape) == 4 and x.shape[3] % 2 == 0
        return tuple(tf.split(x, 2, axis=3)), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        return tf.concat([a, b], axis=3), None

class Sigmoid(Flow):
    def forward(self, x, **kwargs):
        y = tf.sigmoid(x)
        logd = -tf.nn.softplus(x) - tf.nn.softplus(-x)
        return y, sumflat(logd)
    def inverse(self, y, **kwargs):
        x = inverse_sigmoid(y)
        logd = -tf.log(y) - tf.log(1. - y)
        return x, sumflat(logd)

class Norm(Flow):
    def __init__(self, init_scale=1.):
        def f(input_, forward, init, ema):
            assert not isinstance(input_, list)
            if isinstance(input_, tuple):
                is_tuple = True
            else:
                assert isinstance(input_, tf.Tensor)
                input_ = [input_]
                is_tuple = False

            bs = int(input_[0].shape[0])
            g_and_b = []
            for (i, x) in enumerate(input_):
                g, b = init_normalization(x, name='norm{}'.format(i), init_scale=init_scale, init=init, ema=ema)
                g = tf.maximum(g, 1e-10)
                assert x.shape[0] == bs and g.shape == b.shape == x.shape[1:]
                g_and_b.append((g, b))

            logd = tf.fill([bs], tf.add_n([tf.reduce_sum(tf.log(g)) for (g, _) in g_and_b]))
            if forward:
                out = [x * g[None] + b[None] for (x, (g, b)) in zip(input_, g_and_b)]
            else:
                out = [(x - b[None]) / g[None] for (x, (g, b)) in zip(input_, g_and_b)]
                logd = -logd

            if not is_tuple:
                assert len(out) == 1
                return out[0], logd
            return tuple(out), logd

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, init=False, ema=None, **kwargs):
        return self.template(x, forward=True, init=init, ema=ema)

    def inverse(self, y, init=False, ema=None, **kwargs):
        return self.template(y, forward=False, init=init, ema=ema)            

class MixLogisticCoupling(Flow):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, use_nin, components, attn_heads, use_ln,
                 with_affine=True, use_final_nin=False, init_scale=0.1, nonlinearity=concat_elu):
        self.components = components
        self.with_affine = with_affine
        self.scale_flow = Inverse(Sigmoid())

        def f(x, init, ema, dropout_p, verbose, context):
            # if verbose and context is not None:
            #     print('got context')
            if init and verbose:
                # debug stuff
                with tf.variable_scope('debug'):
                    xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
                    x = tf.Print(
                        x,
                        [
                            tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x),
                            tf.reduce_any(tf.is_nan(x)), tf.reduce_any(tf.is_inf(x))
                        ],
                        message='{} (shape/mean/std/min/max/nan/inf) '.format(self.template.variable_scope.name),
                        summarize=10,
                    )
            B, H, W, C = x.shape.as_list()

            pos_emb = to_default_floatx(get_var(
                'pos_emb', ema=ema, shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01),
            ))
            x = conv2d(x, name='c1', num_units=filters, init=init, ema=ema)
            for i_block in range(blocks):
                with tf.variable_scope('block{}'.format(i_block)):
                    x = gated_resnet(
                        x, name='conv', a=context, use_nin=use_nin, init=init, ema=ema, dropout_p=dropout_p
                    )
                    if use_ln:
                        x = norm(x, name='ln1', ema=ema)
            x = nonlinearity(x)
            x = (nin if use_final_nin else conv2d)(
                x, name='c2', num_units=C * (2 + 3 * components), init_scale=init_scale, init=init, ema=ema
            )
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            x = at_least_float32(x)  # do mix-logistics in tf.float32

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            ml_logits, ml_means, ml_logscales = tf.split(x[:, :, :, :, 2:], 3, axis=4)
            ml_logscales = tf.maximum(ml_logscales, -7.)

            assert s.shape == t.shape == [B, H, W, C]
            assert ml_logits.shape == ml_means.shape == ml_logscales.shape == [B, H, W, C, components]
            return s, t, ml_logits, ml_means, ml_logscales

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = tf.exp(
            mixlogistic_logcdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        )
        out, scale_logd = self.scale_flow.forward(out)
        if self.with_affine:
            assert out.shape == s.shape == t.shape
            out = tf.exp(s) * out + t

        logd = mixlogistic_logpdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert scale_logd.shape == logd.shape
        logd += scale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

    def inverse(self, y, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = float_ef
        if self.with_affine:
            out = tf.exp(-s) * (out - t)
        out, invscale_logd = self.scale_flow.inverse(out)
        out = tf.clip_by_value(out, 1e-5, 1. - 1e-5)
        out = mixlogistic_invcdf(y=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)

        logd = mixlogistic_logpdf(x=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = -tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert invscale_logd.shape == logd.shape
        logd += invscale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

class MixLogisticAttnCoupling(Flow):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, use_nin, components, attn_heads, use_ln,
                 with_affine=True, use_final_nin=False, init_scale=0.1, nonlinearity=concat_elu):
        self.components = components
        self.with_affine = with_affine
        self.scale_flow = Inverse(Sigmoid())

        def f(x, init, ema, dropout_p, verbose, context):
            if init and verbose:
                with tf.variable_scope('debug'):
                    xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
                    x = tf.Print(
                        x,
                        [
                            tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x),
                            tf.reduce_any(tf.is_nan(x)), tf.reduce_any(tf.is_inf(x))
                        ],
                        message='{} (shape/mean/std/min/max/nan/inf) '.format(self.template.variable_scope.name),
                        summarize=10,
                    )
            B, H, W, C = x.shape.as_list()

            pos_emb = to_default_floatx(get_var(
                'pos_emb', ema=ema, shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01),
            ))
            x = conv2d(x, name='c1', num_units=filters, init=init, ema=ema)
            for i_block in range(blocks):
                with tf.variable_scope('block{}'.format(i_block)):
                    x = gated_resnet(
                        x, name='conv', a=context, use_nin=use_nin, init=init, ema=ema, dropout_p=dropout_p
                    )
                    if use_ln:
                        x = norm(x, name='ln1', ema=ema)
                    x = attn(
                        x, name='attn', pos_emb=pos_emb, heads=attn_heads, init=init, ema=ema, dropout_p=dropout_p
                    )
                    if use_ln:
                        x = norm(x, name='ln2', ema=ema)
                    assert x.shape == [B, H, W, filters]
            x = nonlinearity(x)
            x = (nin if use_final_nin else conv2d)(
                x, name='c2', num_units=C * (2 + 3 * components), init_scale=init_scale, init=init, ema=ema
            )
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            x = at_least_float32(x)  # do mix-logistics stuff in float32

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            ml_logits, ml_means, ml_logscales = tf.split(x[:, :, :, :, 2:], 3, axis=4)
            ml_logscales = tf.maximum(ml_logscales, -7.)

            assert s.shape == t.shape == [B, H, W, C]
            assert ml_logits.shape == ml_means.shape == ml_logscales.shape == [B, H, W, C, components]
            return s, t, ml_logits, ml_means, ml_logscales

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = tf.exp(
            mixlogistic_logcdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        )
        out, scale_logd = self.scale_flow.forward(out)
        if self.with_affine:
            assert out.shape == s.shape == t.shape
            out = tf.exp(s) * out + t

        logd = mixlogistic_logpdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert scale_logd.shape == logd.shape
        logd += scale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

    def inverse(self, y, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = float_ef
        if self.with_affine:
            out = tf.exp(-s) * (out - t)
        out, invscale_logd = self.scale_flow.inverse(out)
        out = tf.clip_by_value(out, 1e-5, 1. - 1e-5)
        out = mixlogistic_invcdf(y=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)

        logd = mixlogistic_logpdf(x=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = -tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert invscale_logd.shape == logd.shape
        logd += invscale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

def gaussian_sample_logp(shape, dtype):
    eps = tf.random_normal(shape)
    logp = Normal(0., 1.).log_prob(eps)
    assert logp.shape == eps.shape
    logp = tf.reduce_sum(tf.layers.flatten(logp), axis=1)
    return tf.cast(eps, dtype=dtype), tf.cast(logp, dtype=dtype)

class Dequantizer(Flow):
    def __init__(self, dequant_flow):
        super().__init__()
        assert isinstance(dequant_flow, Flow)
        self.dequant_flow = dequant_flow

        def deep_processor(x, *, init, ema, dropout_p):
            (this, that), _ = CheckerboardSplit().forward(x)
            processed_context = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, init=init, ema=ema)
            for i in range(5):
                processed_context = gated_resnet(
                    processed_context, name='c{}'.format(i),
                    a=None, dropout_p=dropout_p, ema=ema, init=init,
                    use_nin=False
                )
                processed_context = norm(processed_context, name='dqln{}'.format(i), ema=ema)
                
            return processed_context

        self.context_proc = tf.make_template("context_proc", deep_processor)

    def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, **kwargs):
        eps, eps_logli = gaussian_sample_logp(x.shape, dtype=DEFAULT_FLOATX)
        unbound_xd, logd = self.dequant_flow.forward(
            eps,
            context=self.context_proc(x / 32.0 - 0.5, init=init, ema=ema, dropout_p=dropout_p),
            init=init, ema=ema, dropout_p=dropout_p, verbose=verbose
        )
        xd, sigmoid_logd = Sigmoid().forward(unbound_xd)
        assert x.shape == xd.shape and logd.shape == sigmoid_logd.shape == eps_logli.shape
        return x + xd, logd + sigmoid_logd - eps_logli


def construct(*, filters, blocks, components, attn_heads, use_nin, use_ln):
    dequant_coupling_kwargs = dict(
        filters=filters, blocks=5, use_nin=use_nin, components=components, attn_heads=attn_heads, use_ln=use_ln
    )
    dequant_flow = Dequantizer(Compose([
        CheckerboardSplit(),
        Norm(), 
        MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),
    ]))

    coupling_kwargs = dict(
        filters=filters, blocks=blocks, use_nin=use_nin, components=components, attn_heads=attn_heads, use_ln=use_ln
    )
    flow = Compose([
        ImgProc(),

        SpaceToDepth(),

        CheckerboardSplit(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        SpaceToDepth(),

        ChannelSplit(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(ChannelSplit()),

        CheckerboardSplit(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        SpaceToDepth(),
        
        ChannelSplit(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(ChannelSplit()),

        CheckerboardSplit(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), 
        MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),
    
    ])
    return dequant_flow, flow

def main():
    global DEFAULT_FLOATX
    DEFAULT_FLOATX = tf.float32

    max_lr = 4e-4
    warmup_steps = 10000
    bs = 60 
    # set this to a smaller value if it can't fit on your GPU. This setting works for a V100 but for smaller GPUs, batch size of 56 or 60 is advised.
    # make sure bs % num_mpi_processes == 0. There will be an assertion error otherwise. 

    def lr_schedule(step, *, decay=0.9995):
        """Ramp up to 4e-4 in 10K steps, stay there till 50K, geometric decay to 3e-4 by 55K steps, stay at 3e-4 till 110K steps, 
           then  warmup from 0 to 1e-5 for 20K steps, stay constant at 1e-5 for rest of the training."""
        global curr_lr
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        elif step > warmup_steps and step < (5 * warmup_steps):
            curr_lr =  max_lr
            return max_lr
        elif step > (5 * warmup_steps) and curr_lr > 3e-4:
            curr_lr *= decay
            return curr_lr
        elif step > (5 * warmup_steps) and curr_lr <= 3e-4 and step < 110000:
            return 3e-4
        elif step > 110000 and step < 130000:
            return (1e-5)*((step - 110000) / (2 * warmup_steps))
        elif step > 130000:
            return 1e-5

    dropout_p = 0.
    filters = 96
    blocks = 16
    components = 4  # logistic mixture components
    attn_heads = 4
    use_ln = True
 
    floatx_str = {tf.float32: 'fp32', tf.float16: 'fp16'}[DEFAULT_FLOATX]
    flow_training_imagenet.train(
        flow_constructor=lambda: construct(
            filters=filters,
            components=components,
            attn_heads=attn_heads,
            blocks=blocks,
            use_nin=True,
            use_ln=use_ln
        ),
        logdir=f'~/logs/2018-11-12/imagenet64_5bit_ELU_code_release_mix{components}_b{blocks}_f{filters}_h{attn_heads}_ln{int(use_ln)}_lr{max_lr}_bs{bs}_drop{dropout_p}_{floatx_str}',
        lr_schedule=lr_schedule,
        dropout_p=dropout_p,
        seed=0,
        init_bs=60, # set this to a smaller value if it can't fit on your GPU. This setting works for a V100 but for smaller GPUs, batch size of 56 is advised.
        dataset='imagenet64_5bit',
        total_bs=bs,
        ema_decay=.999,
        steps_per_log=100,
        steps_per_val=5000000, # basically not validating while training. set it to a lower value if you want to validate more frequently. 
        steps_per_dump=5000,
        steps_per_samples=5000,
        max_grad_norm=1.,
        dtype=DEFAULT_FLOATX,
        scale_loss=1e-2 if DEFAULT_FLOATX == tf.float16 else None,
        n_epochs=2,
        restore_checkpoint=None, # put in path to checkpoint in the format: path_to_checkpoint/model (no .meta / .ckpt)
        dump_samples_to_tensorboard=False, # if you want to push the tiled simples to tensorboard. 
    )

if __name__ == '__main__':
    main()
