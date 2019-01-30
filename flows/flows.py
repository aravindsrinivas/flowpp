from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal
from tqdm import tqdm

from .logistic import mixlogistic_logpdf, mixlogistic_logcdf, mixlogistic_invcdf

VarConfig = namedtuple('VarConfig', ['init', 'ema', 'dtype', 'use_resource'])
VarConfig.__new__.__defaults__ = (False, None, tf.float32, False)


def get_var(var_name, *, shape, initializer, vcfg: VarConfig, trainable=True):
    assert vcfg is not None and isinstance(vcfg, VarConfig)
    if isinstance(initializer, np.ndarray):
        initializer = initializer.astype(vcfg.dtype.as_numpy_dtype)
    v = tf.get_variable(var_name, shape=shape, dtype=vcfg.dtype, initializer=initializer, trainable=trainable,
                        use_resource=vcfg.use_resource)
    if vcfg.ema is not None:
        assert isinstance(vcfg.ema, tf.train.ExponentialMovingAverage)
        v = vcfg.ema.average(v)
    return v


def dense(x, *, name, num_units, init_scale=1., vcfg: VarConfig):
    with tf.variable_scope(name):
        _, in_dim = x.shape
        W = get_var('W', shape=[in_dim, num_units], initializer=tf.random_normal_initializer(0, 0.05), vcfg=vcfg)
        b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), vcfg=vcfg)

        if vcfg.init:
            y = tf.matmul(x, W)
            m_init, v_init = tf.nn.moments(y, [0])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            new_W = W * scale_init[None, :]
            new_b = -m_init * scale_init
            with tf.control_dependencies([W.assign(new_W), b.assign(new_b)]):
                if vcfg.use_resource:
                    return tf.nn.bias_add(tf.matmul(x, new_W), new_b)
                else:
                    x = tf.identity(x)

        return tf.nn.bias_add(tf.matmul(x, W), b)


def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=(1, 1), pad='SAME', init_scale=1., vcfg: VarConfig):
    with tf.variable_scope(name):
        assert x.shape.ndims == 4
        W = get_var('W', shape=[*filter_size, int(x.shape[-1]), num_units],
                    initializer=tf.random_normal_initializer(0, 0.05), vcfg=vcfg)
        b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), vcfg=vcfg)

        if vcfg.init:
            y = tf.nn.conv2d(x, W, [1, *stride, 1], pad)
            m_init, v_init = tf.nn.moments(y, [0, 1, 2])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            new_W = W * scale_init[None, None, None, :]
            new_b = -m_init * scale_init
            with tf.control_dependencies([W.assign(new_W), b.assign(new_b)]):
                if vcfg.use_resource:
                    return tf.nn.bias_add(tf.nn.conv2d(x, new_W, [1, *stride, 1], pad), new_b)
                else:
                    x = tf.identity(x)

        return tf.nn.bias_add(tf.nn.conv2d(x, W, [1, *stride, 1], pad), b)


def init_normalization(x, *, name, init_scale=1., vcfg: VarConfig):
    with tf.variable_scope(name):
        g = get_var('g', shape=x.shape[1:], initializer=tf.constant_initializer(1.), vcfg=vcfg)
        b = get_var('b', shape=x.shape[1:], initializer=tf.constant_initializer(0.), vcfg=vcfg)
        if vcfg.init:
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


def nin(x, *, num_units, **kwargs):
    assert 'num_units' not in kwargs
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units=num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])


def concat_elu(x):
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def gate(x, *, axis):
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.sigmoid(b)


def layernorm(x, *, name, vcfg: VarConfig, e=1e-5):
    """Layer norm over last axis"""
    with tf.variable_scope(name):
        shape = [1] * (x.shape.ndims - 1) + [int(x.shape[-1])]
        g = get_var('g', shape=shape, initializer=tf.constant_initializer(1), vcfg=vcfg)
        b = get_var('b', shape=shape, initializer=tf.constant_initializer(0), vcfg=vcfg)
        u = tf.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.reduce_mean(tf.squared_difference(x, u), axis=-1, keepdims=True)
        return (x - u) * tf.rsqrt(s + e) * g + b


def gated_conv(x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, dropout_p, vcfg: VarConfig):
    with tf.variable_scope(name):
        num_filters = int(x.shape[-1])

        c1 = conv(nonlinearity(x), name='c1', num_units=num_filters, vcfg=vcfg)
        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), name='a_proj', num_units=num_filters, vcfg=vcfg)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)

        c2 = (nin if use_nin else conv)(c1, name='c2', num_units=num_filters * 2, init_scale=0.1, vcfg=vcfg)
        return x + gate(c2, axis=3)


def gated_attn(x, *, name, pos_emb, heads, dropout_p, vcfg: VarConfig):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        assert ch % heads == 0
        timesteps = height * width
        dim = ch // heads
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        # b, h, t, d == batch, num heads, num timesteps, per-head dim (C // heads)
        c = nin(c, name='proj1', num_units=3 * ch, vcfg=vcfg)
        assert c.shape == [bs, height, width, 3 * ch]
        # Split into heads / Q / K / V
        c = tf.reshape(c, [bs, timesteps, 3, heads, dim])  # b, t, 3, h, d
        c = tf.transpose(c, [2, 0, 3, 1, 4])  # 3, b, h, t, d
        q_bhtd, k_bhtd, v_bhtd = tf.unstack(c, axis=0)
        assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == [bs, heads, timesteps, dim]
        # Attention
        w_bhtt = tf.matmul(q_bhtd, k_bhtd, transpose_b=True) / np.sqrt(float(dim))
        w_bhtt = tf.nn.softmax(w_bhtt)
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
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, vcfg=vcfg)
        return x + gate(c2, axis=3)


####

def sumflat(x):
    return tf.reduce_sum(tf.reshape(x, [x.shape[0], -1]), axis=1)


def gaussian_sample_logp(shape):
    eps = tf.random_normal(shape)
    logp = Normal(0., 1.).log_prob(eps)
    assert logp.shape == eps.shape
    return eps, sumflat(logp)


def assert_in_range(x, *, min, max):
    """Asserts that x is in [min, max] elementwise"""
    return tf.Assert(tf.logical_and(
        tf.greater_equal(tf.reduce_min(x), min),
        tf.less_equal(tf.reduce_max(x), max)
    ), [x])


def inverse_sigmoid(x):
    return -tf.log(tf.reciprocal(x) - 1.)


### flows

class Flow:
    def forward(self, x, *, vcfg, dropout_p=0., verbose=True, context=None):
        raise NotImplementedError

    def inverse(self, y, *, vcfg, dropout_p=0., verbose=True, context=None):
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
        for f in self._maybe_tqdm(self.flows, desc='forward {}'.format(kwargs.get('vcfg')),
                                  verbose=kwargs.get('verbose')):
            assert isinstance(f, Flow)
            x, l = f.forward(x, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return x, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)

    def inverse(self, y, **kwargs):
        bs = int((y[0] if isinstance(y, tuple) else y).shape[0])
        logd_terms = []
        for f in self._maybe_tqdm(self.flows[::-1], desc='inverse {}'.format(kwargs.get('vcfg')),
                                  verbose=kwargs.get('verbose')):
            assert isinstance(f, Flow)
            y, l = f.inverse(y, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return y, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)


class Sigmoid(Flow):
    def forward(self, x, **kwargs):
        y = tf.sigmoid(x)
        logd = -tf.nn.softplus(x) - tf.nn.softplus(-x)
        return y, sumflat(logd)

    def inverse(self, y, **kwargs):
        x = inverse_sigmoid(y)
        logd = -tf.log(y) - tf.log(1. - y)
        return x, sumflat(logd)


class ImgProc(Flow):
    def __init__(self, max_val=256):
        self.max_val = max_val

    def forward(self, x, **kwargs):
        x = x * (.9 / self.max_val) + .05  # [0, self.max_val] -> [.05, .95]
        x, logd = Sigmoid().inverse(x)
        logd += np.log(.9 / self.max_val) * int(np.prod(x.shape.as_list()[1:]))
        return x, logd

    def inverse(self, y, **kwargs):
        y, logd = Sigmoid().forward(y)
        y = (y - .05) / (.9 / self.max_val)  # [.05, .95] -> [0, self.max_val]
        logd -= np.log(.9 / self.max_val) * int(np.prod(y.shape.as_list()[1:]))
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


####################################


class Norm(Flow):
    def __init__(self, init_scale=1.):
        def f(input_, forward, vcfg):
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
                g, b = init_normalization(x, name='norm{}'.format(i), init_scale=init_scale, vcfg=vcfg)
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

    def forward(self, x, *, vcfg, **kwargs):
        return self.template(x, forward=True, vcfg=vcfg)

    def inverse(self, y, *, vcfg, **kwargs):
        return self.template(y, forward=False, vcfg=vcfg)


class Pointwise(Flow):
    def __init__(self, noisy_identity_init=0.001):
        def f(input_, forward, vcfg):
            assert not isinstance(input_, list)
            if isinstance(input_, tuple):
                is_tuple = True
            else:
                assert isinstance(input_, tf.Tensor)
                input_ = [input_]
                is_tuple = False

            out, logds = [], []
            for i, x in enumerate(input_):
                _, img_h, img_w, img_c = x.shape.as_list()
                if noisy_identity_init:
                    # identity + gaussian noise
                    initializer = (
                            np.eye(img_c) + noisy_identity_init * np.random.randn(img_c, img_c)
                    ).astype(np.float32)
                else:
                    # random orthogonal
                    initializer = np.linalg.qr(np.random.randn(img_c, img_c))[0].astype(np.float32)
                W = get_var('W{}'.format(i), shape=None, initializer=initializer, vcfg=vcfg)
                out.append(self._nin(x, W if forward else tf.matrix_inverse(W)))
                logds.append(
                    (1 if forward else -1) * img_h * img_w *
                    tf.to_float(tf.log(tf.abs(tf.matrix_determinant(tf.to_double(W)))))
                )
            logd = tf.fill([input_[0].shape[0]], tf.add_n(logds))

            if not is_tuple:
                assert len(out) == 1
                return out[0], logd
            return tuple(out), logd

        self.template = tf.make_template(self.__class__.__name__, f)

    @staticmethod
    def _nin(x, w, b=None):
        _, out_dim = w.shape
        s = x.shape.as_list()
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = tf.matmul(x, w)
        if b is not None:
            assert b.shape.ndims == 1
            x = x + b[None, :]
        return tf.reshape(x, s[:-1] + [out_dim])

    def forward(self, x, *, vcfg, **kwargs):
        return self.template(x, forward=True, vcfg=vcfg)

    def inverse(self, y, *, vcfg, **kwargs):
        return self.template(y, forward=False, vcfg=vcfg)


class ElemwiseAffine(Flow):
    def __init__(self, *, scales, biases, logscales=None):
        self.scales = scales
        self.biases = biases
        self.logscales = logscales

    def _get_logscales(self):
        return tf.log(self.scales) if (self.logscales is None) else self.logscales

    def forward(self, x, **kwargs):
        logscales = self._get_logscales()
        assert logscales.shape == x.shape
        return (x * self.scales + self.biases), sumflat(logscales)

    def inverse(self, y, **kwargs):
        logscales = self._get_logscales()
        assert logscales.shape == y.shape
        return ((y - self.biases) / self.scales), sumflat(-logscales)


class MixLogisticCDF(Flow):
    """
    Elementwise transformation by the CDF of a mixture of logistics
    """

    def __init__(self, *, logits, means, logscales, min_logscale=-7.):
        self.logits = logits
        self.means = means
        self.logscales = logscales
        self.min_logscale = min_logscale

    def _get_logistic_kwargs(self):
        return dict(
            prior_logits=self.logits,
            means=self.means,
            logscales=tf.maximum(self.logscales, self.min_logscale)
        )

    def forward(self, x, **kwargs):
        logistic_kwargs = self._get_logistic_kwargs()
        out = tf.exp(mixlogistic_logcdf(x=x, **logistic_kwargs))
        logd = mixlogistic_logpdf(x=x, **logistic_kwargs)
        return out, sumflat(logd)

    def inverse(self, y, **kwargs):
        logistic_kwargs = self._get_logistic_kwargs()
        out = mixlogistic_invcdf(y=tf.clip_by_value(y, 0., 1.), **logistic_kwargs)
        logd = -mixlogistic_logpdf(x=out, **logistic_kwargs)
        return out, sumflat(logd)


class MixLogisticAttnCoupling(Flow):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, components, heads=4, init_scale=0.1, enable_print=True):
        def f(x, *, vcfg: VarConfig, context=None, dropout_p=0., verbose=True):
            if vcfg.init and verbose and enable_print:
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
                with tf.variable_scope('block{}'.format(i_block)):
                    x = gated_conv(x, name='conv', a=context, use_nin=True, dropout_p=dropout_p, vcfg=vcfg)
                    x = layernorm(x, name='ln1', vcfg=vcfg)
                    x = gated_attn(x, name='attn', pos_emb=pos_emb, heads=heads, dropout_p=dropout_p, vcfg=vcfg)
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


############################


def _run_split_test(split, dtype=tf.float64):
    bs = 5
    ch = 4
    shape = [bs, 8, 8, ch]

    with tf.Graph().as_default() as graph:
        x_in_sym = tf.placeholder(dtype, shape)
        (a_sym, b_sym), logd = split.forward(x_in_sym)
        x2_sym, logd2 = split.inverse((a_sym, b_sym))
        assert logd is None and logd2 is None

    x = np.random.randn(*shape)
    with tf.Session(graph=graph) as sess:
        x2 = sess.run(x2_sym, {x_in_sym: x})
        assert np.allclose(x, x2)


def test_checkerboard_split():
    _run_split_test(CheckerboardSplit())


def test_channel_split():
    _run_split_test(ChannelSplit())


def _finitediff(f, x, *, eps):
    """Log partial derivatives on the diagonal of the Jacobian"""
    x_flat = x.reshape(-1)
    diffs_flat = np.zeros_like(x_flat)
    for i in range(len(x_flat)):
        orig = x_flat[i]
        x_flat[i] = orig + eps
        y2 = f(x).flat[i]
        x_flat[i] = orig - eps
        y1 = f(x).flat[i]
        x_flat[i] = orig
        diffs_flat[i] = y2 - y1
    return (np.log(diffs_flat) - np.log(2 * eps)).reshape(x.shape)


def _run_flow_test(flow: Flow, *, input_bounds=(-5., 5.), input_shape=(5, 8, 8, 3),  # aux_shape=None,
                   check_logd=True, finitediff_eps=1e-6, dtype=tf.float64):
    # TODO: also feed in aux?
    assert isinstance(flow, Flow)
    assert len(input_bounds) == 2 and len(input_shape) == 4

    with tf.Graph().as_default() as graph:
        with tf.variable_scope('test_scope') as scope:
            x_in_sym = tf.placeholder(dtype, input_shape)
            init_syms = flow.forward(x_in_sym, vcfg=VarConfig(init=True, ema=None, dtype=dtype))
            y_sym, logd_sym = flow.forward(x_in_sym, vcfg=VarConfig(init=False, ema=None, dtype=dtype))
            x2_sym, invlogd_sym = flow.inverse(y_sym, vcfg=VarConfig(init=False, ema=None, dtype=dtype))

    with tf.Session(graph=graph) as sess:
        # Initialize
        sess.run(tf.variables_initializer(tf.global_variables(scope.name)))
        x = np.random.uniform(input_bounds[0], input_bounds[1], input_shape).astype(dtype.as_numpy_dtype)
        sess.run(init_syms, {x_in_sym: x})
        # Check inverse
        y, logd, x2, invlogd = sess.run([y_sym, logd_sym, x2_sym, invlogd_sym], {x_in_sym: x})
        assert np.allclose(x, x2)
        assert np.allclose(logd, -invlogd)
        assert x.shape == x2.shape == y.shape
        assert logd.shape == invlogd.shape == (x.shape[0],)
        print(logd)
        # Check logd
        if check_logd:
            finitediff_logd = _finitediff(lambda p: sess.run(y_sym, {x_in_sym: p}), x, eps=finitediff_eps)
            assert np.allclose(logd, finitediff_logd.reshape(input_shape[0], -1).sum(axis=1), atol=1e-5)

            finitediff_invlogd = _finitediff(lambda p: sess.run(x2_sym, {y_sym: p}), y, eps=finitediff_eps)
            assert np.allclose(invlogd, finitediff_invlogd.reshape(input_shape[0], -1).sum(axis=1), atol=1e-5)

    return dict(x=x, logd=logd, y=y)


def test_imgproc():
    _run_flow_test(ImgProc(), input_bounds=(0., 256.))


def test_sigmoid():
    _run_flow_test(Sigmoid())


def test_normalize():
    output = _run_flow_test(Norm())
    assert np.allclose(output['y'].mean(axis=0), 0.)
    assert np.allclose(output['y'].std(axis=0), 1.)


def test_elemwise_affine():
    input_shape = (5, 8, 8, 3)
    _run_flow_test(
        ElemwiseAffine(scales=np.exp(np.random.randn(*input_shape)), biases=np.random.randn(*input_shape)),
        input_shape=input_shape
    )


def test_mix_logistic_cdf():
    input_shape = (2, 4, 4, 3)
    mix_components = 4
    param_shape = (*input_shape, mix_components)
    _run_flow_test(
        MixLogisticCDF(
            logits=np.random.randn(*param_shape),
            means=np.random.randn(*param_shape),
            logscales=np.random.randn(*param_shape)
        ),
        input_shape=input_shape
    )


def test_mix_logistic_coupling():
    for split in [CheckerboardSplit, ChannelSplit]:
        _run_flow_test(
            Compose([
                split(),
                MixLogisticAttnCoupling(filters=16, blocks=2, components=4, heads=2),
                Inverse(split()),
            ]),
            input_shape=(2, 4, 4, 4)
        )
