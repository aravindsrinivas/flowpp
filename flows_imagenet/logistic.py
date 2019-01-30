import numpy as np
import tensorflow as tf


def logistic_logpdf(*, x, mean, logscale):
    """
    log density of logistic distribution
    this operates elementwise
    """
    z = (x - mean) * tf.exp(-logscale)
    return z - logscale - 2 * tf.nn.softplus(z)


def logistic_logcdf(*, x, mean, logscale):
    """
    log cdf of logistic distribution
    this operates elementwise
    """
    z = (x - mean) * tf.exp(-logscale)
    return tf.log_sigmoid(z)


def test_logistic():
    import scipy.stats

    # TF graph for logistic pdf computation
    tf.reset_default_graph()
    in_x = tf.placeholder(tf.float64, [None])
    in_means = tf.placeholder(tf.float64, [None])
    in_logscales = tf.placeholder(tf.float64, [None])
    out_logpdf = logistic_logpdf(x=in_x, mean=in_means, logscale=in_logscales)
    out_logcdf = logistic_logcdf(x=in_x, mean=in_means, logscale=in_logscales)

    # Evaluate log pdf at these points
    n = 100
    xs = np.linspace(-5, 5, n)

    with tf.Session() as sess:
        # Test against scipy
        for loc in np.linspace(-1, 2, 5):
            for scale in np.linspace(.01, 3, 5):
                true_logpdfs = scipy.stats.logistic.logpdf(xs, loc, scale)
                true_logcdfs = scipy.stats.logistic.logcdf(xs, loc, scale)
                logpdfs, logcdfs = sess.run([out_logpdf, out_logcdf], {
                    in_x: xs,
                    in_means: [loc] * n,
                    in_logscales: np.log([scale] * n)
                })
                assert np.allclose(logpdfs, true_logpdfs)
                assert np.allclose(logcdfs, true_logcdfs)


def mixlogistic_logpdf(*, x, prior_logits, means, logscales):
    """logpdf of a mixture of logistics"""
    assert len(x.get_shape()) + 1 == len(prior_logits.get_shape()) == len(means.get_shape()) == len(
        logscales.get_shape())
    return tf.reduce_logsumexp(
        tf.nn.log_softmax(prior_logits, axis=-1) + logistic_logpdf(
            x=tf.expand_dims(x, -1), mean=means, logscale=logscales),
        axis=-1
    )


def mixlogistic_logcdf(*, x, prior_logits, means, logscales):
    """log cumulative distribution function of a mixture of logistics"""
    assert (len(x.get_shape()) + 1 == len(prior_logits.get_shape()) ==
            len(means.get_shape()) == len(logscales.get_shape()))
    return tf.reduce_logsumexp(
        tf.nn.log_softmax(prior_logits, axis=-1) + logistic_logcdf(
            x=tf.expand_dims(x, -1), mean=means, logscale=logscales),
        axis=-1
    )


def test_logistic_mixture():
    import scipy.stats

    tf.reset_default_graph()
    in_x = tf.placeholder(tf.float64, [None])
    in_prior_logits = tf.placeholder(tf.float64, [None, None])
    in_means = tf.placeholder(tf.float64, [None, None])
    in_logscales = tf.placeholder(tf.float64, [None, None])
    out_logpdf = mixlogistic_logpdf(x=in_x, prior_logits=in_prior_logits, means=in_means, logscales=in_logscales)
    out_logcdf = mixlogistic_logcdf(x=in_x, prior_logits=in_prior_logits, means=in_means, logscales=in_logscales)

    n = 100
    xs = np.linspace(-5, 5, n)
    prior_logits = [.1, .2, 4]
    means = [-1., 0., 1]
    logscales = [-5., 0., 0.2]

    with tf.Session() as sess:
        logpdfs, logcdfs = sess.run([out_logpdf, out_logcdf], {
            in_x: xs,
            in_prior_logits: [prior_logits] * n,
            in_means: [means] * n,
            in_logscales: [logscales] * n,
        })

    prior_probs = np.exp(prior_logits) / np.exp(prior_logits).sum()
    scipy_probs = 0.
    scipy_cdfs = 0.
    for p, m, ls in zip(prior_probs, means, logscales):
        scipy_probs += p * scipy.stats.logistic.pdf(xs, m, np.exp(ls))
        scipy_cdfs += p * scipy.stats.logistic.cdf(xs, m, np.exp(ls))

    assert scipy_probs.shape == logpdfs.shape
    assert np.allclose(logpdfs, np.log(scipy_probs))
    assert np.allclose(logcdfs, np.log(scipy_cdfs))


def mixlogistic_sample(*, prior_logits, means, logscales):
    # Sample mixture component
    sampled_inds = tf.argmax(
        prior_logits - tf.log(-tf.log(tf.random_uniform(tf.shape(prior_logits), minval=1e-5, maxval=1. - 1e-5))),
        axis=-1
    )
    sampled_onehot = tf.one_hot(sampled_inds, tf.shape(prior_logits)[-1])
    # Pull out the sampled mixture component
    means = tf.reduce_sum(means * sampled_onehot, axis=-1)
    logscales = tf.reduce_sum(logscales * sampled_onehot, axis=-1)
    # Sample from the component
    u = tf.random_uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(logscales) * (tf.log(u) - tf.log(1. - u))
    return x


def assert_in_range(x, *, min, max):
    """Asserts that x is in [min, max] elementwise"""
    return tf.Assert(tf.logical_and(
        tf.greater_equal(tf.reduce_min(x), min),
        tf.less_equal(tf.reduce_max(x), max)
    ), [x])


def mixlogistic_invcdf(*, y, prior_logits, means, logscales, tol=1e-10, max_bisection_iters=500):
    """inverse cumulative distribution function of a mixture of logistics"""
    assert len(y.shape) + 1 == len(prior_logits.shape) == len(means.shape) == len(logscales.shape)
    dtype = y.dtype
    with tf.control_dependencies([assert_in_range(y, min=0., max=1.)]):
        y = tf.identity(y)

    def body(x, lb, ub, _last_diff):
        cur_y = tf.exp(mixlogistic_logcdf(x=x, prior_logits=prior_logits, means=means, logscales=logscales))
        gt = tf.cast(tf.greater(cur_y, y), dtype=dtype)
        lt = 1 - gt
        new_x = gt * (x + lb) / 2. + lt * (x + ub) / 2.
        new_lb = gt * lb + lt * x
        new_ub = gt * x + lt * ub
        diff = tf.reduce_max(tf.abs(new_x - x))
        return new_x, new_lb, new_ub, diff

    init_x = tf.zeros_like(y)
    maxscales = tf.reduce_sum(tf.exp(logscales), axis=-1, keepdims=True)  # sum of scales across mixture components
    init_lb = tf.reduce_min(means - 50 * maxscales, axis=-1)
    init_ub = tf.reduce_max(means + 50 * maxscales, axis=-1)
    init_diff = tf.constant(np.inf, dtype=dtype)

    out_x, _, _, _ = tf.while_loop(
        cond=lambda _x, _lb, _ub, last_diff: last_diff > tol,
        body=body,
        loop_vars=(init_x, init_lb, init_ub, init_diff),
        back_prop=False,
        maximum_iterations=max_bisection_iters
    )
    assert out_x.shape == y.shape
    return out_x


def test_mixlogistic_invcdf():
    tf.reset_default_graph()

    dtype = tf.float64

    n = 100
    d = 3
    in_x = tf.placeholder(dtype, [n])
    in_prior_logits = tf.placeholder(dtype, [n, d])
    in_means = tf.placeholder(dtype, [n, d])
    in_logscales = tf.placeholder(dtype, [n, d])
    logistic_args = dict(prior_logits=in_prior_logits, means=in_means, logscales=in_logscales)
    out_logcdf = mixlogistic_logcdf(x=in_x, **logistic_args)
    out_inv_cdf = mixlogistic_invcdf(y=tf.exp(out_logcdf), **logistic_args)
    assert out_inv_cdf.shape == in_x.shape
    err = tf.reduce_max(tf.abs(out_inv_cdf - in_x))

    range_max = 30
    xs = np.linspace(-range_max, range_max, n)
    prior_logits = [.1, .2, 4]
    means = [-1., 0., 1]
    logscales = [-5., 0., 0.2]

    with tf.Session() as sess:
        e, a, b = sess.run([err, in_x, out_inv_cdf], {
            in_x: xs,
            in_prior_logits: [prior_logits] * n,
            in_means: [means] * n,
            in_logscales: [logscales] * n,
        })
        print(np.c_[a, b, np.abs(a - b)])
        print(e)
        assert e < 1e-5
        print('ok')


if __name__ == '__main__':
    test_mixlogistic_invcdf()
