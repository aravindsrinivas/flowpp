import os.path as osp
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2

NUMERIC_TYPES = (int, np.int32, np.int64, float, np.float32, np.float64)

class TensorBoardOutput:
    def __init__(self, dir):
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.python.util import compat
        self.dir = dir
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs, step):
        def summary_val(k, v):
            if isinstance(v, NUMERIC_TYPES):
                return tf.Summary.Value(tag=k, simple_value=float(v))
            elif isinstance(v, np.ndarray) and v.ndim == 3 and v.dtype == np.uint8:  # assume image
                # import cv2
                # img = cv2.imencode('.png', v[:, :, ::-1])[1].tostring()
                from PIL import Image
                import io
                img = Image.fromarray(v)
                with io.BytesIO() as f:
                    img.save(f, 'PNG')
                    img = f.getvalue()
                return tf.Summary.Value(
                    tag=k,
                    image=tf.Summary.Image(
                        encoded_image_string=img,
                        height=v.shape[0],
                        width=v.shape[1]
                    )
                )
            raise NotImplementedError

        summary = tf.Summary(value=[summary_val(k, v) for k, v in kvs])
        event = event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = step
        self.writer.WriteEvent(event)
        self.writer.Flush()
        print(' '.join(k + '=' + ('{:07d}' if isinstance(v, int) else '{:.5f}').format(v)
                       for k, v in kvs if isinstance(v, NUMERIC_TYPES)))

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    # arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def tile_imgs(imgs, *, pad_pixels=1, pad_val=255):
    assert pad_pixels >= 0 and 0 <= pad_val <= 255

    imgs = np.asarray(imgs)
    assert imgs.dtype == np.uint8
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    n, h, w, c = imgs.shape
    assert c == 1 or c == 3, 'Expected 1 or 3 channels'

    ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
    imgs = np.pad(
        imgs,
        pad_width=((0, ceil_sqrt_n ** 2 - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
        mode='constant',
        constant_values=pad_val
    )
    h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
    imgs = imgs.reshape(ceil_sqrt_n, ceil_sqrt_n, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4)
    imgs = imgs.reshape(ceil_sqrt_n * h, ceil_sqrt_n * w, c)
    if pad_pixels > 0:
        imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
    return imgs

def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255):
    import cv2
    cv2.imwrite(filename, tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val)[:, :, ::-1])

def save_tiled_imgs_scipy(filename, imgs, pad_pixels=1, pad_val=255):
    import scipy.misc
    scipy.misc.imsave(filename, tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val)[:, :, ::-1])
