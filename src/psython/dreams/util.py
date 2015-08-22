from io import StringIO

import numpy as np
from IPython.display import Image, display
import scipy.ndimage as nd

import PIL.Image


def create_image(rel_path):
    return np.float32(PIL.Image.open(rel_path))


def ls_blob_keys(net):
    return net.blobs.keys()


def iterate_frames(net, img, subdir, iter_n=10, octave_n=4, octave_scale=1.4,
                   end='inception_4c/output', clip=True, **step_params):
    frame = img
    frame_i = 0
    h, w = frame.shape[:2]
    s = 0.05  # scale coefficient
    for i in xrange(iter_n):
        frame = deep_dream(net, frame, octave_n, octave_scale, end, clip, step_params)
        PIL.Image.fromarray(np.uint8(frame)).save("frames/%s/%04d.jpg" % (subdir, frame_i))
        frame = nd.affine_transform(frame, [1 - s, 1 - s, 1], [h * s / 2, w * s / 2, 0], order=1)
        frame_i += 1


def show_array(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
