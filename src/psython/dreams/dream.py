from IPython.display import clear_output
from google.protobuf import text_format
from ..caffe_helper import pil_to_caffe, caffe_to_pil
from .util import show_array
from .objective import *

import numpy as np
import scipy.ndimage as nd

import caffe

DEFAULT_OBJECTIVE = L2Objective()


def load_dnn_model(model_path='../../open-source/caffe/models/bvlc_googlenet/',
                   net_fn_name='deploy.prototxt', param_fn_name='bvlc_googlenet.caffemodel'):
    net_fn = model_path + net_fn_name
    param_fn = model_path + param_fn_name

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
                           channel_swap=(2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    return net


class Dream:
    """
    Keeps track of all the info and logic necessary to create the iconic "deep dream" images.
    """

    def __init__(self, net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True,
                 objective=DEFAULT_OBJECTIVE):
        self.net = net
        self.base_img = base_img
        self.iter_n = iter_n
        self.octave_n = octave_n
        self.octave_scale = octave_scale
        self.end = end
        self.clip = clip
        self.objective = objective

    def dream(self, **step_params):
        """
        Dream the little dream...i.e. send an image through the net associated with this Dream.
        :param step_params: Any of the parameters to the make_step method not needed anywhere else
        :return:
        """
        # prepare base images for all octaves
        octaves = [pil_to_caffe(self.net, self.base_img)]
        for i in range(self.octave_n - 1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0 / self.octave_scale, 1.0 / self.octave_scale), order=1))

        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

            src.reshape(1, 3, h, w)  # resize the network's input image size
            src.data[0] = octave_base + detail
            for i in range(self.iter_n):
                self.make_step(**step_params)

                # visualization
                vis = caffe_to_pil(self.net, src.data[0])
                if not self.clip:  # adjust image contrast if clipping is disabled
                    vis *= 255.0 / np.percentile(vis, 99.98)
                show_array(vis)
                print(octave, i, self.end, vis.shape)
                clear_output()

            # extract details produced on the current octave
            detail = src.data[0] - octave_base
        # returning the resulting image
        return caffe_to_pil(self.net, src.data[0])

    def make_step(self, step_size=1.5, jitter=32):
        """Basic gradient ascent step."""

        src = self.net.blobs['data']  # input image is stored in Net's 'data' blob
        dst = self.net.blobs[self.end]

        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

        self.net.forward(end=self.end)
        self.objective.objective(dst, self.net, self.end)  # specify the optimization objective
        self.net.backward(start=self.end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

        if self.clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255 - bias)
