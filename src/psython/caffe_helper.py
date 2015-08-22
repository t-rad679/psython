import numpy as np

# a couple of utility functions for converting to and from Caffe's input image layout


def pil_to_caffe(net, img):
    """
    Converts a PIL Image to Caffe's input format
    :param net: The net that will process the image after conversion
    :param img: The image, in PIL format
    :return: The image, in Caffe's native tongue
    """
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def caffe_to_pil(net, img):
    """
    Converts an image in Caffe's image data format, and converts it into a PIL Image
    :param net: The net that processed this image
    :param img: The image, in Caffe's native tongue
    :return: The image, in PIL format
    """
    return np.dstack((img + net.transformer.mean['data'])[::-1])
