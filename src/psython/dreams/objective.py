from .. import caffe_helper


class IObjective:
    """
    Interface for an objective function to guide gradient ascent
    """

    def __init__(self):
        pass

    def objective(self, dst, net, end):
        """
        A function run on gradient ascent step
        :param dst: Magical caffe model data
        """
        pass


class L2Objective(IObjective):
    """
    L2 Norm
    """

    def __init__(self):
        IObjective.__init__(self)

    def objective(self, dst, net, end):
        dst.diff[:] = dst.data


class GuideObjective(IObjective):
    """
    Guide model with image.
    """

    def __init__(self, guide_image):
        IObjective.__init__(self)
        self.guide_image = guide_image

    def objective(self, dst, net, end='inception_3c/output'):
        x = dst.data[0].copy()
        y = self.guide_features(net, end)
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        a = x.T.dot(y)  # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch, -1)[:] = y[:, a.argmax(1)]  # select ones that match best

    def guide_features(self, net, end):
        """
        :return: The manipulated image data with which to guide the model
        """
        h, w = self.guide_image.shape[:2]
        src, dst = net.blobs['data'], net.blobs[end]
        src.reshape(1, 3, h, w)
        src.data[0] = caffe_helper.pil_to_caffe(net, self.guide_image)
        net.forward(end=end)
        return dst.data[0].copy()
