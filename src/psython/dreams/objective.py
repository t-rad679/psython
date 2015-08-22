from .. import caffe_helper


class IObjective:
    """
    Interface for an objective function to guide gradient ascent
    """

    def __init__(self):
        pass

    def objective(self, dst):
        """
        A function run on gradient ascent step
        :param dst: Magical caffe model data
        """


class L2Objective(IObjective):
    """
    L2 Norm
    """

    def __init__(self):
        IObjective.__init__(self)

    def objective(self, dst):
        dst.diff[:] = dst.data


class ObjectiveGuide(IObjective):
    """
    Guide model with image.
    """

    def __init__(self, net, guide_image, end):
        IObjective.__init__(self)
        self.net = net
        self.guide_image = guide_image
        self.end = end

    def objective(self, dst):
        x = dst.data[0].copy()
        y = self.guide_features()
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        a = x.T.dot(y)  # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch, -1)[:] = y[:, a.argmax(1)]  # select ones that match best

    def guide_features(self):
        """
        :return: The manipulated image data with which to guide the model
        """
        h, w = self.guide_image.shape[:2]
        src, dst = self.net.blobs['data'], self.net.blobs[self.end]
        src.reshape(1, 3, h, w)
        src.data[0] = caffe_helper.pil_to_caffe(self.net, self.guide_image)
        self.net.forward(end=self.end)
        return dst.data[0].copy()
