__author__ = 'tradical'
from psython.dreams import dream, util, objective

# build the Caffe model
net = dream.load_dnn_model()

# load images
bf = util.create_image('res/images/sky1024px.jpg')
ic = util.create_image('res/images/bluemorpho_scaled.jpg')

# still not sure what this is about; something about octaves
end = 'inception_3b/3x3_reduce'

l2_d = dream.Dream(net, bf, objective=objective.L2Objective()) # l2 norm objective dream
# dream.show_array(l2_d.dream())

go = objective.GuideObjective(ic)
guide_d = dream.Dream(net, bf, objective=go)

dream.show_array(guide_d.dream())
