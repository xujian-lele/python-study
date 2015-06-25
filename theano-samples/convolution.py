'''

@author: xujian

Function: convolution options of two pictures with the same size(width,height)
input:    3 features maps(3 channels <RGB> of a picture)
convolution: two 9*9 concolution filters
'''

from theano.tensor.nnet import conv
import theano.tensor as T
import theano ,numpy
from theano.gof.tests.test_cc import inputs

rng = numpy.random.RandomState(23455)

#symbol variable
input = T.tensor4(name='input')
print input.dtype

#initial weights
w_shape = (2,3,9,9) #2 convolution filters, 3 channels, filter shape : 9*9
w_bound = numpy.sqrt(3*9*9)
W= theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound, size = w_shape), dtype = input.dtype),name = 'W')
print W

b_shape = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape),dtype = input.dtype),name = 'b')
print b
conv_out = conv.conv2d(input,W)

ouput = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
f = theano.function([input], ouput)

#demo
import pylab,numpy
from PIL import Image
from matplotlib.pyplot import *

#------------img1------------
img1 = Image.open('../data/00.jpg')
width1, height1 = img1.size
print width1
print height1
img1 = numpy.asarray(img1, dtype='float32')/256.

#put the image in 4D tensor of shape(1,3,height,width)
img1_rgb = img1.swapaxes(0,2).swapaxes(1,2) #(3,height,width) 
minibatch_img = img1_rgb.reshape(1,3,height1,width1) #(1,3,height,width)
filtered_img = f(minibatch_img)

#plot origin image and two convolution results
pylab.figure(1)
pylab.subplot(1,3,1);pylab.axis('off')
pylab.imshow(img1)
title('origin image')

pylab.gray()
pylab.subplot(2,3,2);pylab.axis('on')
pylab.imshow(filtered_img[0,0,:,:])#0:minibatch_index;0:1-st filter
title('convolution 1')

pylab.subplot(2,3,3);pylab.axis('on')
pylab.imshow(filtered_img[0,1,:,:])
title('convolution 2')

#pylab.show()

# maxpooling
from theano.tensor.signal import downsample

input = T.tensor4(name='input')
maxpool_shape = (2,2)
pooled_img = downsample.max_pool_2d(input, maxpool_shape)
maxpool = theano.function(inputs=[input],outputs=[pooled_img])
print "filtered_img.shape: %s"
print filtered_img.shape
pooled_result = maxpool(filtered_img)
print pooled_img.shape

pooled_res = np.squeeze(maxpool(filtered_img))
print pooled_res.shape

pylab.subplot(2,3,5)
pylab.axis('on')
pylab.imshow(pooled_res[0,:,:])
title('downsample 1')

pylab.subplot(2,3,6)
pylab.axis('on')
pylab.imshow(pooled_res[1,:,:])
title('downsample 2')
pylab.show()