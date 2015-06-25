# DimShuffle([False,False],(1,0))
	# if the input is a matrix whose shape is x*y,DimShuffle will change its shape to y*x
# DimShuffle([False,False],(1,'x',0))
	# if the input is a matrix whose shape is x*y,DimShuffle will change its shape to y*1*x

import theano,numpy
import theano.tensor as T

x=T.matrix('x')	#x.shape is m*n
y=T.matrix('y')
y=x.dimshuffle((1,0))
f=theano.function([x],y)

x=mat([[1,2,3],[2,3,4]])
print 'x dimension is : '
print x.shape()
y=f(x)
print 'y dimension is : '
print y.shape()