'''
#Follow examples in theano.pdf, I do some excise
Created on Jun 1, 2015

@author: xujian
'''
import numpy
from theano import *
import theano

import theano.tensor as T


#print theano.config
x=numpy.asarray([[1., 2], [3, 4], [5, 6]])
print x

#broadcast
a=numpy.asarray([1,2,3])
b=2*a
print b

#Add to scalar
x=T.dscalar('x')
y=T.dscalar('y')
z=x+y
f=function([x,y],z)

print f(2,3)
print f(132.5,322.4)

print z.eval({x:1,y:2})

#Add two mitrix
x=T.dmatrix('x')
y=T.dmatrix('y')
z=x+y
f=function([x,y],z)

print f([[1,2],[1,2]],[[10,20],[10,20]])

