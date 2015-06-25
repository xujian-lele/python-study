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

#multiply outputs
a,b=T.matrices('a','b')
diff=a-b
abs_diff = abs(diff)
diff_squared = diff ** 2
f=theano.function([a,b],[diff,abs_diff,diff_squared])
d,e,f=f([[1,2,3],[10,20,30]],[[100,200,300],[1,2,3]])
print "diff is : \n"
print d
print "abs_diff is : \n"
print e
print "diff_squared is : \n"
print f