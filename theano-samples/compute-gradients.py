'''
Created on Jun 2, 2015

@author: root
'''
import theano
from theano import pp
import theano.tensor as T
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
pp(gy)

f = theano.function([x], gy)
print f(4)