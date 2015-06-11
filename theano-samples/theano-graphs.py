'''
Created on Jun 2, 2015

@author: xujian
'''

import theano
from theano import *
import theano.tensor as T

x = T.dmatrix('x') 
y = x * 2

print y.owner.op.name
print len(y.owner.inputs)
print y.owner.inputs[0]
print y.owner.inputs[1]

print type(y.owner.inputs[1])
print type(y.owner.inputs[1].owner)
