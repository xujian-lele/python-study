'''
Created on Jun 2, 2015

@author: xujian
'''

import theano
import theano.tensor as T

a = T.dvector('a')
b = a + a ** 10
f = theano.function([a], b)
print f([0,1,2])

theano.printing.pydotprint(b, "/home/xujian/Desktop/1.png")