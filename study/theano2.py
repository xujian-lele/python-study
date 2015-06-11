'''
Created on Jun 1, 2015

@author: xujian
'''
#Exerciese
import theano
from theano import *

a=theano.tensor.vector()
out=a+a**10
f=theano.function([a],out)
print f([1,2,3,4])