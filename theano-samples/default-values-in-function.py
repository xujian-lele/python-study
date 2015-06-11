'''
Created on Jun 1, 2015

@author: xujian
'''

import theano
from theano import Param
import theano.tensor as T
from samba.dcerpc.atsvc import Third

a,b,c = T.dscalars('a','b','c')
z=(a+b)*c
f=theano.function([a,Param(b,default=0),Param(c,default=1,name="third_var")],z)
print f(1)
print f(1,2)
print f(1,third_var=2)