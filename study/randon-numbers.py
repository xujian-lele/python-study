'''
Created on Jun 2, 2015

@author: xujian
'''

import theano
from theano import *
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams()
srng.seed(902340)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))

f = function([],rv_u)
g = function([],rv_n,no_default_updates=True)

nearly_zero = function([], rv_u + rv_u - 2*rv_u)

f_va10 = f()
print f_va10
f_va11 = f()
print f_va11;
print "--------------------------"
g_va10 = g()
print g_va10
g_va11 = g()
print g_va11
print "--------------------------"

state_after_v0=rv_u.rng.get_value().get_state()
print "state_after_v0"
print state_after_v0
print "--------------------------"

nearly_zero()
state_after_v0=rv_u.rng.get_value().get_state()
print "state_after_v0"
print state_after_v0
print "--------------------------"

v1=f();
print v1
print "------------v1 is end--------------"
state_after_v0=rv_u.rng.get_value().get_state()
print "state_after_v0"
print state_after_v0
print "--------------------------"

rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng,borrow=True)
v2=f()
print v2
print "------------v2 is end--------------"
v3=f()
print v3
print "------------v3 is end--------------"
