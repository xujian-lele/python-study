'''
Created on Jun 1, 2015

@author: xujian
'''
import theano
import theano.tensor as T
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
f=theano.function([inc],state,updates=[(state, inc+state)])

print state.get_value()
print f(1)
print state.get_value()
print f(300)
print state.get_value()
state.set_value(-1)
print state.get_value()

#Do not use the shared variables temporary
fn_of_state = state * 2 + inc
#The type of foo must match the shared variable we are replacing
foo = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc, foo], fn_of_state,givens=[(state, foo)])
#we are using 3 for the state value
print skip_shared(1, 3)
#state's value is still here.
print state.get_value()

#random shared numbers
