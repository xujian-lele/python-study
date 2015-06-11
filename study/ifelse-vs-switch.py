'''
Created on Jun 2, 2015

@author: root
'''

from theano import tensor as T
from theano.ifelse import ifelse
import theano
import numpy
import time
from sympy.printing.tests.test_theanocode import theano
from theano.gof.link import Linker

a,b = T.scalars('a','b')
x,y = T.matrices('x','y')

z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = theano.function([a,b,x,y], z_switch, mode=theano.Mode(linker='vm'))
f_lazyifelse = theano.function([a,b,x,y], z_lazy, mode=theano.Mode(linker='vm'))

val1 = 0.
val2 = 1.
big_mat1 = numpy.ones((1000, 1000))
big_mat2 = numpy.ones((1000, 1000))

n_times = 10

tic = time.clock()
for i in xrange(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print 'time spent evaluating both values %f sec' % (time.clock() - tic)

tic = time.clock()
for i in xrange(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print 'time spnet evaluating both values %f sec' % (time.clock() - tic)