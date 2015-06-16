import numpy 
from numpy import *
print 'this is a array'
array=random.rand(4,4)
print array

print 'this is a matrix'
randmat=mat(array)
print randmat
print randmat[0,0]
print type(randmat)

print 'this is a inv-matrix'
invrandmat=randmat.I
print invrandmat

print 'this is a T'
randmatT=randmat.T 
print randmatT

print 'this is result of randmat*invrandmat'
multimat=randmat*invrandmat
print multimat

print 'this is the example of eye'
myeye=eye(4,4)
print myeye