'''
Created on Jun 1, 2015

@author: xujian
'''
import theano
import theano.tensor as T

a,b=T.dmatrices('a','b')
diff = a-b
absdiff = abs(diff)
diff_squared = diff**2
f=theano.function([a,b], [diff,absdiff,diff_squared])

print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
