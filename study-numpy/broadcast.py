'''
Created on Jun 2, 2015

@author: root
'''
import numpy as np

a = np.arange(10,70,10).reshape(-1,1) #reshape(-1,1) transfer the matrix's size into 6*1 
print a

b=np.arange(1,6)#matrix's size is: 1*5
print b

c =a+b
print c