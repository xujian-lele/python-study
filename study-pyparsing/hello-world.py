'''
Created on Jun 2, 2015

@author: root
'''
import pyparsing
from pyparsing import Word, alphas
from scipy.constants.constants import alpha

#"Construct the gramma"
greet = Word(alphas)+","+Word(alphas)
greeting = greet.parseString("hello, world")
print greeting
