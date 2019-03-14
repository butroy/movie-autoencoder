#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:28:10 2019

@author: yifengbu
"""

from sklearn.model_selection import train_test_split
import numpy as np

file_name = "ratings.dat"
with open(file_name) as fn:
    content = fn.readlines()
    
train,test = train_test_split(content,test_size=0.2)

train_file = open('train.dat','w')
for line in train:
    train_file.write(line)
    train_file.write('\n')
train_file.close()


test_file = open('test.dat','w')
for line in test:
    test_file.write(line)
    test_file.write('\n')
test_file.close()
