# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:54:28 2018

@author: zhewei
"""
import numpy as np 

def batch_generator(X, y, X_len, batch_size):
    """Primitive batch generator 
    """
    size = len(X)
    X_copy = X.copy()
    y_copy = y.copy()
    X_len_copy = X_len.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    X_len_copy = X_len_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], X_len_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            X_len_copy = X_len_copy[indices]
            continue