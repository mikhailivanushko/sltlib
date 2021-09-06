'''
    A list of basic functions used between solvers
'''

import numpy as np

def signed_distance(X, w):
    w /= np.linalg.norm(w)
    return np.dot(X, w)

def margin_loss(X, Y, w, margin):
    ml_val = 1 - ( (Y * signed_distance(X, w)) / margin )
    ml_val[ml_val<0] = 0
    ml_val[ml_val>1] = 1
    return ml_val
