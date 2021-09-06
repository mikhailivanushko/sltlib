import numpy as np
import pandas as pd
import random as rd
import pickle

# Plotting lib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Models and optimizers
from scipy.optimize import dual_annealing
import cvxpy as cp

# Saving images
import os
from datetime import datetime

def save_file(name,file):
    with open(name, "wb") as fp:
        pickle.dump(file, fp)

def load_file(name):
    with open(name, "rb") as fp:
        file = pickle.load(fp)
    return file

class RDhistory():
    def __init__(self, model_type=None):
        self.model_type = model_type
        self.rademacher = list()
        self.hypothesis = list()
        self.correlation = list()
        self.model = list()

class MLC:
    def __init__(self, X, Y, basefun="sd", pbounds=None, path="history/mlc", radvec=None, margin=0.5,
                maxiter=100, initial_temp=5500, accept=-5, runs_per_sample=10):
        
        def _signed_distance(w):
            w /= np.linalg.norm(w)
            return np.dot(self.X, w)

        if (basefun == "sd"):
            self.basefun = _signed_distance
            self.pbounds = [(-1, 1) for x in range(X.shape[1])]
        else:
            self.basefun = basefun
            self.pbounds = pbounds
            
        self.path = path
        
        self.margin = margin
        
        self.X = X
        self.Y = Y
        self.radvec = radvec
        
        self.maxiter = maxiter
        self.initial_temp = initial_temp
        self.accept = accept
        self.runs_per_sample = runs_per_sample
        
        self.history = RDhistory('margin_loss')
    
    def ml(self, params):
        ml_val = 1 - ( (self.Y * self.basefun(params)) / self.margin )
        ml_val[ml_val<0] = 0
        ml_val[ml_val>1] = 1
        return ml_val
    
    def mlc(self, params, args):
        ml_val = self.ml(params)
        return - np.sum(ml_val * self.radvec) # negate correlation to create minimization problem
    
    def solve(self):
        st_bounds = self.pbounds
        result = dual_annealing(
            func=self.mlc,
            bounds=self.pbounds,
            args=([],),
            maxiter=self.maxiter,
            initial_temp=self.initial_temp,
            accept=self.accept
        )
        return result.x
    

    def pump(self):
        data_size = self.X.shape[0]

        if self.radvec is None:
            self.radvec = np.array([rd.randint(0, 1) * 2 - 1 for x in range(data_size)])

        sample_hypothesis = []
        correlations = []
        accuracies = []
        models = []

        for j in range(self.runs_per_sample):
                
            # use Stochastic method
            model = self.solve()
            pred = self.ml(model)
            models.append(model)

            # got prediction
            sample_hypothesis.append(pred)
            correlations.append(np.sum(self.radvec * pred))

        best_run_index = correlations.index(max(correlations))
        best_pred = sample_hypothesis[best_run_index]

        print('MLC', max(correlations), end=' ', sep='')

        self.history.rademacher.append(self.radvec)
        self.history.hypothesis.append(best_pred)
        self.history.correlation.append(correlations)
        self.history.model.append(models[best_run_index])
        
        self.radvec = None
    
    def load(self):
        self.history = load_file(self.path)
    
    def save(self):
        save_file(self.path, self.history)

class CONF:
    
    def __init__(self, X, Y, path="history/conf", radvec=None):

        def _signed_distance(w):
            w /= np.linalg.norm(w)
            return np.dot(self.X, w)
        
        self.path = path

        self.basefun = _signed_distance

        self.X = X
        self.Y = Y

        self.radvec = radvec
        
        self.history = RDhistory('confidence')

    def pump(self):
        data_size = self.X.shape[0]

        if self.radvec is None:
            self.radvec = np.array([rd.randint(0, 1) * 2 - 1 for x in range(data_size)])

        sample_hypothesis = []
        correlations = []
        accuracies = []
        models = []


        # Linear problem, use analytical solution
        X_signed = np.transpose(np.transpose(self.X)*self.radvec)
        solution = np.mean(X_signed, axis=0)
        # solution[-1] = self.int_bound * np.sign(np.sum(self.radvec))
        pred = self.basefun(solution)
        print('L', np.mean(self.radvec * pred), end=' ', sep='')
        models.append(solution)


        # got prediction
        sample_hypothesis.append(pred)
        correlations.append(np.sum(self.radvec * pred))

        best_run_index = correlations.index(max(correlations))
        best_pred = sample_hypothesis[best_run_index]

        self.history.rademacher.append(self.radvec)
        self.history.hypothesis.append(best_pred)
        self.history.correlation.append(correlations)
        self.history.model.append(models[best_run_index])
        
        self.radvec = None
    
    def load(self):
        self.history = load_file(self.path)
    
    def save(self):
        save_file(self.path, self.history)

class OUT:
    def __init__(self, X, X_masked, Y, Y_masked, basefun="sd", pbounds=None, path="history/out", margin=0.5, 
                    maxiter=100, initial_temp=5500, accept=-5):
        
        def _signed_distance(w, masked=False):
            w /= np.linalg.norm(w)
            if not masked:
                return np.dot(self.X, w)
            else:
                return np.dot(self.X_masked, w)

        if (basefun == "sd"):
            self.basefun = _signed_distance
            self.pbounds = [(-1, 1) for x in range(X.shape[1])]
        else:
            self.basefun = basefun
            self.pbounds = pbounds

        self.path = path
        
        self.margin = margin
        
        self.X = X
        self.Y = Y

        self.X_masked = X_masked
        self.Y_masked = Y_masked
        
        self.maxiter = maxiter
        self.initial_temp = initial_temp
        self.accept = accept
        
        self.history = list()
    
    def ml(self, params, masked=False):
        if not masked:
            ml_val = 1 - ( (self.Y * self.basefun(params, masked)) / self.margin )
        else:
            ml_val = 1 - ( (self.Y_masked * self.basefun(params, masked)) / self.margin )
        ml_val[ml_val<0] = 0
        ml_val[ml_val>1] = 1
        return ml_val

    def outlier(self, params, args):
        mloss1 = self.ml(params, masked=False)
        mloss2 = self.ml(params, masked=True)
        N = self.X.shape[0]         # set size
        M = self.X_masked.shape[0]  # mask size
        return - ( (np.sum(mloss1) / N) - (np.sum(mloss2) / M) ) # negated result (to maximize it using a minimizer)
    
    def solve(self):
        result = dual_annealing(
            func=self.outlier,
            bounds=self.pbounds,
            args=([],),
            maxiter=self.maxiter,
            initial_temp=self.initial_temp,
            accept=self.accept
        ) # This is a minimizer, but the self.outlier function result is negated
        return result

    def pump(self, lin=False):
        res = self.solve()
        self.history.append(-res.fun)
    
    def load(self):
        self.history = load_file(self.path)
    
    def save(self):
        save_file(self.path, self.history)
