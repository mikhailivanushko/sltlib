import numpy as np
import pandas as pd

from scipy.optimize import dual_annealing
import cvxpy as cp

from sltlib.utils import save_file, load_file
from sltlib.basefuncs import signed_distance, margin_loss

'''
    Base class for approximating the 'Outlier' function

    X, Y:           The entire distribution and its binary labels
    sample_size:    The number of points to sample from the whole distribution
    path:           Save / Load filepath for history of calculation
    solver_args:    Arguments for the dual_annealing solver: a dict
                    containing 'maxiter', 'initial_temp' and 'accept'
'''
class Base(object):
    def __init__(
                self, X, Y, sample_size=100,
                path="history/out",
                solver_args={ 'maxiter':100, 'initial_temp':5500, 'accept':-5 }
                ):

        self.path = path
        self.X_full = X
        self.Y_full = Y
        self.X = None
        self.Y = None
        self.sample_size = sample_size
        self.solver_args = solver_args
        self.history = list()
        self.bounds = None
    
    '''
        Samples a subset of size "sample_size" from the whole distibution,
        saving the result to self.X and self.Y

        You can call this function manually and store the resulting subset
        for later use in other solvers.
    '''
    def _sample_set(self):
        self.X = pd.DataFrame(self.X_full).sample(self.sample_size, replace=False)
        self.Y = self.Y_full[self.X.index]

    ''' Base function; To be overriden by children of this class '''
    def _basefunc(self, X, Y, params):
        raise NotImplementedError


    '''
        The 'Outlier' function. The result is negated since we are using a minimizer.
        
        params:     The parameters of the base function to be evaluated,
                    for example, a vector that defines a hyperplane.
        args:       ** Unused, just required by scipy.optimize solvers
    '''
    def _outlier(self, params, args):
        res_full = self._basefunc(self.X_full, self.Y_full, params)
        res_masked = self._basefunc(self.X, self.Y, params)
        N = self.X_full.shape[0]        # set size
        M = self.X.shape[0]             # mask size

        # result is negated!
        return - ( (np.sum(res_full) / N) - (np.sum(res_masked) / M) ) 
    
    ''' The solver method; Calling dual_annealing on the 'Outlier' function. '''
    def _solve(self):
        result = dual_annealing(
            func=self._outlier,
            bounds=self.bounds,
            args=([],),
            maxiter=self.solver_args['maxiter'],
            initial_temp=self.solver_args['initial_temp'],
            accept=self.solver_args['accept']
        )
        return result

    '''
        The driver function. Calculates one term in the expected value
        of the Outlier function, and stores that result in the calculation history.

        resample:   If True, a new subset is sampled before solving. If you want to use
                    a predefined subset (for example, when 'synchronizing' multiple solvers),
                    then you should make this False and assign the self.X and self.Y variables
                    manually before calling calc()
    '''
    def calc(self, resample=True, verbose=False):
        if resample: self._sample_set()
        res = self._solve()
        self.history.append(-res.fun)
        if verbose: self._verbose_result(-res.fun)

    ''' Verbose output format function: computed result '''
    def _verbose_result(self, res):
        print('::', res, sep='')
    
    ''' Try to load the calculation history at self.path '''
    def load(self):
        try:
            self.history = load_file(self.path)
        except:
            print("Failed to load history")
    
    ''' Save the calculation history at self.path '''
    def save(self):
        save_file(self.path, self.history)


'''
   A class for calculating the 'Outlier' function for Margin Loss (linear case).
   Inherits parameters from the Base class.

   margin:          The margin in the Margin Loss.
'''
class LML(Base):
    def __init__(self, margin=0.5, **kwds):
        super().__init__(**kwds)
        self.bounds = [(-1, 1) for x in range(self.X_full.shape[1])]
        self.margin = margin

    ''' Margin Loss is the base function '''
    def _basefunc(self, X, Y, params):
        return margin_loss(X, Y, params, self.margin)

    ''' Verbose output format function: computed result '''
    def _verbose_result(self, res):
        print('LML::', res, sep='')