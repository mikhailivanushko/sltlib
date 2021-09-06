import numpy as np
import pandas as pd
import random as rd

from scipy.optimize import dual_annealing
import cvxpy as cp

from sltlib.utils import save_file, load_file
from sltlib.basefuncs import signed_distance, margin_loss

''' 
    Base class for approximating Rademacher complexity

    X, Y:           The entire distribution and its binary labels
    sample_size:    The number of points to sample from the whole distribution
    path:           Save / Load filepath for history of calculation
    solver_args:    Arguments for the dual_annealing solver: a dict
                    containing 'maxiter', 'initial_temp' and 'accept'
    balanced:       Whether to use a modification of Rademacher complexity, wherein the
                    rademacher vectors have a balanced number of positive/negative labels.
'''
class Base(object):
    
    def __init__(
        self, X, Y, sample_size=100,
        path="history/solve",
        solver_args={ 'maxiter':100, 'initial_temp':5500, 'accept':-5 },
        balanced=False,
        ):

        self.path = path
        self.X_full = X
        self.Y_full = Y
        self.X = None
        self.Y = None
        self.sample_size = sample_size
        self.solver_args = solver_args
        self.history = {'rademacher':[], 'hypothesis':[], 'correlation':[], 'model':[]}
        self.balanced = balanced
        self.bounds = None

    ''' 
        Generates and returns rademacher vector. If self.balanced is True,
        the resulting vector has a balanced number of positive/negative labels.
    '''
    def _gen_radvec(self):
        if self.balanced:
            radvec = np.zeros(self.sample_size)
            radvec[:(self.sample_size//2)] = 1
            radvec = (radvec * 2) - 1
            np.random.shuffle(radvec)
            return radvec
        else:
            return np.array([rd.randint(0, 1) * 2 - 1 for x in range(self.sample_size)])

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
    def _basefunc(self, params):
        raise NotImplementedError

    ''' 
        Correlation of the base function and the rademacher vector.
        The result is negated since we are using a minimizer.
    '''
    def _basefunc_correlation(self, params, args):
        # result is negated!
        return -np.sum(self._basefunc(params) * self.radvec)

    ''' 
        The solver method; Calling dual_annealing to maximize
        the correlation between the Rademacher vector and the base function.
    '''
    def _solve(self):
        result = dual_annealing(
            func=self._basefunc_correlation,
            bounds=self.bounds,
            args=([],),
            maxiter=self.solver_args['maxiter'],
            initial_temp=self.solver_args['initial_temp'],
            accept=self.solver_args['accept']
        )
        return result.x


    ''' The driver function. Calculates one term in the expected value
        of the Rademacher complexity, and stores that result in the calculation history.
        
        radvec:     Optional; Use this if you want to provide the Rademacher vector manually.
                    If None, then a brand new rademacher vector is generated and used.
        nruns:      The number of times to run the solve on a given Rademacher vector.
                    If greater than 1, then only the best-correlating solution is stored
                    in the calculation history.
        resample:   If True, a new subset is sampled before solving. If you want to use
                    a predefined subset (for example, when 'synchronizing' multiple solvers),
                    then you should make this False and assign the self.X and self.Y variables
                    manually before calling calc()
        verbose:    If true, prints out the best correlation after the solve.
    '''
    def calc(self, radvec=None, nruns=1, resample=True, verbose=False):
        
        if resample: self._sample_set()
        if radvec is not None: self.radvec = radvec
        else: self.radvec = self._gen_radvec()

        hypotheses = []
        correlations = []
        models = []

        for j in range(nruns):
            model = self._solve()
            hypo = self._basefunc(model)
            models.append(model)

            hypotheses.append(hypo)
            correlations.append(np.sum(self.radvec * hypo))

        best_run_index = correlations.index(max(correlations))

        if verbose: self._verbose_correlations(correlations)

        self.history['rademacher'].append(self.radvec)
        self.history['hypothesis'].append(hypotheses[best_run_index])
        self.history['correlation'].append(correlations[best_run_index])
        self.history['model'].append(models[best_run_index])
    
    ''' Verbose output format function: best computed correlation. '''
    def _verbose_correlations(self, correlations):
        print('::', max(correlations), sep='')

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
    A class for calculating Rademacher complexity of linear confidence.
    Inherits parameters from the 'Radsolver' class.
'''
class Linear(Base):
    
    def __init__(self, **kwds):
        super().__init__(**kwds)

    ''' Base function overriden by signed distance '''
    def _basefunc(self, params):
        return signed_distance(self.X, params)

    '''
        Since this is a linear problem, get the analytical solution
        instead of using the usual solver
    '''
    def _solve(self):
        X_signed = np.transpose(np.transpose(self.X)*self.radvec)
        return np.mean(X_signed, axis=0)

    ''' Verbose output format function: best computed correlation. '''
    def _verbose_correlations(self, correlations):
        print('LIN::', max(correlations), sep='')

''' 
    A class for calculating Rademacher complexity of for Margin Loss (linear case).
    Inherits parameters from the Base class.

    margin:          The margin in the Margin Loss.
'''
class LML(Base):

    def __init__(self,
                margin=0.5,
                **kwds):
        super().__init__(**kwds)
        self.bounds = [(-1, 1) for x in range(self.X_full.shape[1])]
        self.margin = margin

    ''' Margin Loss of linear signed distance is the base function '''
    def _basefunc(self, params):
        return margin_loss(self.X, self.Y, params, self.margin)

    ''' Verbose output format function: best computed correlation. '''
    def _verbose_correlations(self, correlations):
        print('LML::', max(correlations), sep='')
