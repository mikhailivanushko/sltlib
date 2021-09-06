import pickle
import numpy as np

''' Saving and loading files '''

def save_file(path,file):
    with open(path, "wb") as fp:
        pickle.dump(file, fp)

def load_file(path):
    with open(path, "rb") as fp:
        file = pickle.load(fp)
    return file

'''
    Calculate rademacher complexity from history
    
    history - the dict object
    dist - also return the distribution of correlations (useful for plotting)

'''

def calc_complexity(history, dist=False):
    samples = len(history['hypothesis'])
    data_size = len(history['hypothesis'][0])

    rad_vectors = history['rademacher']
    hypotheses = history['hypothesis']

    if (len(rad_vectors) != samples):
        rad_vectors = rad_vectors[:samples]
    
    assert(len(rad_vectors) == len(hypotheses))
    
    complexity = []

    for s in range(samples):
        complexity.append(np.sum(rad_vectors[s] * hypotheses[s]))

    hypothesis_complexity = sum(complexity) * (1/samples) * (1/data_size)
    
    if dist:
        return hypothesis_complexity, np.array(complexity)
    else:
        return hypothesis_complexity