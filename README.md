# sltlib
sltlib contains functions for working with Empirical Risk Bounds of SVM

Created by Mikhail Ivanushko as part of a summer 2021 assignment @ HSE

## Features

- Approximate Rademacher complexity on arbitrary functions and datasets
- Approximate the 'Outlier' function as defined in "Foundations of Machine Learning", Mohri, Theorem 3.3.
- Save, load and re-use history of computation
- Various other utility functions for working with SVM

## Sub-modules

- **rademacher**: computing Rademacher complexity
- **outlier**: computing the 'Outlier' function
- **plot**: various plotting utilities
- **utils**: various utility functions (saving and loading files, etc)
- **basefuncs**: basic funcions that are shared by solvers (e.g. signed distance, margin loss)
