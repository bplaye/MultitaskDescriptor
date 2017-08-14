# MultitaskDescriptor
Implementation of the Multitask Lasso Descriptor. 
To install the package download it and execute `python setup.py install`.

This branch proposes modifications from the [original version](https://github.com/vmolina/MultitaskDescriptor) which are:
    - cleaning and commenting the code
    - using `scikit-learn` syntax
    - TO DO: an other implementation of `MMLD.fit()` method requiring less RAM and using parralel programming (regressions are 
    done on each task individually instead of all at once).
    - TO DO: 

The package requires 'numpy', 'scikit-learn'.

The package contains two classes `Mumulader` and `RandomizedMumulader`. 
To train the model the method `optimize` must be called. The method for predicting is `predict`.
