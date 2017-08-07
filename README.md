# MultitaskDescriptor
Implementation of the Multitask Lasso Descriptor. 
To install the package download it and execute `python setup.py install`.

This branch proposes slight modifications from the [original version](https://github.com/vmolina/MultitaskDescriptor).

The package requires 'numpy', 'scikit-learn'.

The package contains two classes `Mumulader` and `RandomizedMumulader`. 
To train the model the method `optimize` must be called. The method for predicting is `predict`.
