import numpy as np
from MultitaskDescriptor.MMLD import *


class RandomizedMuMuLaDer(object):
    """
    Implements the randomized-MMLD feature scoring approach.
    (see "thesis is not online yet")

    Parameters:
    -----------
    descriptors : array, shape (n_tasks, n_descriptors)
        feature vector for each task.
    MMLD_params: dict, all parameters used by MMLD except descriptors.

    Attributes
    ----------
    feature_importance_ = array, shape (n_tasks, n_features)
    """

    def __init__(self, descriptors, MMLD_params={}):
        """
        Reminder, default parameter values for MMLD:
        {'lambda_1': 1, 'lambda_2': 1, 'fit_intercept': False,
        'fit_task_intercept': False, optimizer='spams', loss_f': MSE,
        'max_ite': 100, 'max_inner_iter': 10, 'tol': 0.01,
        'warm_start': False, 'random_init': False,
        'alpha_params': None, 'theta_params': None}
        """
        self.MMLD_params = MMLD_params
        self.MMLD_params['descriptors'] = descriptors
        self.feature_importance_ = None

    def fit(self, X, Y, tasks, n_steps=100, scaling=0.1, verbose=False):
        """
        Computes the feature_importance matrix (array, (n_samples,n_tasks))
        which is the selection frequency of each feature for each task.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            design matrix
        Y : array, shape (1,n_samples)
            target
        tasks : array, shape (n_samples, n_tasks)
            every row contains just one element 1 and 0 for the other
            numbers to indicate to which task the sample belongs.
        n_steps : int, number of run of the random procedure
        verbose: bool
            print run steps.
        """
        n, p = X.shape
        _, k = tasks.shape
        self.feature_importance_ = np.zeros((p, k))

        # repeat n_steps time the 'random procedure', i.e. the MMLD
        # approach on bootstrapped instances and reweighted features
        for b in range(n_steps):
            if verbose:
                print('n_step:%d' % (b))
                sys.stdout.flush()
            # bootstraping instances
            boots1 = np.random.choice(np.arange(n), size=n,
                                      replace=True)
            # uniform sampling for reweighting features
            W = np.random.uniform(low=scaling,
                                  high=1.0, size=(1, p))

            # get modified X, Y and tasks for the randomized approach
            new_X = X[boots1, :] * W
            new_Y = Y[boots1, :]
            new_tasks = tasks[boots1, :]

            # run MMLD on the randomized X and Y
            mmld = MuMuLaDer(**self.MMLD_params)
            mmld.fit(new_X, new_Y, new_tasks, verbose=verbose)

            # update feature importance
            self.feature_importance_ += (mmld.coef_ != 0) / float(n_steps)

    def transform(self, X, threshold, ind_task):
        return X[:, self.feature_importance_[:, ind_task] > threshold]

    def fit_transform(self, X, Y, tasks, threshold, ind_task,
                      n_steps=100, scaling=0.1, verbose=False):
        self.fit(X, Y, tasks, n_steps, scaling, verbose=verbose)
        return self.transform(X, threshold, ind_task)
