from __future__ import division

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import sys


def MSE(y_te, y_pred):
    """
    Computes mean square error
    """
    return np.sqrt(mean_squared_error(y_te, y_pred))


class MuMuLaDer(object):
    """
    Implements the MMLD approach
    (see https://psb.stanford.edu/psb-online/proceedings/psb16/bellon.pdf)

    Parameters:
    -----------
    descriptors : array, shape (n_tasks, n_descriptors)
        feature vector for each task.
    lambda_1 : float, optional
        Constant that multiplies the L1 term when optimizing alpha.
        Defaults to 1.0.
    lambda_1 : float, optional
        Constant that multiplies the L1 term when optimizing theta.
        Defaults to 1.0.
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    fit_task_intercept : boolean
        whether to consider the intercept for task features
        for this model.
    loss_f: 'MSE' only
        To minimize in cross_validation.
    max_iter : int, optional
        The maximum number of iterations
    max_inner_iter : int, optional
        The maximum number of iterations when running 'lasso' from sklearn
    tol : float, optional
        The tolerance for the optimization: if the updates in beta is
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    random_init : bool
        Considered only if warm_start=True.
        When set to true, alpha is initialized by sampling a normal
        distribution instead of being all zeros.
    alpha_params : dictionary containing the parameters
        for the alpha optimizer
    theta_params : dictionary containing the parameters
        for the theta optimizer

    Attributes
    ----------
    coef_ : array, shape (n_tasks, n_features)
        parameter vector (w in the cost function formula).
        Equals to beta.T in MMLD formulation.
    alpha_ : array, shape (n_feature,n_descriptors)
        alpha parameter
    theta_ : array, shape (1,n_feature)
        theta parameter
    n_iter_ : real value
        number of iterations run by the fit method to reach
        the specified tolerance.

    Methods
    -------
    fit: Fit model.
    predict: Predict using the MMLD model
    get_params: Get parameters for this estimator.
    get_beta: Computes beta based on theta, alpha and
        the task descriptors matrix

    Notes
    -----


    """

    def __init__(
            self, descriptors, lambda_1=1, lambda_2=1, fit_intercept=False,
            fit_task_intercept=False, loss_f=MSE,
            max_iter=100, max_inner_iter=10, tol=0.0000001, warm_start=False,
            random_init=True, alpha_params=None, theta_params=None):

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept
        self.fit_task_intercept = fit_task_intercept
        self.loss_f = loss_f
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_init = random_init

        self.theta_params = {} if theta_params is None else theta_params
        self.theta_params['alpha'] = lambda_1
        self.theta_params['warm_start'] = True
        self.theta_params['positive'] = True
        self.theta_params['max_iter'] = max_inner_iter

        self.alpha_params = {} if alpha_params is None else alpha_params
        self.alpha_params['alpha'] = lambda_2
        self.alpha_params['warm_start'] = True
        self.alpha_params['max_iter'] = max_inner_iter

        self.descriptors = np.concatenate(
            (descriptors,
                np.ones((descriptors.shape[0], 1.),
                        dtype=descriptors.dtype)),
            axis=1) if self.fit_task_intercept else descriptors

        self.coef_ = None
        self.alpha_ = None
        self.theta_ = None
        self.n_iter_ = None

    def fit(self, X, Y, tasks, verbose=False):
        """
        It fits the model. Updates alpha and theta and then beta (coef_)

        Parameters:
        -----------
        X: array, shape (n_samples, n_features)
             design matrix
        Y: array, shape (1,n_samples)
            target
        tasks: array, shape (n_samples, n_tasks)
            every row contains just one element 1 and 0 for the other
            numbers to indicate to which task the sample belongs.
        verbose: bool
            print optimization steps.

        """
        if self.lambda_1 == 0 or self.lambda_2 == 0:
            raise "inner lasso penalty is set to 0"

        X_copy = np.concatenate(
            (X, np.ones((X.shape[0], 1), dtype=X.dtype)),
            axis=1) if self.fit_intercept else X

        # Initialize theta and alpha optimizers
        optimize_theta = Lasso(**self.theta_params)
        optimize_alpha = Lasso(**self.alpha_params)

        # get input sizes and initizialize parameters
        # n is the total amount of instances among all tasks
        # p is number of features
        n, p = X_copy.shape
        # T is the number of task
        T = tasks.shape[1]
        # L is number of task descriptor
        L = self.descriptors.shape[1]

        # initialize theta, alpha and beta
        if not self.warm_start or self.coef_ is None:
            theta = np.ones((1, p), dtype=np.float64)
            if self.random_init:
                alpha = np.random.normal(size=(p, L))\
                    .astype(np.float64)
            else:
                alpha = np.zeros((p, L), dtype=np.float64)
                optimize_alpha.coef_ = alpha.flatten()
            beta_0 = self.get_beta(alpha, theta)
        else:
            beta_0 = self.coef_.T
            alpha = self.alpha_
            theta = self.theta_

        # Precalculate x_alpha matrix for optimizing alpha
        # It is a (n, p*L) where xD_{i,(j-1)L+l) = \theta_{j}d_l
        # for the corresponding task.
        # In the algorithm 3.1 in the paper, w = \theta * x_alpha
        repeated_descriptors = np.repeat(self.descriptors, p, axis=0)
        repeated_descriptors.shape = (T, L * p)
        x_alpha = tasks.dot(repeated_descriptors)
        x_alpha = x_alpha * np.repeat(X_copy, L, axis=1)
        del repeated_descriptors

        # initilize x_theta: refering to "z" in the algorithm 3.1
        x_theta = tasks.dot(self.descriptors.dot(alpha.T)) \
            * X_copy

        # Start the two phase optimization
        continue_optimization = True
        self.n_iter_ = 0
        while continue_optimization:
            if verbose:
                print('n_iter:%d' % (self.n_iter_))
                print('\toptimize alpha')
                sys.stdout.flush()
            # Optimize for alpha and update alpha
            X_alpha = np.repeat(theta, L, axis=1) * x_alpha
            alpha[:] = optimize_alpha.fit(X_alpha, Y).coef_\
                .reshape((p, L))

            if verbose:
                print('\toptimize theta')
                sys.stdout.flush()
            # Optimize for theta and update theta
            x_theta[:] = tasks.dot(self.descriptors.dot(alpha.T)) * X_copy
            theta[:] = optimize_theta.fit(x_theta, Y).coef_

            if verbose:
                print('\tupdate beta')
                sys.stdout.flush()
            # update beta: \beta = \theta * (D.\alpha^T)
            beta = self.get_beta(alpha, theta)

            self.n_iter_ += 1
            if (np.linalg.norm(beta.flatten() - beta_0.flatten()) <
                    self.tol):
                continue_optimization = False
            else:
                beta_0 = beta
            if self.n_iter_ >= self.max_iter and continue_optimization is True:
                continue_optimization = False
                warnings.warn("warning: exit optimization because number " +
                              "of iterations exceeded maximum " +
                              "number of allowed iterations")
        self.coef_ = beta.T
        self.alpha_ = alpha
        self.theta_ = theta

    def get_beta(self, alpha=None, theta=None):
        """
        Computes beta based on theta, alpha and the task descriptors
        matrix.
        \beta = \theta * (D.\alpha^T)
        """
        alpha = self.alpha_ if alpha is None else alpha
        theta = self.theta_ if theta is None else theta
        return theta * self.descriptors.dot(alpha.T)

    def predict(self, X, tasks):
        """
        Predict using the linear model.
        """
        new_X = np.concatenate(
            (X, np.ones((X.shape[0], 1), dtype=X.dtype)),
            axis=1) if self.fit_intercept else X

        beta = self.coef_.T
        return np.sum(tasks.dot(beta) * new_X, axis=1)

    def get_params(self):
        return {name: self.__dict__[name]
                for name in self.__dict__.keys() if name[-1] != '_'}
