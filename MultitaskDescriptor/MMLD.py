from __future__ import division

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.simplefilter('error', UserWarning)

LASSO = 'Lasso'
LASSOLARS = 'LassoLars'
DEFAULT_OPTIMIZER = LASSO


def MSE(y_te, y_pred):
    """
    Computes mean square error
    """
    return np.sqrt(mean_squared_error(y_te, y_pred))


def assign_task(n, T, task):
    """
    if number of instances per task remains the same, it fills
    automatically the 'instance to task assignment' matrix
    (called 'task' here) per increasing task index
    """
    for i in range(T):
        task[i * n:(i + 1) * n, i] = 1.0


class MuMuLaDer():
    """

    Parameters:
    -----------
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
    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.
    alpha_params: dictionary containing the parameters
        for the alpha optimizer
    theta_params: dictionary containing the parameters
        for the theta optimizer

    Attributes
    ----------
    coef_ : array, shape (n_tasks, n_features)
        parameter vector (w in the cost function formula)
    sparse_coef_ : scipy.sparse matrix, shape (n_tasks, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``
    intercept_ : float | array, shape (n_tasks,)
        independent term in decision function.
    n_iter_ : real value
        number of iterations run by the fit method to reach
        the specified tolerance.

    Methods
    -------

    Notes
    -----


    """

    def __init__(
            self, lambda_1=1, lambda_2=1, fit_intercept=False,
            fit_task_intercept=False, loss_f=MSE,
            max_iter=100, max_inner_iter=10, tol=0.01, warm_start=False,
            random_init=False, alpha_params=None, theta_params=None):

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

        self.coef_ = None
        self.alpha_ = None
        self.theta_ = None
        self.intercept_ = 0.0
        self.n_iter_ = None

    def fit(self, X, Y, tasks, descriptors):
        """
        It fits the model.
        Does not return anything but updates alpha and theta
        and most of all the parameter beta of the linear model.

        Parameters:
        -----------
        X: A numpy array of shape (n,p) for the independent variables.
        Y: A numpy array of shape (n,1) for the dependent variables.
        tasks: A numpy array with (n,T) where every row contain just
            one element 1 and 0 for the other numbers to indicate to
            which task belongs.
        descriptors: An (T,L) array containing the different descriptors
            or each task.
        """
        if self.lambda_1 == 0 or self.lambda_2 == 0:
            raise "inner lasso penalty is set to 0"

        X_copy = np.concatenate(
            (X, np.ones((X.shape[0], 1), dtype=X.dtype)),
            axis=1) if self.fit_intercept else X
        Y_copy = Y
        descriptors_copy = np.concatenate(
            (descriptors,
                np.ones((descriptors.shape[0], 1.),
                        dtype=descriptors.dtype)),
            axis=1) if self.fit_task_intercept else descriptors

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
        L = descriptors_copy.shape[1]

        # initialize theta, alpha and beta
        if not self.warm_start or self.coef_ is None:
            theta = np.ones((1, p), dtype=np.float64)
            if self.random_init:
                alpha = np.random.normal(size=(p, L))\
                    .astype(np.float64)
            else:
                alpha = np.zeros((p, L), dtype=np.float64)
                optimize_alpha.coef_ = alpha.flatten()
            beta_0 = self.get_beta(descriptors_copy, alpha, theta)
        else:
            beta_0 = self.beta_
            alpha = self.alpha_
            theta = self.theta_

        # Precalculate x_alpha matrix for optimizing alpha
        # It is a (n, p*L) where xD_{i,(j-1)L+l) = \theta_{j}d_l
        # for the corresponding task.
        # In the algorithm 3.1 in the paper, w = \theta * x_alpha
        repeated_descriptors = np.repeat(descriptors_copy, p, axis=0)
        repeated_descriptors.shape = (T, L * p)
        x_alpha = tasks.dot(repeated_descriptors)
        x_alpha = x_alpha * np.repeat(X_copy, L, axis=1)
        del repeated_descriptors

        # initilize x_theta: refering to "z" in the algorithm 3.1
        x_theta = tasks.dot(descriptors_copy.dot(alpha.T)) \
            * X_copy

        # Start the two phase optimization
        continue_optimization = True
        self.n_iter_ = 0
        while continue_optimization:
            # Optimize for alpha and update alpha
            X_alpha = np.repeat(theta, L, axis=1) * x_alpha
            alpha[:] = optimize_alpha.fit(X_alpha, Y_copy).coef_\
                .reshape((p, L))

            # update beta: \beta = \theta * (D.\alpha^T)
            beta = self.get_beta(descriptors_copy, alpha, theta)

            # Optimize for theta and update theta
            x_theta[:] = tasks.dot(descriptors_copy.dot(alpha.T)) * X_copy
            theta[:] = optimize_theta.fit(x_theta, Y_copy).coef_

            self.n_iter_ += 1
            if (np.linalg.norm(beta.flatten() - beta_0.flatten()) <
                    self.tol):
                continue_optimization = False
            else:
                beta_0 = beta
            if self.n_iter_ >= self.max_iter:
                continue_optimization = False
                warnings.warn("warning: exit optimization because \
                              number of iterations exceeded maximum \
                              number of allowed iterations")
        self.coef_ = beta.T
        self.alpha_ = alpha
        self.theta_ = theta

    def get_beta(self, descriptors, alpha=None, theta=None):
        """
        Computes beta based on theta, alpha and the task descriptors
        matrix.
        Required in the optimize function.
        \beta = \theta * (D.\alpha^T)
        """
        desc_ = np.concatenate(
            (descriptors,
                np.ones((descriptors.shape[0], 1),
                        dtype=descriptors.dtype)),
            axis=1) if self.fit_task_intercept else descriptors
        alpha = self.alpha_ if alpha is None else alpha
        theta = self.theta_ if theta is None else theta
        return theta * desc_.dot(alpha.T)

    def predict(self, X, tasks, descriptors):
        """
        Computes the predicted output: ŷ = \beta * x
        """
        new_X = np.concatenate(
            (X, np.ones((X.shape[0], 1), dtype=X.dtype)),
            axis=1) if self.fit_intercept else X

        # new_descriptors = np.concatenate(
        #     (descriptors,
        #         np.ones((descriptors.shape[0], 1),
        #                 dtype=descriptors.dtype)),
        #     axis=1) if self.fit_task_intercept else descriptors
        # beta = self.get_beta(new_descriptors)

        beta = self.coef_.T
        return np.sum(tasks.dot(beta) * new_X, axis=1)

# class MuMuLaDerCV():
#     def _cross_validation(self, X, Y, tasks, descriptors, cv_indices=None,
#                           lambda_1=1.0, lambda_2=1.0, std=False):
#         """
#         Compute the mean (and std if required) mse scores
#         obtained over a cross validation scheme.

#         Parameters:
#         -----------
#         lambda_1 : float, optional
#             Constant that multiplies the L1 term when optimizing theta.
#             Defaults to 1.0.
#         lambda_2 : float, optional
#             Constant that multiplies the L1 term when optimizing alpha.
#             Defaults to 1.0.
#         cv_indices:

#         std: boolean, requiring standard deviation or not
#         """
#         cv_indices = KFold(self.x.shape[0], n_folds=10,
#                            shuffle=True) if cv_indices is None
#                                          else cv_indices
#         mse = []
#         for train, test in cv_indices:
#             x_tr = X[train, :]
#             y_tr = Y[train]
#             x_te = X[test, :]
#             y_te = Y[test]
#             tasks_tr = tasks[train, :]
#             tasks_te = tasks[test, :]
#             model = MuMuLaDer(
#                 x_tr, y_tr, tasks_tr, descriptors, lambda_1=lambda_1,
#                 lambda_2=lambda_2,
#                 intercept=False, task_intercept=False,
#                 alpha_params=self.alpha_params,
#                 theta_params=self.theta_params, tol=self.tol,
#                 random_init=self.random_init)
#             model.optimize(max_iters=self.cross_max_iters)
#             y_pr = model.predict(x_te, tasks_te, self.descriptors)
#             mse.append(self.loss_f(y_te.flatten(), y_pr.flatten()))
#         if std:
#             return np.mean(mse), np.std(mse)
#         return np.mean(mse)

#     def set_lambda_no_search(self, max_iters=100, n_folds=3,
#                              lambda_1_list=None, lambda_2_list=None):
#         """
#         Find the best "lambda_1" and "lambda_2" where performances
#         are assessed via "n_folds"-CV and searched by GridSearchCV.
#         In addition, the considered best parameter is maximizing both
#         the performance (first priority) AND the sparcity (second)
#         (see below).

#         Parameters:
#         -----------
#         max_iter: maximum nb of iterations at the optimization step
#             of the MMLD model.
#         n_folds: nb of fold in the CV scheme
#         lambda_1_list: list of lambda_1 parameter for GridSearchCV
#         lambda_2_list: list of lambda_2 parameter for GridSearchCV
#         """
#         self.cv_indices = KFold(self.x.shape[0], n_folds=n_folds,
#                                 shuffle=True)
#         self.cross_max_iters = max_iters
#         params = []
#         mean_scores = []
#         std_scores = []
#         for l_1 in lambda_1_list:
#             for l_2 in lambda_2_list:
#                 local_params = {'lambda_1': l_1, 'lambda_2': l_2}
#                 m_score, std_score = self.__cross_validation(params,
#                                                              std=True)
#                 mean_scores.append(m_score)
#                 std_scores.append(std_score)
#                 params.append(local_params)
#         best_i = np.argmin(mean_scores)
#         best_param = params[best_i]
#         best_std = std_scores[best_i]
#         best_mean = mean_scores[best_i]

#         # let's define as valid every set of parameters
#         # (lambda_1,lambda_2) both bigger than
#         # (best_lambda_1,best_lambda_2) (which would mean sparser
#         # model), for which the associated performance is in the
#         # range of the standard deviation around the best performance.
#         # valid set of parameters are addded to a list in increasing
#         # order of their corresponding performance.
#         # The best parameter is then chosen as the one in the middle
#         # of this list.
#         valid_params = []
#         valid_means = []
#         for i, param in enumerate(params):
#             # check wether local_lambda_1 and local_lambda_2 are bigger
#             # respectively to best_lambda_1 and best_lambda_2
#             bigger = True
#             for key in param:
#                 if param[key] < best_param[key]:
#                     bigger = False
#             if bigger:
#                 if np.abs(mean_scores[i] - best_mean) <= best_std:
#                     valid_means.append(mean_scores[i])
#                     valid_params.append(param)
#         ordered_indices = np.argsort(valid_means)
#         best = valid_params[ordered_indices[
#                             np.floor(len(ordered_indices) / 2.)]]

#         self.lambda_1 = best['lambda_1']
#         self.lambda_2 = best['lambda_2']
#         self.path = {'params': params, 'scores': mean_scores,
#                      'std_scores': std_scores}
#         return best
