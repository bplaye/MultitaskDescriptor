from __future__ import division

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, LassoLars
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


def regress(x, descriptors, alpha, theta, task):
    """
    NOT USED (to my knowledge)
    it returns the predicted output (of size
    (nb_of_task*nb_of_instances_per_task,1) ) given instance feature
    "x", task features "descriptors", alpha (see model),
    theta (see model), and 'instance to task assignment' matrix 'task'
    """
    y = np.zeros((x.shape[0], 1), dtype=np.float64)
    beta = task.dot(theta * alpha.dot(descriptors.T).T)
    y[:, 0] = np.sum(beta * x, axis=1)

    return y


def assign_task(n, T, task):
    """
    if number of instances per task remains the same, it fills
    automatically the 'instance to task assignment' matrix
    (called 'task' here) per increasing task index
    """
    for i in range(T):
        task[i * n:(i + 1) * n, i] = 1.0


class MuMuLaDer(object):
    theta = None
    alpha = None

    def __init__(
            self, x, y, tasks, descriptors, intercept=False,
            task_intercept=False, lambda_1=None, lambda_2=None,
            alpha_params=None, theta_params=None, tol=0.01,
            random_init=False, loss_f=MSE):
        """
        Define the model. It does not uses the traditional format by
        SciKit but the one from GPy

        Parameters:
        -----------
        x: A numpy array of shape (n,p) for the independent variables.
        y: A numpy array of shape (n,1) for the dependent variables.
        tasks: A numpy array with (n,T) where every row contain just
            one element 1 and 0 for the other numbers to indicate to
            which task belongs.
        descriptors: An (T,L) array containing the different descriptors
            or each task.
        intercept: If True an intercept variable is used
        task_intercept: If True an intercept variable is added
            to task descriptors
        lambda_1: Regularization parameter for theta
        lambda_2: Regularization parameter for alpha
        alpha_params: dictionary containing the parameters
            for the alpha optimizer
        theta_params: dictionary containing the parameters
            for the theta optimizer
        tol: float. if the change in beta is smaller than this number,
        it will stop the optimization.
        random_init: weather to use random initialization or not
        loss_f: loss function to minimize in cross_validation
        epsilon: A parameter for defining when to stop the optimization.
        """
        self.intercept = intercept
        self.task_intercept = task_intercept

        self.x = np.concatenate(
            (x, np.ones((x.shape[0], 1), dtype=x.dtype)),
            axis=1) if self.intercept else x
        self.y = y
        self.tasks = tasks
        self.descriptors = np.concatenate(
            (descriptors,
                np.ones((descriptors.shape[0], 1.),
                        dtype=descriptors.dtype)),
            axis=1) if self.task_intercept else descriptors
        self.lambda_1 = 1. if lambda_1 is None else lambda_1
        self.lambda_2 = 1. if lambda_2 is None else lambda_2
        self.theta = None
        self.alpha = None
        self.alpha_params = {} if alpha_params is None else alpha_params
        self.theta_params = {} if theta_params is None else theta_params
        self.tol = tol
        self.iterations = 0
        self.random_init = random_init
        self.loss_f = loss_f

    def optimize(self, optimizer=DEFAULT_OPTIMIZER, max_iters=100,
                 max_inner_iters=10):
        """
        It fits the model.
        Does not return anything but updates alpha and theta
        and most of all the parameter beta of the linear model.

        Parameters:
        -----------
        self. parameters needed: x, tasks, descriptors,
            lambda_1, lambda_2, tol
        max_iters: maximum iterations
        max_inner_itters: how many iterations to use
            in every lasso optimization
        optimizer: Select the algorithm for optimize.
            'Lasso' uses the Lasso implementation of sklearn
            'LassoLars' uses the LassoLars implementation of sklearn

        """
        # get input sizes and initizialize parameters
        # n is the total amount of instances among all tasks
        # p is number of features
        n, p = self.x.shape
        # T is the number of task
        T = self.tasks.shape[1]
        # L is number of task descriptor
        L = self.descriptors.shape[1]

        # Precalculate x_alpha matrix for optimizing alpha
        # It is a (n, p*L) where xD_{i,(j-1)L+l) = \theta_{j}d_l
        # for the corresponding task.
        # In the algorithm 3.1 in the paper, w = \theta * x_alpha
        repeated_descriptors = np.repeat(self.descriptors, p, axis=0)
        repeated_descriptors.shape = (T, L * p)
        x_alpha = self.tasks.dot(repeated_descriptors)
        x_alpha = x_alpha * np.repeat(self.x, L, axis=1)
        del repeated_descriptors

        # Initialize theta
        self.theta = np.ones((1, p), dtype=np.float64)

        # Initialize theta and alpha optimizers
        if optimizer == LASSO:
            optimize_theta = Lasso(warm_start=True,
                                   positive=True,
                                   alpha=self.lambda_1,
                                   max_iter=max_inner_iters,
                                   **self.theta_params)
            optimize_alpha = Lasso(warm_start=True,
                                   alpha=self.lambda_2,
                                   max_iter=max_inner_iters,
                                   **self.alpha_params)
        elif optimizer == LASSOLARS:
            optimize_theta = LassoLars(alpha=self.lambda_1,
                                       **self.theta_params)
            optimize_alpha = LassoLars(alpha=self.lambda_2,
                                       **self.alpha_params)
        else:
            raise Exception("Not a valid value")

        # Initialize alpha
        if self.random_init:
            self.alpha = np.random.normal(size=(p, L))\
                .astype(np.float64)
        else:
            self.alpha = np.zeros((p, L), dtype=np.float64)
        optimize_alpha.coef_ = self.alpha.flatten()
        print(optimize_alpha.coef_.shape)
        print("ok")

        # Initialize beta based on the task descriptor
        # and the initial values of theta and alpha.
        # \beta = \theta * (D.\alpha^T)
        # beta_0 is the beta value at the step "t-1"
        beta_0 = self.get_beta(self.descriptors)

        # initilize x_theta: refering to "z" in the algorithm 3.1
        x_theta = self.tasks.dot(self.descriptors.dot(self.alpha.T)) \
            * self.x

        # Start the two phase optimization
        continue_optimization = True
        self.iterations = 0
        while continue_optimization:
            # Optimize for alpha and update alpha
            print(self.theta.shape)
            print(x_alpha.shape)
            X_ALPHA = np.repeat(self.theta, L, axis=1) * x_alpha
            print(X_ALPHA.shape)
            print(self.y.shape)
            coefs = optimize_alpha.fit(X_ALPHA, self.y).coef_
            print(coefs.shape)
            self.alpha[:] = optimize_alpha.fit(X_ALPHA, self.y)\
                .coef_.reshape((p, L))

            # update beta: \beta = \theta * (D.\alpha^T)
            beta = self.get_beta(self.descriptors)

            # Optimize for theta and update theta
            x_theta[:] = self.tasks.dot(
                self.descriptors.dot(self.alpha.T)) * self.x
            self.theta[:] = optimize_theta.fit(x_theta, self.y).coef_
            # self.theta.shape = (1, len(self.theta))

            self.iterations += 1
            # print 'Mumulader', self.iterations
            # f = plt.figure()
            # plt.matshow(beta - beta_0)
            # plt.colorbar()
            # plt.show()
            # print "Norm: {}".format(np.linalg.norm(beta.flatten()
            # - beta_0.flatten()))
            if (np.linalg.norm(beta.flatten() - beta_0.flatten()) <
                    self.tol):
                continue_optimization = False
            else:
                beta_0 = beta
            if self.iterations >= max_iters:
                continue_optimization = False
                warnings.warn("warning: exit optimization because \
                              number of iterations exceeded maximum \
                              number of allowed iterations")

    def get_beta(self, descriptors=None):
        """
        Computes beta based on theta, alpha and the task descriptors
        matrix.
        Required in the optimize function.
        \beta = \theta * (D.\alpha^T)
        """
        if descriptors is None:
            descriptors = self.descriptors
        if self.descriptors.shape[1] == descriptors.shape[1]:
            desc_ = descriptors
        else:
            desc_ = np.concatenate(
                (descriptors,
                    np.ones((descriptors.shape[0], 1),
                            dtype=descriptors.dtype)),
                axis=1) if self.task_intercept else descriptors

        return self.theta * desc_.dot(self.alpha.T)

    def predict(self, x, tasks, descriptors):
        """
        Computes the predicted output: ŷ = \beta * x
        """
        new_x = np.concatenate(
            (x, np.ones((x.shape[0], 1), dtype=x.dtype)),
            axis=1) if self.intercept else x

        new_descriptors = np.concatenate(
            (descriptors,
                np.ones((descriptors.shape[0], 1),
                        dtype=descriptors.dtype)),
            axis=1) if self.task_intercept else descriptors
        beta = self.get_beta(new_descriptors)
        return np.sum(tasks.dot(beta) * new_x, axis=1)

    def get_params(self):
        """
        Returns theta and alpha values in a dictionary format
        """
        return {
            'theta': self.theta,
            'alpha': self.alpha
        }

    def __cross_validation(self, values, std=False):
        """
        Compute the mean (and std if required) mse scores
        obtained over a cross validation scheme.

        Parameters:
        -----------
        values: a dictionary (in order to use hyperopt) with two keys
            'lambda_1' and 'lambda_2' referring to
            their corresponding values
        std: boolean, requiring standard deviation or not of the scores
        """
        lambda_1 = values['lambda_1']
        lambda_2 = values['lambda_2']

        mse = []
        for train, test in self.cv_indices:
            x_tr = self.x[train, :]
            y_tr = self.y[train]
            x_te = self.x[test, :]
            y_te = self.y[test]
            tasks_tr = self.tasks[train, :]
            tasks_te = self.tasks[test, :]
            descriptors = self.descriptors
            model = MuMuLaDer(
                x_tr, y_tr, tasks_tr, descriptors, lambda_1=lambda_1,
                lambda_2=lambda_2,
                intercept=False, task_intercept=False,
                alpha_params=self.alpha_params,
                theta_params=self.theta_params, tol=self.tol,
                random_init=self.random_init)
            model.optimize(max_iters=self.cross_max_iters)
            y_pr = model.predict(x_te, tasks_te, self.descriptors)
            mse.append(self.loss_f(y_te.flatten(), y_pr.flatten()))
        if std:
            return np.mean(mse), np.std(mse)
        return np.mean(mse)

    def set_lambda_no_search(self, max_iters=100, n_folds=3,
                             lambda_1_list=None, lambda_2_list=None):
        """
        Find the best "lambda_1" and "lambda_2" where performances
        are assessed via "n_folds"-CV and searched by GridSearchCV.
        In addition, the considered best parameter is maximizing both
        the performance (first priority) AND the sparcity (second)
        (see below).

        Parameters:
        -----------
        max_iter: maximum nb of iterations at the optimization step
            of the MMLD model.
        n_folds: nb of fold in the CV scheme
        lambda_1_list: list of lambda_1 parameter for GridSearchCV
        lambda_2_list: list of lambda_2 parameter for GridSearchCV
        """
        self.cv_indices = KFold(self.x.shape[0], n_folds=n_folds,
                                shuffle=True)
        self.cross_max_iters = max_iters
        params = []
        mean_scores = []
        std_scores = []
        for l_1 in lambda_1_list:
            for l_2 in lambda_2_list:
                local_params = {'lambda_1': l_1, 'lambda_2': l_2}
                m_score, std_score = self.__cross_validation(params,
                                                             std=True)
                mean_scores.append(m_score)
                std_scores.append(std_score)
                params.append(local_params)
        best_i = np.argmin(mean_scores)
        best_param = params[best_i]
        best_std = std_scores[best_i]
        best_mean = mean_scores[best_i]

        # let's define as valid every set of parameters
        # (lambda_1,lambda_2) both bigger than
        # (best_lambda_1,best_lambda_2) (which would mean sparser
        # model), for which the associated performance is in the
        # range of the standard deviation around the best performance.
        # valid set of parameters are addded to a list in increasing
        # order of their corresponding performance.
        # The best parameter is then chosen as the one in the middle
        # of this list.
        valid_params = []
        valid_means = []
        for i, param in enumerate(params):
            # check wether local_lambda_1 and local_lambda_2 are bigger
            # respectively to best_lambda_1 and best_lambda_2
            bigger = True
            for key in param:
                if param[key] < best_param[key]:
                    bigger = False
            if bigger:
                if np.abs(mean_scores[i] - best_mean) <= best_std:
                    valid_means.append(mean_scores[i])
                    valid_params.append(param)
        ordered_indices = np.argsort(valid_means)
        best = valid_params[ordered_indices[
                            np.floor(len(ordered_indices) / 2.)]]

        self.lambda_1 = best['lambda_1']
        self.lambda_2 = best['lambda_2']
        self.path = {'params': params, 'scores': mean_scores,
                     'std_scores': std_scores}
        return best

    def get_alpha(self):
        return {'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2}
