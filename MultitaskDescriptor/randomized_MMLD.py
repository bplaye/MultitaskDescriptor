import numpy as np
from sklearn.model_selection import KFold
from MultitaskDescriptor.MMLD import *
import itertools
from copy import deepcopy


class RandomizedMumulader(object):
    importance = None
    params = None
    model_class = MuMuLaDer

    def __init__(self, X, Y, tasks, descriptors, B, alpha, threshold,
                 kwargs, loss_f=MSE):
        """
        Define the randomized-MMLD method.

        Parameters:
        -----------
        x: A numpy array of shape (n,p) for the independent variables.
        y: A numpy array of shape (n,1) for the dependent variables.
        tasks: A numpy array with (n,T) where every row contain just
            one element 1 and 0 for the other numbers to indicate to
            which task belongs.
        descriptors: An (T,L) array containing the different descriptors
            or each task.
        lambda_1: regularization parameter in MMLD
        lambda_2: regularization parameter in MMLD
        B: Number of repetitions (1 repetition is a MMLD method
            applied on bootstrapped samples and sampled reweighted
            features)
        alpha: minim for sampling algorithm
        """
        self.X = X
        self.Y = Y
        self.tasks = tasks
        self.descriptors = descriptors
        self.B = B
        self.alpha = alpha
        self.kwargs = kwargs
        self.loss_f = loss_f
        self.threshold = threshold

    def optimize(self, optimizer='lasso', max_iters=100,
                 max_inner_iters=10):
        n, p = self.X.shape
        _, k = self.tasks.shape

        # initializing theta (MMLD parameter) and selected
        # (recording feature selection for each task)
        # if adding an intercept variable to instance
        # features is requested, do so
        if self.kwargs.get('intercept', False):
            params = {
                'theta': np.zeros((1, p + 1)),
                'selected': np.zeros((k, p + 1))
            }
        else:
            params = {
                'theta': np.zeros((1, p)),
                'selected': np.zeros((k, p))
            }

        # initializing alpha ()
        # if adding an intercept variable to task descriptors
        # is requested, do so
        if self.kwargs.get('task_intercept', False):
            params['alpha'] = np.zeros((params['theta'].shape[1],
                                        self.descriptors.shape[1] + 1))
        else:
            params['alpha'] = np.zeros((params['theta'].shape[1],
                                        self.descriptors.shape[1]))

        # lets repeat B times the MMLD method on bootstrapped samples
        # and reweighted features
        for b in xrange(self.B):
            # bootstraping instances
            boots1 = np.random.choice(np.arange(n), size=n,
                                      replace=True)
            # uniform sampling for reweighting features
            W = np.random.uniform(low=self.alpha,
                                  high=1.0, size=(1, p))

            # get modified X, Y and tasks for the randomized approach
            new_X = self.X[boots1, :] * W
            new_Y = self.Y[boots1, :]
            new_tasks = self.tasks[boots1, :]

            # run MMLD on the randomized X and Y
            model = self.model_class(new_X, new_Y, new_tasks,
                                     self.descriptors, **self.kwargs)
            model.optimize(optimizer=optimizer,
                           max_iters=max_iters,
                           max_inner_iters=max_inner_iters)

            # u pdate feature importance
            beta = model.get_beta()
            params['selected'] += (beta != 0) / float(self.B)

            # update mean of theta and alpha
            local_params = model.get_params()
            params['theta'] += local_params['theta'] / float(self.B)
            params['alpha'] += local_params['alpha'] / float(self.B)
        # recording the result of optimization
        self.params = params

    def get_params(self):
        return self.params

    def get_final_model(self):
        """
        Final model is the MMLD model whose parameter
        beta (=theta*descriptors*alpha) is the mean of the betas
        obtained after running B times MMLD on bootstrapped instances
        and reweighted features.
        """
        model = self.model_class(self.X, self.Y, self.tasks,
                                 self.descriptors, **self.kwargs)
        model.alpha = self.params['alpha']
        model.theta = self.params['theta']
        return model

##################################################
    def __cross_validation(self, values, std=False):
        lambda_1 = values['lambda_1']
        lambda_2 = values['lambda_2']
        B = values['B']
        threshold = values['threshold']
        alpha = values['alpha']

        mse = []
        for train, test in self.cv_indices:
            x_tr = self.X[train, :]
            y_tr = self.Y[train, :]
            x_te = self.X[test, :]
            y_te = self.Y[test, :]
            tasks_tr = self.tasks[train, :]
            tasks_te = self.tasks[test, :]
            descriptors = self.descriptors
            kwargs = deepcopy(self.kwargs)
            kwargs['lambda_1'] = lambda_1
            kwargs['lambda_2'] = lambda_2

            # run randomized-MMLD to get feature importance
            # (final beta parameter)
            rmodel = RandomizedMumulader(x_tr, y_tr, tasks_tr,
                                         descriptors, B, threshold,
                                         alpha, kwargs)
            rmodel.optimize(max_iters=self.cross_max_iters)

            # run final model (a MMLD whose beta is the mean of betas
            # learned on B MMLD
            # trained on bootstrapped samples and reweighted features)
            # and compute performance on the current fold
            model = rmodel.get_final_model()
            y_pr = model.predict(x_te, tasks_te, self.descriptors)
            mse.append(self.loss_f(y_te.flatten(), y_pr.flatten()))
        if std:
            return np.mean(mse), np.std(mse)
        return np.mean(mse)

    def set_lambda_no_search(self, max_iters=100, n_folds=3,
                             B_list=None, threshold_list=None,
                             alpha_list=None, lambda_1_list=None,
                             lambda_2_list=None):
        """
        Find the best "lambda_1" and "lambda_2" where performances are
        assessed via "n_folds"-CV and searched by GridSearchCV.
        In addition, the considered best parameter is maximizing both
        the performance (first priority) and the sparcity (second)
        (see below).

        Parameters:
        -----------
        max_iter: maximum nb of iterations at the optimization step
            of the MMLD model.
        n_folds: nb of fold in the CV scheme
        B_list: list of integers, each referring to the number of steps
            in the randomized procedure. B is an hyper-parameter.
        threshold_list: list of integers, each referring to the
            threshold considered in the final model.
            It is an hyper-parameter.
        alpha_list:
        lambda_1_list: list of lambda_1 parameter for GridSearchCV
        lambda_2_list: list of lambda_2 parameter for GridSearchCV
        """
        self.cv_indices = KFold(self.X.shape[0], n_folds=n_folds,
                                shuffle=True)
        self.cross_max_iters = max_iters
        search_params = []
        mean_scores = []
        std_scores = []
        for B, threshold, alpha, l_1, l_2 \
            in itertools.product(B_list, threshold_list, alpha_list,
                                 lambda_1_list, lambda_2_list):
            local_params = {'B': B, 'threshold': threshold,
                            'alpha': alpha, 'lambda_1': l_1,
                            'lambda_2': l_2}
            m_score, std_score = self.__cross_validation(local_params,
                                                         std=True)
            mean_scores.append(m_score)
            std_scores.append(std_score)
            search_params.append(local_params)
        best_i = np.argmin(mean_scores)
        best_param = pars[best_i]
        best_std = std_scores[best_i]
        best_mean = mean_scores[best_i]

        # let's define as valid every set of parameters
        # (lambda_1,lambda_2) both bigger than
        # (best_lambda_1,best_lambda_2) (which would mean sparser
        # model), for which the associated performance is
        # in the range of the standard deviation
        # around the best performance.
        # valid set of parameters are addded to a list in increasing
        # order of their corresponding performance.
        # The best parameter is then chosen as the one in the
        # middle of this list.
        valid_params = []
        valid_means = []
        for i, param in enumerate(search_params):
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
        self.kwargs['lambda_1'] = best['lambda_1']
        self.kwargs['lambda_2'] = best['lambda_2']
        self.B = best['B']
        self.threshold = best['threshold']
        self.alpha = best['alpha']
        self.path = {'params': pars, 'scores': mean_scores,
                     'std_scores': std_scores}
        return best
