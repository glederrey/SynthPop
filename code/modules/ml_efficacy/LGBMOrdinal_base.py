from inspect import classify_class_attrs
from lightgbm.callback import early_stopping
import numpy as np
import copy

import pickle

import warnings

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, brier_score_loss
from sklearn.base import clone

from typing import Callable, Optional, Union

from scipy.optimize import minimize

from collections.abc import Iterable

import math
import time
import sys

from lightgbm.sklearn import (LGBMModel, LGBMClassifier, LGBMRanker, LGBMRegressor, 
    _ObjectiveFunctionWrapper, _EvalFunctionWrapper, 
    Dataset, _ConfigAliases, _log_warning,
    LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
    _LGBMCheckClassificationTargets, _LGBMCheckSampleWeight, _LGBMCheckXY,
    _LGBMComputeSampleWeight, _LGBMLabelEncoder, dt_DataTable,
    pd_DataFrame,
    train)

from lightgbm.callback import (log_evaluation, record_evaluation)

from sklearn.model_selection import ParameterGrid, ParameterSampler


class LGBMOrdinal(LGBMModel):
    '''A classifier to perform unimodal ordinal regression on ordinal multinomial 
    data'''

    def __init__(
        self, 
        distribution: str = 'logbinomial',
        fit_weights: bool = True,
        partial_weights_fit: bool = False,
        prefit_weights: bool = True,
        proportion_weights_fit: float = 1, 
        tol: float = 1e-10, 
        boosting_type: str = 'gbdt',
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        min_split_gain: float = 0.,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.,
        reg_alpha: float = 0.,
        reg_lambda: float = 0.,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: int = -1,
        silent: Union[bool, str] = 'warn',
        importance_type: str = 'split',
        **kwargs):
        self.distribution = distribution
        self.fit_weights = fit_weights
        self.partial_weights_fit = partial_weights_fit
        self.prefit_weights = prefit_weights
        self.proportion_weights_fit = proportion_weights_fit
        self.tol=tol
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        self._Booster = None
        self._evals_result = None
        self._best_score = None
        self._best_iteration = None
        self._other_params = {}
        self.class_weight = None
        self._class_weight = None
        self._class_map = None
        self._n_features = None
        self._n_features_in = None
        self._classes = None
        self._n_classes = None
        self.set_params(**kwargs)

    def fit(self, X, y, 
            init_score=None, 
            eval_set=None, eval_names=None,
            eval_init_score=None,
            eval_metric='log_loss', early_stopping_rounds=None,
            verbose='warn', feature_name='auto', categorical_feature='auto',
            callbacks=None, init_model=None, init_weights=None):
        # Check that X and y have correct shape
        
        #X, y = check_X_y(X, y)
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)

        # Transform labels and store the classes seen during fit
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)

        self._class_map = dict(zip(self._le.classes_, self._le.transform(self._le.classes_)))

        self._classes = self._le.classes_
        self._counter = 0
        self._n_classes = self._classes.shape[0]

        self._classes = self._le.classes_
        self._n_classes = self._classes.shape[0]
        if self._n_classes <= 2:
            raise ValueError('Ordinal classification requires more than 2 classes')

        # do not modify args, as it causes errors in model selection tools
        valid_sets = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = [None] * len(eval_set)
            for i, (valid_x, valid_y) in enumerate(eval_set):
                if valid_x is X and valid_y is y:
                    valid_sets[i] = (valid_x, _y)
                else:
                    valid_sets[i] = (valid_x, self._le.transform(valid_y))

        eval_metric = self._prepare_metrics(eval_metric)

        if self.distribution=='logbinomial':
            self._probs = self._logbinomial_probs
            self._grad_hess = self._logbinomial_grad_hess
        elif self.distribution=='binomial':
            self._probs = self._binomial_probs
            self._grad_hess = self._binomial_grad_hess
            self.fit_weights=False
        elif self.distribution=='poisson':
            self._probs = self._poisson_probs
            self._grad_hess = self._poisson_grad_hess
            self._factvet = np.log(np.array([np.math.factorial(i+1) for i in range(self._n_classes)]))
        else:
            raise ValueError('At this stage, only logbinomial, binomial and poisson distributions are supported')


        self.weights_ = np.zeros(self._n_classes)

        if init_weights:
            init_weights = np.array(init_weights).astype(float)
            if init_weights.shape != (self.n_classes,):
                raise ValueError('Init weights must have shape of (n_classes,)')
            self.weights_ = init_weights

        elif self.fit_weights and self.prefit_weights:
            self.weights_ = self._fit_w(_y, np.zeros(len(_y)), tol=self.tol)

        self._fit(X, _y, init_score=init_score, eval_set=valid_sets,
                    eval_names=eval_names,
                    eval_init_score=eval_init_score,
                    eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose, feature_name=feature_name, categorical_feature=categorical_feature,
                    callbacks=callbacks, init_model=init_model)

        if self.fit_weights:
            self.weights_ = self._fit_w(_y, self.predict_regression(X), tol=self.tol)
        return self

    
    def predict_regression(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        check_is_fitted(self)
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             f"match the input. Model n_features_ is {self._n_features} and "
                             f"input n_features is {n_features}")
        return self._Booster.predict(X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration,
                                     pred_leaf=pred_leaf, pred_contrib=pred_contrib, **kwargs)        

    def predict_proba(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        check_is_fitted(self)
        result = self.predict_regression(X, raw_score, start_iteration, num_iteration,
                                    pred_leaf, pred_contrib, **kwargs)
        return self._probs(result)

    def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(X, raw_score, start_iteration, num_iteration,
                                    pred_leaf, pred_contrib, **kwargs)
        class_index = np.argmax(result, axis=1)
        return self._le.inverse_transform(class_index)

    def score(self, X, y_true): 
        _y_true = self._le.transform(y_true)
        return log_loss(_y_true, self.predict_proba(X))

    def _prepare_metrics(self, eval_metric):
        if eval_metric is None:
            return self._prepare_metric('log_loss')
        elif isinstance(eval_metric, str) or callable(eval_metric):
            return self._prepare_metric(eval_metric)
        elif isinstance(eval_metric, Iterable):
            return [self._prepare_metric(m) for m in eval_metric]
        else:
            raise ValueError('''eval_metric must be None, a string, a callable 
            function, or a list of strings and callable functions''')

    def _prepare_metric(self, metric):
        if isinstance(metric, str):
            if metric in ['neg_log_loss', 'log_loss', 'multi_log_loss']:
                return lambda y_true, z:(metric, log_loss(y_true, self._probs(z), labels=self._classes), False)
            elif metric in ['accuracy', 'accuracy_score']:
                return lambda y_true, z: (metric, accuracy_score(y_true, np.argmax(self._probs(z), axis=1)), True)
            elif metric in ['error', 'merror', 'multi_error']:
                return lambda y_true, z: (metric, 1-accuracy_score(y_true, np.argmax(self._probs(z), axis=1)), False)
            #elif metric in ['brier_score', 'brier_score_loss']:
                #return lambda y_true, z: (metric, brier_score_loss(y_true, self._probs(z)), False)
            else:
                raise ValueError(f'Metric {metric} not supported')
        elif callable(metric):
            return lambda y_true, z: (metric(y_true, self._probs(z)))
        else:
            raise ValueError('''eval_metrics in an interable must be a string 
            or a callable function''')
            
    def _stablesoftmax(self, x):
        """Compute the softmax of vector x in a numerically stable way."""
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / exps.sum(axis=1)[:,None]

    def _reset_weights(self):
        self.weights_ = np.zeros(self._n_classes)

    def _logbinomial_probs(self, z, w=None):
        lam = 1/(1 + np.exp(-z))
        J = self._n_classes
        if w is None:
            w = self.weights_
        U = (np.array(w) #+ [np.log(math.comb(J-1,i)) for i in range(J)]
             + np.arange(J)*np.log(lam).reshape(-1,1) +
             ((J-1-np.arange(J))*np.log(1-lam).reshape(-1,1)))
        P = self._stablesoftmax(U)
        return P

    def _binomial_probs(self, z, w=None):
        lam = 1/(1 + np.exp(-z))
        J = self._n_classes
        P = [math.comb(J-1,i)*(lam**i)*((1-lam)**(J-1-i)) for i in range(J)]
        return np.array(P).T

    def _poisson_probs(self, z, w=None):
        lam = np.log(1 + np.exp(z))
        if w is None:
            w = self.weights_
        imat = np.tile(np.arange(self._n_classes), (len(z), 1))
        factmat = np.tile(self._factvet, (len(z), 1))
        wmat = np.tile(w, (len(z), 1))
        U = wmat + imat*np.log(lam[:,None]) - lam[:,None] - factmat
        P = self._stablesoftmax(U)
        return P
    
    def _fit_w_partial(self, y_true, z, tol=None):
        w = self.weights_[1:]
        sol = minimize(self._eval_fun, w, args=(y_true, z), jac=self._jac_w, hess=self._hess_w, method='newton-cg', tol=tol).x
        return np.append(0, w + self.learning_rate*(sol-w))

    def _fit_w(self, y_true, z, tol=None):
        w = self.weights_[1:]
        sol = minimize(self._eval_fun, w, args=(y_true, z), jac=self._jac_w, hess=self._hess_w, method='newton-cg', tol=tol).x
        return np.append(0, sol)

    def _objective(self, y_true, z):
        if self.fit_weights and self._counter>self.n_estimators*(1-self.proportion_weights_fit):
            if self.partial_weights_fit:
                self.weights_ = self._fit_w_partial(y_true, z)
            else:
                self.weights_ = self._fit_w(y_true, z)
        self._counter+=1
        return self._grad_hess(y_true, z)
    
    def _logbinomial_grad_hess(self, y_true, z):
        P = self._probs(z)
        N = len(z)
        J = self._n_classes
        nez = np.exp(-z)
        jfrac = ((1-J)/(1+nez)).reshape(-1,1)
        y_mat = np.eye(J)[y_true.astype(int)]
        grad = 1/N*((P-y_mat)*(np.arange(J)+jfrac)).sum(axis=1)
        hess = (P-y_mat)*(jfrac*(nez/(1+nez)).reshape(-1,1))
        for i in range(J):
            for j in range(J):
                hess[:,i]+=(P[:,i]*((i==j)-P[:,j]))*((i+jfrac)*(j+jfrac)).reshape(-1)
        hess = 1/N*hess.sum(axis=1)
        return grad, hess

    def _binomial_grad_hess(self, y_true, z):
        N = len(z)
        J = self._n_classes
        ez = np.exp(z)
        y_mat = np.eye(J)[y_true.astype(int)]
        grad = np.zeros((N,J))
        hess = np.zeros((N,J))
        for i in range(J):
            grad[:,i] = 1/N*(y_mat[:,i])*(((J-1-i)*ez)/(1+ez)-i*(1/(1+ez)))
            hess[:,i] = 1/N*(y_mat[:,i])*(((J-1-i)*ez)/((1+ez)**2)+i*(ez/((1+ez)**2)))
        return grad.sum(axis=1), hess.sum(axis=1)

    def _poisson_grad_hess(self, y_true, z):
        P = self._probs(z)
        N = len(z)
        J = self._n_classes
        ez = np.exp(z)
        iez = 1/(1+ez)
        lez = np.log(1+ez)
        ilez = 1/lez
        tz = ez*iez
        y_mat = np.eye(J)[y_true.astype(int)]
        grad = tz/N * np.array([(P[:,i]-y_mat[:,i])*(i*ilez-1) for i in range(J)]).sum(axis=0)
        tz1 = (ez - lez)/lez**2
        us_hess = []
        for i in range(J):
            us_hess.append(-(P[:,i]-y_mat[:,i])*(i*tz1+1))
            for j in range(J):
                us_hess[i]+=ez*(i*ilez-1)*P[:,i]*((i==j)-P[:,j])*(j*ilez-1)
        hess = (tz*iez/N) * np.array(us_hess).sum(axis=0)
        return grad, hess
    
    def _eval_fun(self, w, y_true, z):
        P = self._probs(z, w=np.append(0, w))
        return log_loss(y_true,P)
    
    def _jac_w(self, w, y_true, z):
        N = len(z)
        y_mat = np.eye(self._n_classes)[y_true.astype(int)]
        P = self._probs(z, w=np.append(0, w))
        return 1/N*(P-y_mat).sum(axis=0)[1:]
    
    def _hess_w(self, w, y_true, z):
        N = len(z)
        P = self._probs(z, w=np.append(0, w))
        return 1/N*(P.sum(axis=0)*np.eye(self._n_classes)-np.dot(P.T,P))[1:,1:]

    def _fit(self, X, y,
            sample_weight=None, init_score=None, group=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_group=None,
            eval_metric=None, early_stopping_rounds=None, verbose='warn',
            feature_name='auto', categorical_feature='auto',
            callbacks=None, init_model=None):
        """Docstring is set after definition, using a template."""
        if self._objective is None:
            if isinstance(self, LGBMRegressor):
                self._objective = "regression"
            elif isinstance(self, LGBMClassifier):
                self._objective = "binary"
            elif isinstance(self, LGBMRanker):
                self._objective = "lambdarank"
            else:
                raise ValueError("Unknown LGBMModel type.")
        if callable(self._objective):
            self._fobj = _ObjectiveFunctionWrapper(self._objective)
        else:
            self._fobj = None

        params = self.get_params()
        # user can set verbose with kwargs, it has higher priority
        if self.silent != "warn":
            _log_warning("'silent' argument is deprecated and will be removed in a future release of LightGBM. "
                         "Pass 'verbose' parameter via keyword arguments instead.")
            silent = self.silent
        else:
            silent = True
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and silent:
            params['verbose'] = -1
        params.pop('silent', None)
        params.pop('importance_type', None)
        params.pop('n_estimators', None)
        params.pop('class_weight', None)
        params.pop('tol', None)
        params.pop('proportion_weights_fit', None)
        params.pop('distribution', None)
        params.pop('partial_weights_fit', None)
        params.pop('fit_weights', None)
        params.pop('prefit_weights', None)

        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(np.iinfo(np.int32).max)
        for alias in _ConfigAliases.get('objective'):
            params.pop(alias, None)
        if self._n_classes is not None and self._n_classes > 2:
            for alias in _ConfigAliases.get('num_class'):
                params.pop(alias, None)
            #params['num_class'] = self._n_classes
        if hasattr(self, '_eval_at'):
            eval_at = self._eval_at
            for alias in _ConfigAliases.get('eval_at'):
                if alias in params:
                    _log_warning(f"Found '{alias}' in params. Will use it instead of 'eval_at' argument")
                    eval_at = params.pop(alias)
            params['eval_at'] = eval_at
        params['objective'] = self._objective
        if self._fobj:
            params['objective'] = 'None'  # objective = nullptr for unknown objective

        # Do not modify original args in fit function
        # Refer to https://github.com/microsoft/LightGBM/pull/2619
        eval_metric_list = copy.deepcopy(eval_metric)
        if not isinstance(eval_metric_list, list):
            eval_metric_list = [eval_metric_list]

        # Separate built-in from callable evaluation metrics
        eval_metrics_callable = [_EvalFunctionWrapper(f) for f in eval_metric_list if callable(f)]
        eval_metrics_builtin = [m for m in eval_metric_list if isinstance(m, str)]

        # # register default metric for consistency with callable eval_metric case
        # original_metric = self._objective if isinstance(self._objective, str) else None
        # if original_metric is None:
        #     # try to deduce from class instance
        #     if isinstance(self, LGBMRegressor):
        #         original_metric = "l2"
        #     elif isinstance(self, LGBMClassifier):
        #         original_metric = "multi_logloss" if self._n_classes > 2 else "binary_logloss"
        #     elif isinstance(self, LGBMRanker):
        #         original_metric = "ndcg"

        # # overwrite default metric by explicitly set metric
        # params = _choose_param_value("metric", params, original_metric)

        # # concatenate metric from params (or default if not provided in params) and eval_metric
        # params['metric'] = [params['metric']] if isinstance(params['metric'], (str, type(None))) else params['metric']
        # params['metric'] = [e for e in eval_metrics_builtin if e not in params['metric']] + params['metric']
        # params['metric'] = [metric for metric in params['metric'] if metric is not None]

        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            _X, _y = _LGBMCheckXY(X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2)
            if sample_weight is not None:
                sample_weight = _LGBMCheckSampleWeight(sample_weight, _X)
        else:
            _X, _y = X, y

        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)

        self._n_features = _X.shape[1]
        # copy for consistency
        self._n_features_in = self._n_features

        def _construct_dataset(X, y, sample_weight, init_score, group, params,
                               categorical_feature='auto'):
            return Dataset(X, label=y, weight=sample_weight, group=group,
                           init_score=init_score, params=params,
                           categorical_feature=categorical_feature)

        train_set = _construct_dataset(_X, _y, sample_weight, init_score, group, params,
                                       categorical_feature=categorical_feature)

        valid_sets = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError(f"{name} should be dict or list")

            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(eval_sample_weight, 'eval_sample_weight', i)
                    valid_class_weight = _get_meta_data(eval_class_weight, 'eval_class_weight', i)
                    if valid_class_weight is not None:
                        if isinstance(valid_class_weight, dict) and self._class_map is not None:
                            valid_class_weight = {self._class_map[k]: v for k, v in valid_class_weight.items()}
                        valid_class_sample_weight = _LGBMComputeSampleWeight(valid_class_weight, valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = _get_meta_data(eval_init_score, 'eval_init_score', i)
                    valid_group = _get_meta_data(eval_group, 'eval_group', i)
                    valid_set = _construct_dataset(valid_data[0], valid_data[1],
                                                   valid_weight, valid_init_score, valid_group, params)
                valid_sets.append(valid_set)

        if isinstance(init_model, LGBMModel):
            init_model = init_model.booster_

        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            _log_warning("'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
                         "Pass 'early_stopping()' callback via 'callbacks' argument instead.")
            params['early_stopping_rounds'] = early_stopping_rounds

        if callbacks is None:
            callbacks = []
        else:
            callbacks = copy.copy(callbacks)  # don't use deepcopy here to allow non-serializable objects

        if verbose != 'warn':
            _log_warning("'verbose' argument is deprecated and will be removed in a future release of LightGBM. "
                         "Pass 'log_evaluation()' callback via 'callbacks' argument instead.")
        else:
            if callbacks:  # assume user has already specified log_evaluation callback
                verbose = False
            else:
                verbose = True
        callbacks.append(log_evaluation(int(verbose)))

        evals_result = {}
        callbacks.append(record_evaluation(evals_result))

        self._Booster = train(
            params=params,
            train_set=train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=eval_names,
            fobj=self._fobj,
            feval=eval_metrics_callable,
            init_model=init_model,
            feature_name=feature_name,
            callbacks=callbacks
        )

        if evals_result:
            self._evals_result = evals_result
        else:  # reset after previous call to fit()
            self._evals_result = None

        if self._Booster.best_iteration != 0:
            self._best_iteration = self._Booster.best_iteration
        else:  # reset after previous call to fit()
            self._best_iteration = None

        self._best_score = self._Booster.best_score

        self.fitted_ = True

        # free dataset
        self._Booster.free_dataset()
        del train_set, valid_sets
        return self

    @property
    def n_features_(self):
        """:obj:`int`: The number of features of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_features found. Need to call fit beforehand.')
        return self._n_features

    @property
    def n_features_in_(self):
        """:obj:`int`: The number of features of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_features_in found. Need to call fit beforehand.')
        return self._n_features_in

    @property
    def best_score_(self):
        """:obj:`dict`: The best score of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No best_score found. Need to call fit beforehand.')
        return self._best_score

    @property
    def best_iteration_(self):
        """:obj:`int` or :obj:`None`: The best iteration of fitted model if ``early_stopping()`` callback has been specified."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No best_iteration found. Need to call fit with early_stopping callback beforehand.')
        return self._best_iteration

    @property
    def objective_(self):
        """:obj:`str` or :obj:`callable`: The concrete objective used while fitting this model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No objective found. Need to call fit beforehand.')
        return self._objective

    @property
    def booster_(self):
        """Booster: The underlying Booster of this model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No booster found. Need to call fit beforehand.')
        return self._Booster

    @property
    def evals_result_(self):
        """:obj:`dict` or :obj:`None`: The evaluation results if validation sets have been specified."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No results found. Need to call fit with eval_set beforehand.')
        return self._evals_result

    @property
    def feature_importances_(self):
        """:obj:`array` of shape = [n_features]: The feature importances (the higher, the more important).

        .. note::

            ``importance_type`` attribute is passed to the function
            to configure the type of importance values to be extracted.
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self._Booster.feature_importance(importance_type=self.importance_type)

    @property
    def feature_name_(self):
        """:obj:`array` of shape = [n_features]: The names of features."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No feature_name found. Need to call fit beforehand.')
        return self._Booster.feature_name()

class LightGBMCV:
    def __init__(self,
        lgbm_type: str,
        splitter: Callable,
        eval_metric = None,
        eval_sets: dict = None,
        observe_sets: dict = None,
        separate_observation_split: bool = True,
        early_stopping_rounds: int = None,
        return_cv_models: bool = False,
        refit_model: bool = False,
        verbose: bool = True):
        self.lgbm_type = lgbm_type
        self.splitter = splitter  
        self.eval_metric = eval_metric
        self.eval_sets = eval_sets
        self.observe_sets = observe_sets
        self.separate_observation_split = separate_observation_split
        self.early_stopping_rounds = early_stopping_rounds
        self.return_cv_models = return_cv_models
        self.refit_model = refit_model
        self.verbose = verbose

    def fit(self, X, y, categorical_feature='auto', params=None):
        clf = False
        ordi = False
        reg = False
        metric = None
        alias = None
        metric_names = []
        metric_values = []
        self.classes_ = y.unique()
        if self.lgbm_type=='LGBMClassifier':
            metric = 'binary_logloss'
            if len(self.classes_)>2:
                metric = 'multi_logloss'
            alias = 'log_loss'
            clf = True
            self.estimator_ = LGBMClassifier
            y=y.astype(int)
        elif self.lgbm_type=='LGBMOrdinal':
            metric = 'log_loss'
            alias = 'log_loss'
            ordi = True
            self.estimator_ = LGBMOrdinal
            y=y.astype(int)
        elif self.lgbm_type=='LGBMRegressor':
            metric = 'l2'
            alias = 'l2'
            reg = True
            self.estimator_ = LGBMRegressor
        else:
            raise ValueError('lgbm_type must be LGBMClassifier, LGBMRegressor, or LGBMOrdinal')
        if not self.eval_metric:
            self.eval_metric = []
        elif isinstance(self.eval_metric, dict):
            self.eval_metric = [self.eval_metric]
        elif not isinstance(self.eval_metric, Iterable):
            self.eval_metric = [self.eval_metric]
        if metric not in self.eval_metric:
            self.eval_metric.insert(0,metric)
        for m in self.eval_metric:
            if isinstance(m, str):
                metric_names.append(m)
                metric_values.append(m)
            # elif isinstance(m, dict):
            #     for k, v in m.items():
            #         metric_names.append(k)
            #         if callable(v):
            #             hack = lambda y, x, name=k: (name, v(y, x), False)
            #             _v = lambda y, x: hack(y,x)
            #         else:
            #             _v = v
            #         metric_values.append(_v)
            # elif isinstance(m, Iterable):
            #     k=m[0]
            #     v=m[1]
            #     metric_names.append(k)
            #     if callable(v):
            #         hack = lambda y, x, name=k: (name, v(y, x), False)
            #         _v = lambda y, x: hack(y,x)
            #     else:
            #         _v = v
            #     metric_values.append(_v)
            else:
                raise ValueError('For now, eval_metric can only be a string or list of strings. Callables are yet to be implemented.')
        
        if self.lgbm_type=='LGBMClassifier':
            metric_values = ['binary_error' if v=='error' else v for v in metric_values]
            if len(y.unique())>2:
                metric_values = ['multi_error' if v=='error' else v for v in metric_values]

        

        self.result_dict_ = {'train_scores': {m: [] for m in metric_names}, 
            'test_scores': {m: [] for m in metric_names},
            'best_iteration': [],
            'fit_times': [],
            'train_size': [],
            'test_size': [],
            'train_scores_mean': {},
            'test_scores_mean': {}
            }
        if params:
            self.result_dict_['specified_params'] = params
        if self.eval_sets:
            for k in self.eval_sets.keys():
                self.result_dict_[f'{k}_scores'] = {m: [] for m in metric_names}
                self.result_dict_[f'{k}_scores_mean'] = {}
        if self.observe_sets:
            for k in self.observe_sets.keys():
                self.result_dict_[f'{k}_{alias}_scores'] = []
        if self.return_cv_models:
            self.cv_models_ = []
        ifold = 0
        for train_i, test_i in self.splitter.split(X,y):
            if self.early_stopping_rounds:
                _callbacks = [
                    early_stopping(self.early_stopping_rounds, 
                        first_metric_only=True,
                        verbose=False)]
            if self.verbose:
                str_ = ' -- Fold {}/{}'.format(ifold+1, self.splitter.get_n_splits())

                if ifold > 0:
                    for i in range(len(str_)):
                        print('\b', end='')
                        sys.stdout.flush()

                print(str_, end="")
                sys.stdout.flush()

                if ifold+1 == self.splitter.get_n_splits():

                    for i in range(len(str_)):
                        print('\b', end='')
                        sys.stdout.flush()

                    print('', end='\r')
                    sys.stdout.flush()

            if params:
                _est = self.estimator_(**params)
            else:
                _est = self.estimator_()
            X_train = X.iloc[train_i]
            y_train = y.iloc[train_i]
            X_test = X.iloc[test_i]
            y_test = y.iloc[test_i]
            if clf or ordi:
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
                X_test = X_test[y_test.isin(y_train.unique())]
                y_test = y_test[y_test.isin(y_train.unique())]
            self.result_dict_['train_size'].append(train_i.shape[0])
            self.result_dict_['test_size'].append(test_i.shape[0])
            eval_set=[(X_train, y_train),
                        (X_test, y_test)]
            if self.eval_sets:
                for v in self.eval_sets.values():
                    eval_set.append((v[0].iloc[test_i], v[1].iloc[test_i]))
            ts = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                _est.fit(
                    X_train, y_train, 
                    eval_set=eval_set, 
                    eval_metric=metric_values, 
                    categorical_feature=categorical_feature, 
                    callbacks=_callbacks
                )
            self.result_dict_['fit_times'].append(time.time()-ts)
            if _est.best_iteration_:
                self.result_dict_['best_iteration'].append(_est.best_iteration_)
            train_name = list(_est.best_score_.keys())[0]
            test_name = list(_est.best_score_.keys())[1]
            for i, m in enumerate(metric_names):
                _k=m

                if m=='error':
                    _k=metric_values[i]

                # Very specific bug between binary and multi logloss
                if _k not in _est.best_score_[train_name].keys():
                    _k = 'binary_logloss'

                self.result_dict_['train_scores'][m].append(_est.best_score_[train_name][_k])
                self.result_dict_['test_scores'][m].append(_est.best_score_[test_name][_k])
                if self.eval_sets:
                    for j, k in enumerate(self.eval_sets.keys()):
                        res_name = list(_est.best_score_.keys())[j+2]
                        self.result_dict_[f'{k}_scores'][m].append(_est.best_score_[res_name][m])
            if self.observe_sets:
                for k, v in self.observe_sets.items():
                    X_obs = v[0]
                    y_obs = v[1]
                    if self.separate_observation_split:
                        test_i = list(self.splitter.split(X_obs,y_obs))[i][1]
                    X_test = X_obs.iloc[test_i]
                    y_test = y_obs.iloc[test_i]
                    if clf or ordi:
                        try:
                            s = log_loss(_est._le.transform(y_test), _est.predict_proba(X_test), labels=_est._le.classes_)
                            self.result_dict_[f'{k}_{alias}_scores'].append(s)
                        except:
                            self.result_dict_[f'{k}_{alias}_scores'].append(9999)
                    elif reg:
                        s = mean_squared_error(y_test, _est.predict(X_test))
                        self.result_dict_[f'{k}_{alias}_scores'].append(s)
            if self.return_cv_models:
               self.cv_models_.append(_est)
            ifold+=1
        for m in metric_names:
            self.result_dict_['train_scores_mean'][m] = sum([self.result_dict_['train_scores'][m][i]*self.result_dict_['train_size'][i] for i in range(len(self.result_dict_['train_scores'][m]))])/sum(self.result_dict_['train_size'])
            self.result_dict_['test_scores_mean'][m] = sum([self.result_dict_['test_scores'][m][i]*self.result_dict_['test_size'][i] for i in range(len(self.result_dict_['test_scores'][m]))])/sum(self.result_dict_['test_size'])
            if self.eval_sets:
                for j, k in enumerate(self.eval_sets.keys()):
                    res_name = list(_est.best_score_.keys())[j+2]
                    self.result_dict_[f'{k}_scores_mean'][m] = sum([self.result_dict_[f'{k}_scores'][m][i]*self.result_dict_['test_size'][i] for i in range(len(self.result_dict_[f'{k}_scores'][m]))])/sum(self.result_dict_['test_size'])
        self.result_dict_[f'train_{alias}'] = self.result_dict_['train_scores_mean'][metric]
        self.result_dict_[f'test_{alias}'] = self.result_dict_['test_scores_mean'][metric]
        self.train_score_ = self.result_dict_[f'train_{alias}']
        self.test_score_ = self.result_dict_[f'test_{alias}']
        if self.eval_sets:
            for k in self.eval_sets.keys():
                self.result_dict_[f'{k}_{alias}'] = self.result_dict_[f'{k}_scores_mean'][metric]
        if self.observe_sets:
            for k in self.observe_sets.keys():
                self.result_dict_[f'{k}_{alias}'] = sum([self.result_dict_[f'{k}_{alias}_scores'][i]*self.result_dict_['test_size'][i] for i in range(len(self.result_dict_['test_scores'][m]))])/sum(self.result_dict_['test_size'])
        self.result_dict_['params'] = _est.get_params()
        self.result_dict_['fit_times_mean'] = sum(self.result_dict_['fit_times'])/len(self.result_dict_['fit_times'])
        if 'best_iteration' in self.result_dict_.keys():
            if self.result_dict_['best_iteration']:
                self.best_iteration_max_ = max(self.result_dict_['best_iteration'])
        if self.refit_model:
            print('refitting final model')
            _est = self.estimator_()
            if hasattr(self, best_iteration_max_):
                if not params:
                    params = {}
                params['n_estimators'] = self.best_iteration_max_
            if params:
                _est.set_params(**params)
            _est.fit(X, y)
            self.refitted_model_ = _est
        return self

class LightGBMGridSearchCV:
    def __init__(self, 
        lgbm_type, param_grid, 
        splitter=None,
        save_name=None,
        eval_metric=None, 
        early_stopping_rounds=None):
        self.lgbm_type=lgbm_type
        self.param_grid = param_grid
        self.splitter = splitter  
        self.save_name = save_name
        self.eval_metric = eval_metric 
        self.early_stopping_rounds = early_stopping_rounds


    def fit(self, X, y, categorical_feature='auto',
        refit_models = False,
        refit_best_model=False):
        cv =  LightGBMCV(
            lgbm_type=self.lgbm_type, 
            splitter=self.splitter, 
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            return_cv_models=False,
            refit_model = refit_models)
        self.results_=[]
        self.refitted_models_ = {}
        if self.lgbm_type=='LGBMClassifier':
            alias = 'log_loss'
            _estimator = LGBMClassifier
        elif self.lgbm_type=='LGBMOrdinal':
            alias = 'log_loss'
            _estimator = LGBMOrdinal
        elif self.lgbm_type=='LGBMRegressor':
            alias = 'l2'
            _estimator = LGBMRegressor
        if self.save_name:
            try:
                self.results_=pickle.load(open(f'{self.save_name}.pickle','rb'))
                print(f'found result file: {self.save_name}.pickle')
            except:
                print(f'file  {self.save_name}.pickle not found, starting fresh')
        _params = copy.copy(self.param_grid)
        if isinstance(_params, dict):
            for k, v in self.param_grid.items():
                if (isinstance(v, str)) or (not isinstance(v, Iterable)):
                    _params[k] = [v]
        else:
            for i, d in enumerate(self.param_grid):
                for k, v in d.items():
                    if (isinstance(v, str)) or (not isinstance(v, Iterable)):
                        _params[i][k] = [v]
        _param_grid = ParameterGrid(_params)
        for i, p in enumerate(_param_grid):
            if i+1>len(self.results_):    
                print(f'Running trial {i} of {len(_param_grid)}: {p}')
                cv.fit(X,y,categorical_feature, params=p)
                self.results_.append(cv.result_dict_)
                if self.save_name:
                    pickle.dump(self.results_,open(f'{self.save_name}.pickle','wb'))
                if refit_models:
                    self.refitted_models_[i] = cv.refitted_model_
        self.best_iteration_ = np.argmin(np.array([v[f'test_{alias}'] for v in self.results_]))
        self.best_result_ = self.results_[self.best_iteration_]
        self.best_score_ = self.best_result_[f'test_{alias}']
        _params = self.best_result_[f'params']
        self.best_params_=_params
        if 'best_iteration' in self.best_result_.keys():
            if self.best_result_['best_iteration']:
                self.best_iteration_max_ = max(self.best_result_['best_iteration'])
        _params['n_estimators'] = self.best_iteration_max_
        if refit_best_model:
            if refit_models:
                if len(refit_models = len(enumerate(_param_grid))):
                    self.best_model_ = self.refitted_models_[self.best_iteration_]
            else:
                print(f'Refitting best model with {_params}')
                self.best_model_ = _estimator(**_params)
                self.best_model_.fit(X,y)
        return self

def emae(y_true, preds):
    if preds.shape[0] == len(y_true):
        try:
            if preds.shape[1]<=1:
                raise ValueError('''preds must have width of n_classes''')
        except:
            raise ValueError(f'''preds must have shape (len(y_true), J) or (len(y_true)*J,). 
            It appears preds has shape len(y_true,), i.e. {preds.shape}''')
        y_prob=preds
    elif preds.shape[0]%len(y_true)==0:
        y_prob = preds.reshape(int(preds.shape[0]/len(y_true)),-1).T
    else:
        raise ValueError(f'preds is weird shape of {preds.shape}')
    diffs = np.abs(np.tile(np.arange(y_prob.shape[1]), (y_prob.shape[0], 1)) - np.tile(y_true, (y_prob.shape[1],1)).T)
    return ((diffs*y_prob).sum(axis=1)).mean()


def emse(y_true, preds):
    if preds.shape[0] == len(y_true):
        try:
            if preds.shape[1]<=1:
                raise ValueError('''preds must have width of n_classes''')
        except:
            raise ValueError(f'''preds must have shape (len(y_true), J) or (len(y_true)*J,). 
            It appears preds has shape len(y_true,), i.e. {preds.shape}''')
        y_prob=preds
    elif preds.shape[0]%len(y_true)==0:
        y_prob = preds.reshape(int(preds.shape[0]/len(y_true)),-1).T
    else:
        raise ValueError(f'preds is weird shape of {preds.shape}')
    diffs = (np.tile(np.arange(y_prob.shape[1]), (y_prob.shape[0], 1)) - np.tile(y_true, (y_prob.shape[1],1)).T)**2
    return ((diffs*y_prob).sum(axis=1)).mean()