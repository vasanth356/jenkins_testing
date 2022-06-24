#importing the necessary libraries
import numpy as np
# import pyspark
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR,SVC
import xgboost as xg
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import SGDRegressor, LinearRegression, ElasticNet, SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessRegressor
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import pandas as pd
from sklearn.metrics import r2_score
import hyperopt
import random

def regression_models(train_x,train_y):
# intialzing the data for the training
    X = train_x
    y = train_y

    def regressor(params):
        regressor_type = params['type']
        del params['type']
        if regressor_type == 'svr':
            clf = SVR(**params)
        elif regressor_type == 'xg':
            clf = xg.XGBRegressor(**params)
        elif regressor_type == 'sgd':
            clf = SGDRegressor(**params)
        elif regressor_type == 'lr':
            clf = LinearRegression()
        elif regressor_type == 'lgbm':
            clf = lgb.LGBMRegressor(**params)
        elif regressor_type == 'el':
            clf = ElasticNet(**params)
        elif regressor_type == 'knn':
            clf = KNeighborsRegressor(**params)
        elif regressor_type == 'dc':
            clf = DecisionTreeRegressor(**params)
        elif regressor_type == 'rf':
            clf = RandomForestRegressor(**params)
        elif regressor_type == 'gpr':
            clf = GaussianProcessRegressor(**params)
        else:
            return 0
        return clf.fit(X,y)

    def objective(params):

        clf = regressor(params)
        score_manual = r2_score(y,clf.predict(X))
        r2 = cross_val_score(clf, X, y, scoring = 'r2' ).mean()
        
        # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
        return {'loss': -r2, 'status': STATUS_OK}


    # search space need to be converted to regression
    search_space = hp.choice('regressor_type', [
        {
            'type': 'svr',
            'C': hp.lognormal('SVR_C', 0, 1.0),
            'kernel': hp.choice('kernel', ['linear', 'rbf'])
        },
        {
            'type': 'xg',
            'max_depth': hp.choice('xg_max_depth', np.arange(5, 16, 1, dtype=int)),
            'eta': hp.lognormal('eta', 0, 1.0)
        },
        {
            'type': 'sgd',
            'loss': hp.choice('loss', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'] ),
            'penalty': hp.choice('penalty', ['l2', 'l1', 'elasticnet']      )
        },
        {
            'type': 'lr'
        },
        {
        'type' : 'lgbm',
        'max_depth':   hp.choice('lgbm_max_depth', np.arange(5, 16, 1, dtype=int)),
        'n_estimators':  hp.choice('lgbm_n_estimators', np.arange(100, 500, 100, dtype=int))
        },
   {
        'type' : 'el',
        'alpha':        hp.uniform('el_alpha', 0.0, 1),
        'l1_ratio':     hp.uniform('el_l1_ratio', 0.0, 1)
        },
        {
        'type' : 'knn',
        'n_neighbors':  hp.choice('knn_n_neighbors', np.arange(2, 15, 1, dtype=int)),
        'weights':     hp.choice('knn_weights', ['uniform','distance'])
        },

        {
        'type' : 'dc',
        'max_depth' :   hp.choice('dc_max_depth', np.arange(1, 16, 1, dtype=int)),
        'min_samples_leaf' : hp.choice('min_samples_leaf', np.arange(1, 15, 1, dtype=int))
        },

        {
        'type' : 'rf',
        'max_depth' :   hp.choice('rf_max_depth', np.arange(5, 16, 1, dtype=int)),
        'n_estimators' : hp.choice('n_estimators', np.arange(100, 500, 100, dtype=int))
        },

        {
        'type' : 'gpr',
        'alpha': hp.uniform('alpha', 0.0, 1),
        }

    ])

    # using the tpe algorthim
    algo=tpe.suggest

    # using the spark cluster
    spark_trials = SparkTrials()

    # rnadom state
    rstate = np.random.default_rng(42)

    best_result = fmin( fn=objective, space=search_space, algo=algo, max_evals=2, trials=spark_trials, rstate=rstate)

    best_model_params = hyperopt.space_eval(search_space, best_result)
    best_params = best_model_params.copy()

    return best_params, regressor(best_model_params)

def classification_models(train_x,train_y):
# intialzing the data for the training

    X = train_x
    y = train_y

    def classifier(params):
        classifier_type = params['type']
        del params['type']
        if classifier_type == 'svc':
            clf = SVC(**params)
        elif classifier_type == 'xg':
            clf = XGBClassifier(**params)
        elif classifier_type == 'sgd':
            clf = SGDClassifier(**params)
        elif classifier_type == 'lr':
            clf = LogisticRegression()
        elif classifier_type == 'lgbm':
            clf = lgb.LGBMClassifier(**params)
        elif classifier_type == 'nvb':
            clf = MultinomialNB(**params)
        elif classifier_type == 'knn':
            clf = KNeighborsClassifier(**params)
        elif classifier_type == 'dc':
            clf = DecisionTreeClassifier(**params)
        elif classifier_type == 'rf':
            clf = RandomForestClassifier(**params)
        elif classifier_type == 'adb':
            clf = AdaBoostClassifier(**params)
        else:
            return 0
        return clf.fit(X,y)

    def objective(params):

        clf = classifier(params)
        f1 = cross_val_score(clf, X, y, scoring = 'f1_macro' ).mean()
        # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
        return {'loss': -f1, 'status': STATUS_OK}


    # search space need to be converted to regression
    search_space = hp.choice('classifier_type', [
        {
            'type': 'svc',
            'C': hp.lognormal('svc_C', 0, 1.0),
            'kernel': hp.choice('svc_kernel', ['linear', 'rbf'])
        },
        {
            'type': 'xg',
            'max_depth': hp.choice('xg_max_depth', np.arange(5, 16, 1, dtype=int)),
            'eta': hp.lognormal('xg_eta', 0, 1.0)
        },
        {
            'type': 'sgd',
            'loss': hp.choice('loss', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'] ),
            'penalty': hp.choice('sgd_penalty', ['l2', 'l1', 'elasticnet']      )
        },
        {
            'type': 'lr',
            'C': hp.lognormal('lr_C', 0, 1.0),
            'penalty': hp.choice('lr_penalty', ['l1', 'l2','elasticnet'])
        },
        {
        'type' : 'lgbm',
        'max_depth':   hp.choice('lgbm_max_depth', np.arange(5, 16, 1, dtype=int)),
        'n_estimators':  hp.choice('lgbm_n_estimators', np.arange(100, 500, 100, dtype=int))
        },
        {
        'type' : 'nvb',
        'alpha': hp.uniform('nvb_alpha', 0.0, 1)
        },
        {
        'type' : 'knn',
        'n_neighbors':  hp.choice('knn_n_neighbors', np.arange(2, 15, 1, dtype=int)),
        'weights':     hp.choice('knn_weights', ['uniform','distance'])
        },

        {
        'type' : 'dc',
        'max_depth' :   hp.choice('dc_max_depth', np.arange(1, 16, 1, dtype=int)),
        'min_samples_leaf' : hp.choice('dc_min_samples_leaf', np.arange(1, 15, 1, dtype=int))
        },

        {
        'type' : 'rf',
        'max_depth' :   hp.choice('rf_max_depth', np.arange(5, 16, 1, dtype=int)),
        'n_estimators' : hp.choice('rf_n_estimators', np.arange(100, 500, 100, dtype=int))
        },

        {
        'type' : 'adb',
        'n_estimators': hp.choice('adb_n_estimators', np.arange(100, 500, 100, dtype=int)),
        'algorithm': hp.choice('algorithm', ['SAMME','SAMME.R'])
        }

    ])

    # using the tpe algorthim
    algo=tpe.suggest

    # using the spark cluster
    spark_trials = SparkTrials()
    
    best_result = fmin( fn=objective, space=search_space, algo=algo, max_evals=2, trials=spark_trials)

    best_model_params = hyperopt.space_eval(search_space, best_result)
    best_params = best_model_params.copy()

    return best_params, classifier(best_model_params)
