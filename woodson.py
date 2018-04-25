from __future__ import print_function

# system libraries
import os
import sys
import json
import re
import pickle
import main

# machine learning libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

target = 'Attended Michigan'

def XGModelFit(XGBModel, df_reviews, features, plot, useTrainCV=True, cv_folds=7, early_stopping_rounds=25):
    if useTrainCV:
        xgb_param = XGBModel.get_xgb_params()
        xgtrain = xgb.DMatrix(df_reviews[features].values, label=df_reviews[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=XGBModel.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        XGBModel.set_params(n_estimators=cvresult.shape[0])

    XGBModel.fit(df_reviews[features], df_reviews[target], eval_metric='auc')

    df_review_predictions = XGBModel.predict(df_reviews[features])
    df_review_predprob = XGBModel.predict_proba(df_reviews[features])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Train AUC Score: %.4g" % metrics.roc_auc_score(df_reviews[target], df_review_predprob))
    print("Accuracy : %.4g" % metrics.accuracy_score(df_reviews[target].values, df_review_predictions))

    if plot:
        plot_importance(XGBModel, importance_type='gain')
        plt.show()

# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

def model_pipeline(X_train, y_train, X_test, y_test):
    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, X_train, y_train, X_test, y_test))

    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                           tol=1e-3), X_train, y_train, X_test, y_test))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty), X_train, y_train, X_test, y_test))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet"), X_train, y_train, X_test, y_test))

    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid(), X_train, y_train, X_test, y_test))

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01), X_train, y_train, X_test, y_test))
    results.append(benchmark(BernoulliNB(alpha=.01), X_train, y_train, X_test, y_test))

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                      tol=1e-3))),
      ('classification', LinearSVC(penalty="l2"))]), X_train, y_train, X_test, y_test))

    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()

def model_generation():
    # test split
    test_split = 0.7

    df_twitter_2016 = main.main(2016)
    df_twitter_2017 = main.main(2017)

    dfs_twitter = [df_twitter_2016, df_twitter_2017]

    features = ['Intercept', 'Miles from AA', 'First Offer',
                'Last Offer', 'Official Visit', 'Last Official Visit', 'In-State',
                'michFavToAllTweetRatio', 'michTweetToAllTweetRatio', 'michOverallTweetRatio',
                'michNativeRTweetRatio', 'michNativeTweetRatio']

    #intercept term
    df_twitter_2016.insert(0, 'Intercept', 1, allow_duplicates=True)
    df_twitter_2017.insert(0, 'Intercept', 1, allow_duplicates=True)

    X_train = df_twitter_2016[features].values
    y_train = df_twitter_2016[target].values

    X_test = df_twitter_2017[features].values
    y_test = df_twitter_2017[target].values

    df_twitter = pd.concat(dfs_twitter, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

    X = df_twitter[features].values
    y = df_twitter[target].values

    '''
    Data Exploration
    '''
    print("Number of features: " + str(len(features)))
    group_by_help_2016 = df_twitter_2016.groupby(target).mean()
    group_by_help_2017 = df_twitter_2017.groupby(target).mean()

    '''
    Model Generation
    '''
    cross = False
    # using 7 fold cross validation

    print ("Logistic Regression Model Results: ")
    penalty = ["l1", "l2"]
    for pen in penalty:
        print("Using " + pen + " Regularization")
        logModel = LogisticRegression(penalty=pen, C=1)

        if cross:
            print("Performing Cross Validation")
            logModel = LogisticRegression(penalty=pen)
            params = {'C': [.1, .5, 1, 5, 10]}
            logModel = GridSearchCV(logModel, params, scoring='neg_log_loss', refit=True, cv=7)

            logModel.fit(X_train, y_train)
            bestParams = logModel.best_params_

            logModel = LogisticRegression(penalty=pen, C=bestParams['C'])

        logModel.fit(X_train, y_train)

        print("Score for training set")
        print(str(logModel.score(X_train, y_train)))
        #print("Null score for training set")
        #print(str(y_train.mean()))

        '''
        print("Coeficents for training set: ")
        df_coef = pd.DataFrame(data=logModel.coef_, columns=features, dtype=None, copy=False)
        with pd.option_context('display.max_rows', None, 'display.max_columns', len(features)):
            f = open('model_data_analysis_log.txt', 'w')
            f.write(df_coef)
        '''

        print("Predicted Labels: ")
        predicted = logModel.predict(X_test)
        print(predicted)

        print("Predicted probabilities for each label: ")
        probs = logModel.predict_proba(X_test)
        print(probs)

        print("Print accuracy score: ")
        print(metrics.accuracy_score(y_test, predicted))
        '''
        print("Print roc_auc_score: ")
        print(metrics.roc_auc_score(y_test, probs[:, 1]))
        '''

        print("Confusion matrix: ")
        print(metrics.confusion_matrix(y_test, predicted))
        print("Classification report: ")
        print(metrics.classification_report(y_test, predicted))

    if cross:
        print("Using Cross Validation to see if results hold up across all of the training set + model generalizes well: ")
        penalty = ["l1", "l2"]
        for pen in penalty:
            print("Using " + pen + " Regularization")
            scores = cross_val_score(LogisticRegression(penalty=pen), X, y, scoring='accuracy', cv=7)
            print(scores)
            print(scores.mean())

    # Random Forest
    print ("Random Forest: " )
    rf = RandomForestClassifier(n_estimators=500, oob_score=True) #oob_score makes cv unnecessary for paramater tuning
    rf.fit(X_train, y_train)

    '''
    print("Feature Importantces for training set: ")
    feature_imp = np.reshape(rf.feature_importances_, (1, len(features)))
    df_coef = pd.DataFrame(data=feature_imp, columns=features, dtype=None, copy=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', len(features)):
        f = open('model_data_analysis_random_forest.txt', 'w')
        f.write(df_coef)
    '''

    print("Predicted Labels: ")
    predicted = rf.predict(X_test)
    print(predicted)

    print("Predicted probabilities for each label: ")
    probs = rf.predict_proba(X_test)
    print(probs)

    accuracy = metrics.accuracy_score(y_test, predicted)
    print('Out-of-bag score estimate:' + str(rf.oob_score_))
    print('Mean accuracy score: ' + str(accuracy))

    print("Confusion matrix: ")
    print(metrics.confusion_matrix(y_test, predicted))

    if cross:
        print("Using Cross Validation to see if results hold up across all of the training set + model generalizes well: ")
        scores = cross_val_score(RandomForestClassifier(n_estimators=500, oob_score=True), X, y, scoring='accuracy', cv=7)
        print(scores)
        print(scores.mean())

    # XGBoosted
    colsample_bytree = 0.9
    subsample = 0.9
    num_estimators = 90
    max_depth = 3
    min_child_weight = 5
    gamma = 0.2
    reg_alpha = 0.01

    '''
    Cross Validation
    '''
    if cross:
        tree_params_test_one = {
            'max_depth': range(1, 9, 2),
            'min_child_weight': range(1, 6, 2)
        }

        tree_search = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,gamma=0, n_estimators=90, max_depth=5,
                                                        min_child_weight=1, nthread=4,subsample=0.8, colsample_bytree=0.8,
                                                        objective='binary:logistic', scale_pos_weight=1),
                                param_grid=tree_params_test_one, scoring='roc_auc', n_jobs=4, iid=False, cv=7)

        tree_search.fit(df_twitter[features], df_twitter[target])

        print("Best Tree Params: ")
        print(tree_search.best_params_)

        max_depth = tree_search.best_params_['max_depth']
        min_child_weight = tree_search.best_params_['min_child_weight']

        print("Best Model Score: ")
        print(tree_search.best_score_)

        # tune gamma
        gamma_param = {
            'gamma': [i / 10.0 for i in range(0, 5)]
        }
        gamma_search = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=num_estimators, max_depth=max_depth,
                                                        min_child_weight=min_child_weight, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                        objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                        seed=27),
                                param_grid=gamma_param, scoring='roc_auc', n_jobs=4, iid=False, cv=7)
        gamma_search.fit(df_twitter[features], df_twitter[target])
        print("Best Tree Params: ")
        print(gamma_search.best_params_)

        gamma = gamma_search.best_params_['gamma']

        print("Best Model Score: ")
        print(gamma_search.best_score_)

        # tune subsample and colsample_bytree

        subsample_colsample_bytree = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }
        subsample_colsample_bytree_search = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=num_estimators, max_depth=max_depth,
                                                        min_child_weight=min_child_weight, gamma=gamma, subsample=0.8, colsample_bytree=0.8,
                                                        objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                        seed=27),
                                param_grid=subsample_colsample_bytree, scoring='roc_auc', n_jobs=4, iid=False, cv=7)

        subsample_colsample_bytree_search.fit(df_twitter[features], df_twitter[target])
        print("Best Tree Params: ")
        print(subsample_colsample_bytree_search.best_params_)

        subsample = subsample_colsample_bytree_search.best_params_['subsample']
        colsample_bytree = subsample_colsample_bytree_search.best_params_['colsample_bytree']

        print("Best Model Score: ")
        print(subsample_colsample_bytree_search.best_score_)

        # Tune regularization paramater
        reg_params = {
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
        }
        reg_search = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=num_estimators, max_depth=max_depth,
                                                        min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
                                                        objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                        seed=27),
                                param_grid=reg_params, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        reg_search.fit(df_twitter[features], df_twitter[target])
        print("Best Tree Params: ")
        print(reg_search.best_params_)

        reg_alpha = reg_search.best_params_['reg_alpha']

        print("Best Model Score: ")
        print(reg_search.best_score_)

    # reduce learning rate and generate many trees
    # get non linear relationships
    modelXG = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        booster='gbtree')

    modelXG.fit(X_train, y_train)

    plot_importance(modelXG, importance_type='gain', xlabel='Information Gain') # plot importance of features by information gain
    plt.show()

    # make predictions for test data
    y_pred = modelXG.predict(X_test)
    predictions = [round(value) for value in y_pred]

    print("Predicted probabilities for each label: ")
    probs = modelXG.predict_proba(X_test)
    print(probs)

    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

if __name__ == '__main__':
    model_generation()
