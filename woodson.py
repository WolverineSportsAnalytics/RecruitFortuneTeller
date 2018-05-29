from __future__ import print_function

# system libraries
import main

# machine learning libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# helper libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

target = 'Attended Michigan'

def XGModelFit(XGBModel, df_twitter, features, plot, useTrainCV=True, cv_folds=7, early_stopping_rounds=25):
    if useTrainCV:
        xgb_param = XGBModel.get_xgb_params()
        xgtrain = xgb.DMatrix(df_twitter[features].values, label=df_twitter[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=XGBModel.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        XGBModel.set_params(n_estimators=cvresult.shape[0])

    XGBModel.fit(df_twitter[features], df_twitter[target], eval_metric='auc')

    df_review_predictions = XGBModel.predict(df_twitter[features])
    df_review_predprob = XGBModel.predict_proba(df_twitter[features])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Train AUC Score: %.4g" % metrics.roc_auc_score(df_twitter[target], df_review_predprob))
    print("Accuracy : %.4g" % metrics.accuracy_score(df_twitter[target].values, df_review_predictions))

    if plot:
        xgb.plot_importance(XGBModel, importance_type='weight')
        plt.show()

def model_generation():
    df_twitter_2016 = main.main(2016)
    df_twitter_2017 = main.main(2017)

    official_visit_analysis = True

    if official_visit_analysis:
        df_twitter_2016 = df_twitter_2016.query('OfficialVisit==1')
        df_twitter_2017 = df_twitter_2017.query('OfficialVisit==1')

    dfs_twitter = [df_twitter_2016, df_twitter_2017]

    features = ['Intercept', 'Miles from AA', 'First Offer',
                'Last Offer', 'OfficialVisit', 'Last Official Visit', 'In-State',
                'michFavToAllTweetRatio', 'michTweetToAllTweetRatio',
                'michNativeRTweetRatio', 'michNativeTweetRatio']

    if official_visit_analysis:
        features = ['Intercept', 'Miles from AA', 'First Offer',
                    'Last Offer', 'Last Official Visit', 'In-State',
                    'michFavToAllTweetRatio', 'michTweetToAllTweetRatio',
                    'michNativeRTweetRatio', 'michNativeTweetRatio']

    #intercept term
    df_twitter_2016.insert(0, 'Intercept', 1, allow_duplicates=True)
    df_twitter_2017.insert(0, 'Intercept', 1, allow_duplicates=True)

    df_twitter_tot = pd.concat([df_twitter_2016, df_twitter_2017])

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
    cross = True
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

        log_label = "Logisitc Regression " + pen + " Pred Label"

        predicted = logModel.predict(X_test)
        df_twitter_2017[log_label] = predicted

        probs = logModel.predict_proba(X_test)
        log_label_prob_no = "Logisitc Regression " + pen + " No Prob"
        log_label_prob_yes = "Logisitc Regression " + pen + " Yes Prob"
        df_twitter_2017[log_label_prob_no] = probs[:,0]
        df_twitter_2017[log_label_prob_yes] = probs[:,1]

        print("Print accuracy score: ")
        print(metrics.accuracy_score(y_test, predicted))

        df_cm = pd.DataFrame(metrics.confusion_matrix(y_test, predicted), index=[i for i in ["Other", "Michigan"]],
                             columns=[i for i in ["Other", "Michigan"]])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)

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

    # Naive Bayes
    print ("K-Neighbors Classifier: ")
    knn = KNeighborsClassifier(n_neighbors=7)

    # TODO: cross validation
    if cross:
        params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
        knn = GridSearchCV(knn, params, scoring='neg_log_loss', refit=True, cv=7)

        knn.fit(X_train, y_train)
        bestParams = knn.best_params_

        knn = KNeighborsClassifier(n_neighbors=bestParams['n_neighbors'])

    knn.fit(X_train, y_train)

    print("Score for training set")
    print(str(knn.score(X_train, y_train)))

    knn_pred_label = "KNN Predictions"

    predicted = knn.predict(X_test)
    df_twitter_2017[knn_pred_label] = predicted

    probs = knn.predict_proba(X_test)
    knn_prob_no = "KNN Prob No"
    knn_prob_yes = "KNN Prob Yes"
    df_twitter_2017[knn_prob_no] = probs[:, 0]
    df_twitter_2017[knn_prob_yes] = probs[:, 1]

    print("Print accuracy score: ")
    print(metrics.accuracy_score(y_test, predicted))

    df_cm = pd.DataFrame(metrics.confusion_matrix(y_test, predicted), index=[i for i in ["Other", "Michigan"]],
                         columns=[i for i in ["Other", "Michigan"]])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    print("Classification report: ")
    print(metrics.classification_report(y_test, predicted))

    # Random Forest
    print ("Random Forest: " )
    rf = RandomForestClassifier(n_estimators=500, oob_score=True) #oob_score makes cv unnecessary for paramater tuning
    rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    df_twitter_2017["Random Forest Model Predicted Labels"] = predicted

    probs = rf.predict_proba(X_test)
    random_f_prob_no = "Random Forest No Prob"
    random_f_prob_yes = "Random Forest Yes Prob"
    df_twitter_2017[random_f_prob_no] = probs[:, 0]
    df_twitter_2017[random_f_prob_yes] = probs[:, 1]

    accuracy = metrics.accuracy_score(y_test, predicted)
    print('Out-of-bag score estimate:' + str(rf.oob_score_))
    print('Mean accuracy score: ' + str(accuracy))

    df_cm = pd.DataFrame(metrics.confusion_matrix(y_test, predicted), index=[i for i in ["Other", "Michigan"]],
                         columns=[i for i in ["Other", "Michigan"]])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    print("Classification report: ")
    print(metrics.classification_report(y_test, predicted))

    if cross:
        print("Using Cross Validation to see if results hold up across all of the training set + model generalizes well: ")
        scores = cross_val_score(RandomForestClassifier(n_estimators=500, oob_score=True), X, y, scoring='accuracy', cv=7)
        print(scores)
        print(scores.mean())

    # XGBoosted
    colsample_bytree = 0.8
    subsample = 0.7
    num_estimators = 140
    max_depth = 1
    min_child_weight = 5
    gamma = 0
    reg_alpha = 0.1

    '''
    Cross Validation
    '''
    if cross:
        tree_params_test_one = {
            'max_depth': range(1,2,9),
            'min_child_weight': range(1, 6, 2)
        }

        tree_search = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,gamma=0, n_estimators=90, max_depth=5,
                                                        min_child_weight=1, nthread=4,subsample=0.8, colsample_bytree=0.8,
                                                        objective='binary:logistic', scale_pos_weight=1),
                                param_grid=tree_params_test_one, scoring='roc_auc', n_jobs=4, iid=False, cv=7)

        tree_search.fit(df_twitter_tot[features], df_twitter_tot[target])

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
        gamma_search.fit(df_twitter_tot[features], df_twitter_tot[target])
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

        subsample_colsample_bytree_search.fit(df_twitter_tot[features], df_twitter_tot[target])
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
        reg_search.fit(df_twitter_tot[features], df_twitter_tot[target])
        print("Best Tree Params: ")
        print(reg_search.best_params_)

        reg_alpha = reg_search.best_params_['reg_alpha']

        print("Best Model Score: ")
        print(reg_search.best_score_)

    # reduce learning rate and generate many trees
    # get non linear relationships
    modelXG = XGBClassifier(
        learning_rate=0.001,
        n_estimators=1000,
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

    XGModelFit(modelXG, df_twitter_tot, features, True, useTrainCV=True, cv_folds=7, early_stopping_rounds=25)

    modelXG.fit(X_train, y_train)

    # make predictions for test data
    y_pred = modelXG.predict(X_test)
    df_twitter_2017["XGB Predicted Labels"] = y_pred

    probs = modelXG.predict_proba(X_test)
    random_f_prob_no = "XGB No Prob"
    random_f_prob_yes = "XGB Yes Prob"
    df_twitter_2017[random_f_prob_no] = probs[:, 0]
    df_twitter_2017[random_f_prob_yes] = probs[:, 1]

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    df_cm = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), index=[i for i in ["Other", "Michigan"]],
                         columns=[i for i in ["Other", "Michigan"]])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    print("Classification report: ")
    print(metrics.classification_report(y_test, y_pred))

    df_twitter_2017.to_csv(path_or_buf='recruits_2017_results.csv')

if __name__ == '__main__':
    model_generation()
