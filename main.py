# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from model_results import *
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import preprocessing, linear_model, model_selection, metrics, datasets, base
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import pylint
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import create_engine
import sqlite3


#df.drop(columns=['index','index.1'],inplace=True)
#print(df.columns)

################### Train Test Split ################

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = pd.DataFrame(scaler.transform(X_train),columns=list(X_train.columns))
# X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=list(X_train.columns))

################### Logistic Regression ################

#Establish baseline logistic regression model
# lr = LogisticRegression(solver='liblinear', penalty = 'l2',random_state=42)
# lr.fit(X_train_scaled, y_train)
# lr_predictions=lr.predict(X_test_scaled)

# #Understanding model outcomes
# lr_results=final_comparison([lr], X_test, y_test)
# print(lr_results)

# cm_plt=make_confusion_matrix(lr, X_test_scaled, y_test,class_labels=['not upset','upset'])
# cm_plt.show()
# roc_curve=plot_roc(lr, X_test, y_test)
# roc_curve.show()

# coef_table = pd.DataFrame(list(X_train_scaled.columns),columns=['Variable']).copy()
# coef_table.insert(len(coef_table.columns),"Coefs",lr.coef_.transpose())
# coefficient_table= pd.concat([coef_table.sort_values('Coefs', ascending=False)[:5], coef_table.sort_values('Coefs', ascending=False)[-5:]])
# print(coefficient_table)
# coefficient_table.to_csv('coefficient_table.csv')

################### Decision Tree ################

#For Decision Trees, going to tune hyperparameters to get best model performance
# check_params={
#                 'max_leaf_nodes': list(range(1000,11000,1000)),
#                 'criterion': ['gini','entropy'],
#                 'max_depth': np.arange(50 ,100,5),
#                 'min_impurity_decrease': np.arange(0.0,0.1,0.01)
#             }
# clf = tree.DecisionTreeClassifier(random_state=65)
# clf.fit(X_train, y_train)
# create_grid=GridSearchCV(clf, param_grid=check_params, cv=4, verbose=10, return_train_score=True, scoring='roc_auc')
# create_grid.fit(X_train, y_train)
# print("Train score for %d fold CV := %3.2f" %(4, create_grid.score(X_train, y_train)))
# print("Test score for %d fold CV := %3.2f" %(4, create_grid.score(X_test, y_test)))
# print ("!!!! best fit parameters from GridSearchCV !!!!")
# print (create_grid.best_params_)
#
# #save grid search results in case we need to debug
# joblib.dump(create_grid, 'grid_search_clf.pkl')
# #
# # #WINNING MODEL
#print(X_train.iloc[2])
#clf = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth=50, max_leaf_nodes= 1000, min_impurity_decrease=0.01, random_state=65)
#clf.fit(X_train, y_train)
# #grid_search= joblib.load("grid_search_xgb.pkl")
# predictions=clf.predict(X_test)
# print('Round 1')
# print(classification_report(y_test, predictions))
# print(roc_auc_score(y_test, predictions))
# score=metrics.accuracy_score(y_test,predictions)
# precision=metrics.precision_score(y_test,predictions)
# recall=metrics.recall_score(y_test,predictions)


# clf_results=final_comparison([clf], X_test, y_test)
# print(clf_results)
#
# cm_plt=make_confusion_matrix(clf, X_test, y_test,class_labels=['not upset','upset'])
# cm_plt.show()
# roc_curve=plot_roc(clf, X_test, y_test)
# roc_curve.show()

# ################### Random Forest ###################

# Valid parameters:
# ['bootstrap', 'ccp_alpha', 'class_weight', 'criterion',
# 'max_depth', 'max_features', 'max_leaf_nodes',
# 'max_samples', 'min_impurity_decrease', 'min_samples_leaf',
# 'min_samples_split', 'min_weight_fraction_leaf',
# 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start'].

# check_params={
#                 'class_weight': ["balanced", {0: 1, 1: 2}]
#                 ,'n_estimators': list(range(200, 1100,100))
#                 ,'max_features': ['sqrt', 'log2', None]
#                 #'max_depth': np.arange(10, 60, 10),
#                 #'min_impurity_decrease': np.arange(0.0,0.1,0.01),
#                 #'criterion': ['mse','mae'],
#             }
# rf = RandomForestClassifier(n_jobs=-1, random_state=65)
#
# create_grid=GridSearchCV(rf, param_grid=check_params, cv=4, verbose=10, return_train_score=True, scoring='roc_auc')
# create_grid.fit(X_train, y_train)
# print("Train score for %d fold CV := %3.2f" %(4, create_grid.score(X_train, y_train)))
# #print("Validation score for %d fold CV := %3.2f" %(4, create_grid.score(X_val, y_val)))
# print ("!!!! best fit parameters from GridSearchCV !!!!")
# print (create_grid.best_params_)

# print(create_grid.cv_results_)

#save your model or results
# joblib.dump(create_grid, 'grid_search_rf.pkl')

#WINNING MODEL
# rf = RandomForestClassifier(class_weight= {0: 1, 1: 2}, max_features= None, n_estimators= 900, random_state=65)
# rf.fit(X_train, y_train)
#
# rf_results=final_comparison([rf], X_test, y_test)
# print(rf_results)
# cm_plt=make_confusion_matrix(rf, X_test, y_test,class_labels=['not upset','upset'])
# cm_plt.show()
# roc_curve=plot_roc(rf, X_test, y_test)
# roc_curve.show()
#
# ################### XGBoost ###################
#
#
# xgb_model = xgb.XGBClassifier(random_state = 89)
# xgb_model.fit(X_train,y_train)
# xgb_params={
#                 'n_estimators': list(range(1000, 2000,500))
#                 ,'max_depth': np.arange(10,12)
#                 , 'eta': [0.0,0.5,1]
#                 , 'subsample': [0.5, 1]
#                 , 'colsample_bytree': [0.5, 1]
#              }
# create_grid=GridSearchCV(xgb_model, param_grid=xgb_params, cv=4, verbose=10, return_train_score=True, scoring='roc_auc')
# create_grid.fit(X_train, y_train)
# print("Train score for %d fold CV := %3.2f" %(4, create_grid.score(X_train, y_train)))
# print("Validation score for %d fold CV := %3.2f" %(4, create_grid.score(X_val, y_val)))
# print ("!!!! best fit parameters from GridSearchCV !!!!")
# print (create_grid.best_params_)
# joblib.dump(create_grid, 'grid_search_xgb.pkl')
# #
# # #WINNING MODEL
# xgb_model = xgb.XGBClassifier(
#      max_depth=10
#     ,random_state=89
#     ,verbosity=3
# )
# xgb_model.fit(X_train, y_train)
#
# xgb_results=final_comparison([xgb_model], X_test, y_test)
# print(xgb_results)
# cm_plt=make_confusion_matrix(xgb_model, X_test, y_test,class_labels=['not upset','upset'])
# cm_plt.show()
# roc_curve=plot_roc(xgb_model, X_test, y_test)
# roc_curve.show()

# ################### Model Comparison ###################

# final_scores = final_comparison([clf, rf,xgb_model], X_test, y_test)
# final_scores.columns=['Decision Tree', 'Random Forest','XGB']
# print(final_scores)
# final_scores.to_csv('model_baseline.csv')

if __name__ == '__main__':
    engine = create_engine(r'sqlite:///C:\Users\emwang\Downloads\tennis.db')
    df = pd.read_sql('SELECT * FROM updated_data;', engine)
    #df=pd.read_csv(r'updated_data.csv')
    #print(list(df.columns))

    #print(list(df.select_dtypes(include=['int']).columns))
    cat_vars= ["Series","Court", "Surface","Round","Tournament","Comment",'Bestof']
    numeric_vars=['Year','elo_winner','elo_loser']
                 # 'CBW', 'CBL', 'GBW', 'GBL', 'IWW', 'IWL',\
                 #  'SBW', 'SBL', 'B365W', 'B365L', 'B&WW', \
                 #  'B&WL', 'EXW', 'EXL', 'PSW', 'PSL', \
                 #  'UBW', 'UBL', 'LBW', 'LBL', 'SJW', 'SJL',\
                 #  'MaxW', 'MaxL', 'AvgW', 'AvgL'
                 #  ]

    X_train, X_test, y_train, y_test=get_model_ready(df, cat_vars,numeric_vars)
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=50, max_leaf_nodes=1000,
                                       min_impurity_decrease=0.01, random_state=65)
    clf.fit(X_train, y_train)
    #joblib.dump(clf, 'Decision_Tree.pkl')
    #model_filename = 'ufo-model.pkl'
    pickle.dump(clf, open('Decision_Tree.pkl', 'wb'))


    #rf = RandomForestClassifier(class_weight= {0: 1, 1: 2}, max_features= None, n_estimators= 900, random_state=65)
    rf = RandomForestClassifier(random_state=65)
    rf.fit(X_train, y_train)
    #joblib.dump(rf, 'Random_Forest.pkl')
    pickle.dump(rf, open('Random_Forest.pkl', 'wb'))

    # xgb = xgb.XGBClassifier(
    #          max_depth=10
    #         ,random_state=89
    #         ,verbosity=3
    #     )

    xgb = xgb.XGBClassifier(random_state=89,verbosity=3)
    xgb.fit(X_train, y_train)
    pickle.dump(xgb, open('XGBoost.pkl', 'wb'))

    #models={'Decision_Tree': clf,'Random_Forest': rf,'XGBoost': xgb}
    # models={'Decision_Tree': clf}
    # for key, value in models.items():
    #     mdl=value.fit(X_train,y_train)
    #     model_filename = str(key)+'.pkl'
    #     joblib.dump(mdl, model_filename)
        # joblib.dump(create_grid, 'grid_search_rf.pkl')

    # scores=final_comparison(models, X_test, y_test)
    # print(scores)
    # coef_table, fig=generate_coef_table(list(X_train.columns), clf, 'tree', 10)
    # print(coef_table)
    # fig.show()
    #model_filename = 'dectree-model.pkl'
    #clf = pickle.load(open('dectree-model.pkl', 'rb'))
    print('Percent Upsets: {a}'.format(a=len(df[df['Upset'] == 1]) / len(df)))
    print('Percent Expected: {a}'.format(a=len(df[df['Upset'] == 0]) / len(df)))
    #predictions = clf.predict(X_test)
    # print(classification_report(y_test, predictions))
    # roc_curve=plot_roc(clf, X_test, y_test)
    # roc_curve.show()
################### END OF CODE ###################
