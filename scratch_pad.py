import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
import sqlite3

engine=create_engine(r'sqlite:///C:\Users\emwang\Downloads\tennis.db')
print(engine.table_names())
doctors_data=pd.read_sql('SELECT * FROM updated_data;', engine)
print(doctors_data.head(10))

#Rank_Delta=Winner Rank- Loser Rank so if Delta>0, that means the winner has a higher numerical rank and the match is an upset
def upset(row):
    if row['Rank_Delta']>0:
        return 1
    else:
        return 0
# grid_search= joblib.load("grid_search_xgb.pkl")
# print (grid_search.best_params_)
#
# cols=['mean_train_score','std_train_score','mean_test_score', 'std_test_score', 'rank_test_score']
#
# means=grid_search.cv_results_['mean_test_score']
# stds=grid_search.cv_results_['std_test_score']

# for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#     print(mean, std, params)

#print(a.items()[0])
#print(grid_search.cv_results_.keys())

#print(clf.best_params_)