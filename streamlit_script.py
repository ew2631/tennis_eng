"""
Streamlit Housing App Demo

Make sure to install Streamlit with `pip install streamlit`.

Run `streamlit hello` to get started!

To run this app:

1. cd into this directory
2. Run `streamlit run streamlit_script.py`
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from model_results import *
import pickle
from PIL import Image
from sqlalchemy import create_engine
import sqlite3
import seaborn as sns
# We begin with Parts 3, 6, and 7, but uncomment the code in each of the other parts and save to see how the Streamlit application updates in your browser.

#TODO: Create the one hot encoded to be able to do predictions: https://stackoverflow.com/questions/54786266/prediction-after-one-hot-encoding
#TODO: In order to do that, also need to pull elo_winner/elo_loser
# 1. User inputs name
# 2. Search through original dataset and find all matches between two players
# 3. Select match with most recent date--> get that elo_winner/elo_loser
# 4. If they havent played set to 1500 for both

def get_model_ready(df, cat_vars, numeric_vars):
    target_var=df['Upset']
    cat_features=pd.get_dummies(df[cat_vars], drop_first=True)
    numeric_features=df[numeric_vars]
    all_features=pd.concat([numeric_features,cat_features],1)
    X_train, X_test, y_train, y_test = train_test_split(all_features, target_var, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

engine = create_engine(r'sqlite:///C:\Users\emwang\Downloads\tennis.db')
df = pd.read_sql('SELECT * FROM updated_data;', engine)
df['Year']=df['Year'].apply(lambda x: int(x))

## PART 2 - Markdown Syntax
#
# st.write(
# '''
### Markdown Syntax
# You can use Markdown syntax to style your text. For example,
#
## Main Title
### Subtitle
#### Header
#
# **Bold Text**
#
# *Italics*
#
# Ordered List
#
# 1. Apples
# 2. Oranges
# 3. Bananas
#
# [This is a link!](https://docs.streamlit.io/en/stable/getting_started.html)
#
# '''
# )

################### Introduction ################


img = Image.open("Tennis_Racket_and_Balls.jpg")
st.image(img,width=600)

st.write(
    '''
    ### ATP Tennis Matches
    My goal is to build a match prediction tool for ATP matches.\
    I’ll be using the dataset of all ATP matches from 2000 to 2022 \
    with roughly 30,000 matches 
    ''')

################### Introduction ################
st.write(
    '''
    Here is a  table showing the raw data used to build the model
    ''')

st.dataframe(df.head(10))
st.write(
    '''
    Some of the variables are tournament related, such as the tournament \
    that was played, the tournament level (Series), and the date it is played.\
    There are also player related statistics such as the Elo Ranking. \
    This is a well known probability calculation that takes into account player \
    ranking to determine a match outcome.In terms of interpretation, if a player \
    has an Elo rating of 1,800 and his opponent has a rating of 2,000, the probability \
    of the lower Elo rating player winning becomes is 24.1%
    ''')

st.write(
    '''
    ## ATP Tennis Matches
    Input a year and you can find out the top 10 tournaments with the most upsets that year
    ''')

year = st.number_input('Calendar Year', value=2001)
t=df.groupby(['Tournament','Year'],as_index=False)['Upset'].value_counts()
t.rename(columns={"count": "Num of Upsets"}, inplace=True)
tourn_upsets=pd.DataFrame(t[t['Upset']==1]).sort_values(by='Num of Upsets',ascending=False)
tourn_upsets=tourn_upsets.loc[:, tourn_upsets.columns != 'Upset']
tourn_upsets=tourn_upsets[tourn_upsets['Year']==year][:10]
st.dataframe(tourn_upsets, width=800, height=400)
#plt.bar(x=tourn_upsets['Num of Upsets'], height=400, tick_label=tourn_upsets['Tournament'].values)
#plt.show()

################### Line Graph ################

st.write(
'''
#### Upsets occur when a lower ranked player beats a higher ranked player. Here's a graph showing \
how upsets have trended since 2000 
'''
)

a=pd.DataFrame(df.groupby(['Year'],as_index=False)['Upset'].value_counts(normalize=True))
a=a[a['Upset']==1]

#if i wanted both lines plotted
#sns.lineplot(x ='Year', y = 'proportion',hue='upset',data = a)

fig = plt.figure(figsize=(10, 4))
g=sns.lineplot(x ='Year', y = 'proportion',data = a)
g.set_xlabel("Year", fontsize = 14)
g.set_ylabel("% Upsets", fontsize = 14)
g.set_xlim(min(a['Year']), max(a['Year']))
st.pyplot(fig)

st.write(
''' There seems to be a downward trend of Upsets from  2000 to 2015 \
but in the last 7 years there has been a significant increase from 33% \
to 37-38%
'''
)

st.write(
    '''
    ## Train and Compare Models
    Now let's create a model to predict whether an upset will occur
    '''
)

from sklearn.model_selection import train_test_split

cat_vars= ["Series","Court", "Surface","Round","Tournament","Comment",'Bestof']
numeric_vars=['Year','elo_winner','elo_loser']
X_train, X_test, y_train, y_test=get_model_ready(df, cat_vars,numeric_vars)
scores=final_comparison(models, X_test, y_test)
#models={'Decision_Tree': clf,'Random_Forest': rf,'XGBoost': xgb}
dectree = pickle.load(open('Decision_Tree.pkl','rb'))
rf=pickle.load(open('Random_Forest.pkl','rb'))
xgb=pickle.load(open('Random_Forest.pkl','rb'))

predictions=rf.predict(X_test)

st.write(
    '''
    ### Confusion Matrix
    ##### The confusion matrix shows the performance of the model, both \
    what the model got right (true positive=2631, true negatives=6455) and \
    what the model got wrong (false positives=933, false negatives=1484)
    '''
)
matrix_fig = plt.figure(figsize=(10, 4))
#matrix_fig, matrix_ax = plt.subplots()
st.pyplot(make_confusion_matrix(rf, X_test, y_test,class_labels=['not upset','upset']))

#st.pyplot(make_confusion_matrix(dectree, X_test, y_test,class_labels=['not upset','upset']))
roc_auc=metrics.roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

st.write(
    '''
    ### ROC Curve
    ##### The ROC Curve combines the different performance metrics to show \
    the tradeoff between TP Rate and FP Rate. Generally, a model's performance\
    is compared to a 45 degree line (the dotted pink line), which is complete guessing \
    
    '''
)
st.write('ROC Curve Score:{a}'.format(a=roc_auc))
roc_curve=plot_roc(dectree, X_test, y_test)
st.pyplot(roc_curve)
roc_curve.show()



# st.write(
#     '''
#     '''
# )
