

import streamlit as st
from model_results import *
import pickle
from PIL import Image
from sqlalchemy import create_engine
import sqlite3
import seaborn as sns


from sklearn.model_selection import train_test_split

@st.cache(persist=True)
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
cat_vars= ["Series","Court", "Surface","Round","Tournament","Comment",'Bestof']
numeric_vars=['Year','elo_winner','elo_loser']
X_train, X_test, y_train, y_test=get_model_ready(df, cat_vars,numeric_vars)

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


img = Image.open("tennis.jpg")
st.image(img,width=800)

st.write(
    '''
    ### ATP Tennis Matches
    My goal is to build a match prediction tool for ATP matches.\
    I’ll be using the dataset of all ATP matches from 2000 to 2022 \
    with roughly 57,000 matches 
    ''')

################### Introduction ################
st.write(
    '''
    Here is a  table showing the raw data used to build the model
    ''')

all_features=cat_vars + numeric_vars
st.dataframe(df[all_features].head(10))

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

################### Table ################

st.write(
    '''
    ### Most "Upsetting" Tournaments
    Input a year and you can find out the top 10 tournaments with the most upsets that year
    ''')

year = st.number_input('Calendar Year', value=2001)
t=df.groupby(['Tournament','Year'],as_index=False)['Upset'].value_counts()
t.rename(columns={"count": "Num of Upsets"}, inplace=True)
tourn_upsets=pd.DataFrame(t[t['Upset']==1]).sort_values(by='Num of Upsets',ascending=False)
tourn_upsets=tourn_upsets.loc[:, tourn_upsets.columns != 'Upset']
tourn_upsets=tourn_upsets[tourn_upsets['Year']==year][:10]
st.dataframe(tourn_upsets)


################### Line Graph ################

st.write(
'''
### Trend of Match Upsets
Upsets occur when a lower ranked player beats a higher ranked player. Here's a graph showing \
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

################### Models ################

st.write(
    '''
    ### Train and Compare Models
    We trained and compared performance of 3 different models \
     on predicting whether an upset will occur
    '''
)

dectree = pickle.load(open('Decision_Tree.pkl','rb'))
rf=pickle.load(open('Random_Forest.pkl','rb'))
xgb=pickle.load(open('XGBoost.pkl','rb'))
models={'Decision_Tree': dectree,'Random_Forest': rf,'XGBoost': xgb}
scores=final_comparison(models, X_test, y_test)
st.dataframe(scores)

st.write('''XGBoost clearly performed the best with both the highest precision \
        and ROC. Since this is a somewhat imbalanced dataset, it's important \
        that both metrics are considered
''')

################### Performance Analysis ################

st.write(
    '''
    ### Confusion Matrix
    The confusion matrix shows the performance of the model, both \
    what the model got right (true positive=2954, true negatives=6606) and \
    what the model got wrong (false positives=782, false negatives=1161)
    '''
)
matrix_fig = plt.figure(figsize=(10, 4))
st.pyplot(make_confusion_matrix(xgb, X_test, y_test,class_labels=['not upset','upset']))


st.write(
    '''
    ### ROC Curve
    The ROC Curve combines the different performance metrics to show \
    the tradeoff between TP Rate and FP Rate. Generally, a model's performance\
    is compared to a 45 degree line (the dotted pink line), which is complete guessing.\
    If you want to see how the AUC of the model changes with different probability \
    thresholds, click the box!
    '''
)

show_slider = st.checkbox('I want to adjust the probability threshold', value=True)

if show_slider:
    threshold = st.slider('Probability Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.1 )
else:
    threshold=0.5

roc_auc=metrics.roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]>= threshold)
st.write('ROC Curve Score: {a}'.format(a=round(roc_auc, 2) ))
roc_curve=plot_roc(xgb, X_test, y_test,model_type='tree', threshold=threshold)
st.pyplot(roc_curve)


st.write(
    '''
    ### Model Feature Importance
    Since XGBoost is the best model, we can look \
    at the most important features that contributed to the overall \
    model performance
    '''
)

coef_table, fig=generate_coef_table(list(X_train.columns), xgb, 'tree', 10)
st.pyplot(fig)

st.write('''We find that the Elo rating variable and whether a a specific tournament \
            was played were the best determinants for an upset
        ''')

#Create the one hot encoded to be able to do predictions: https://stackoverflow.com/questions/54786266/prediction-after-one-hot-encoding
#In order to do that, also need to pull elo_winner/elo_loser
# 1. User inputs name
# 2. Search through original dataset and find all matches between two players
# 3. Select match with most recent date--> get that elo_winner/elo_loser
# 4. If they havent played set to 1500 for both