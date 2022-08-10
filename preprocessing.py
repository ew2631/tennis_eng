import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# import chardet
# file='data/2021.xlsx'
# with open(file, 'rb') as rawdata:
#     result = chardet.detect(rawdata.read(100000))
# print(result)




def combine_files(directory):
    size_lst=[]
    df = pd.DataFrame()

    for filename in os.listdir(directory):
        f=os.path.join(directory, filename)
        each_df=pd.read_excel(f)
        size_lst.append(len(each_df))
        df=pd.concat([df,each_df])
    print('Size of combined files: {a} '.format(a=sum(size_lst)))
    print('Size of final df: {a} '.format(a=len(df)))
    return df


def compute_elo_rankings(data):
    """
    Given the list on matches in chronological order, for each match, computes
    the elo ranking of the 2 players at the beginning of the match

    """
    print("Elo rankings computing...")
    players = list(pd.Series(list(data.Winner) + list(data.Loser)).value_counts().index)
    elo = pd.Series(np.ones(len(players)) * 1500, index=players)
    ranking_elo = [(1500, 1500)]
    for i in range(1, len(data)):
        w = data.iloc[i - 1, :].Winner
        l = data.iloc[i - 1, :].Loser
        elow = elo[w]
        elol = elo[l]
        pwin = 1 / (1 + 10 ** ((elol - elow) / 400))
        K_win = 32
        K_los = 32
        new_elow = elow + K_win * (1 - pwin)
        new_elol = elol - K_los * (1 - pwin)
        elo[w] = new_elow
        elo[l] = new_elol

        ranking_elo.append((elo[data.iloc[i, :].Winner], elo[data.iloc[i, :].Loser]))
        if i % 5000 == 0:
            print(str(i) + " matches computed...")
    ranking_elo = pd.DataFrame(ranking_elo, columns=["elo_winner", "elo_loser"])
    ranking_elo["proba_elo"] = 1 / (1 + 10 ** ((ranking_elo["elo_loser"] - ranking_elo["elo_winner"]) / 400))
    return ranking_elo

def clean_data(df):
    df = df.sort_values("Date")
    df["WRank"] = df["WRank"].replace(np.nan, 0)
    df["WRank"] = df["WRank"].replace("NR", 2000)
    df["LRank"] = df["LRank"].replace(np.nan, 0)
    df["LRank"] = df["LRank"].replace("NR", 2000)
    df["WRank"] = df["WRank"].astype(int)
    df["LRank"] = df["LRank"].astype(int)
    df["Wsets"] = df["Wsets"].astype(float)
    df['Wsets'] = df['Wsets'].replace(np.nan, 0.0)
    df["Lsets"] = df["Lsets"].replace("`1", 1)
    df["Lsets"] = df["Lsets"].astype(float)
    df['Lsets'] = df['Lsets'].replace(np.nan, 0.0)
    #df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
    df['Year'] = df['Date'].apply(lambda x: x.year)
    # new var: Rank_Delta
    df['Rank_Delta'] = df['WRank'] - df['LRank']
    df['Upset']=df.apply(lambda row: upset(row), axis=1)
    #df['col_3'] = df.apply(lambda x: get_sublist(x.col_1, x.col_2), axis=1)
    #df['Elo_Higher'], df['Elo_Lower']=df.apply(lambda x: convert_elo(x),axis=1)
    #df['Elo_Lower']= df.apply(lambda x: convert_elo(x) , axis=1)
    df = df.sort_values("Date")
    df.reset_index(drop=True, inplace=True)
    return df

def convert_elo(row):

    if row['elo_winner'] >= row['elo_loser']:
        a=row['elo_winner']
        b=row['elo_loser']

    else:
        a=row['elo_loser']
        b=row['elo_winner']

    return pd.Series([a, b])


def upset(row):
    if row['Rank_Delta']>0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    directory = 'data'
    df_final = combine_files(directory)
    df_final.sort_values('Date', ascending=True)
    print('Stop Check: After combining files, make sure the data is in chronological order')
    print('First Date: {a}'.format(a=df_final.iloc[1]['Date']))
    print('Last Date: {a}'.format(a=df_final.iloc[-1]['Date']))

    elo_rankings = compute_elo_rankings(df_final)
    elo_rankings = elo_rankings.reset_index()
    df_final = df_final.reset_index()
    df_final = pd.concat([df_final, elo_rankings], 1)
    #print(df_final.info())
    #print(type(df_final['Date'].iloc[1]))
    df_final=clean_data(df_final)
    df_final.sort_values('Date', ascending=True)
    print('Stop Check: After cleaning data, make sure the data is STILL in chronological order')
    print('First Date: {a}'.format(a=df_final.iloc[1]['Date']))
    print('Last Date: {a}'.format(a=df_final.iloc[-1]['Date']))
    print(df_final['Upset'].value_counts())

    df_final.to_csv('updated_data.csv',index=False)

    #print(df_final.info())
    #print(df_final.head(10))
    #print(len(elo_rankings))
    #return df_final.head(10)
    #main()


