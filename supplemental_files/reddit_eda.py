#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:41:50 2024

@author: stephanie
"""

import pandas as pd
import matplotlib as plt
import seaborn as sns

from nltk.tag import pos_tag


#%% data import
rawdf = pd.read_csv('/Users/stephanie/data_projects/takehome-stephanie-workingcopy/askscience_data.csv')

df = rawdf.drop(["Unnamed: 0", "url"], axis=1, inplace=True)
#df.drop(df['score'].idxmax(), axis=0, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[~df['tag'].isin(['Meta', 'META'])]

#df_mod = df[df.author == 'AskScienceModerator'] #Checked mod, keep these in


#%% Put together a basic eda dataframe for reference
eda = pd.DataFrame(data={'Feature':df.columns, 
                         'dtype':df.dtypes, 
                         'null_count':df.isna().sum(), 
                         }
                   ).reset_index()

# TODO: Remove this, i'm getting distracted
#def get_numerical_aggregate(df, method):
#    
#    aggregate_col = []
#    for col in df.columns:
#        if df[col].dtype in ['float64', 'datetime64[ns]']:
#            myValue = df[col].method()
#        else: 
#            myValue = df[col].str.len().method()
#        
#    aggregate_col.append(myValue)
#    return aggregate_col



    
maxes=[]
for col in df.columns:
    if col in ["body", "title", "author", "tag"]:
        myMax = df[col].str.len().max()
    else:
        myMax = df[col].max()
        
    maxes.append(myMax)
    
eda["Max"] = maxes

mins=[]
for col in df.columns:
    if col in ["body", "title", "author", "tag"]:
        myMin = df[col].str.len().min()
    else:
        myMin = df[col].min()
        
    mins.append(myMin)
    
means = []
for col in df.columns:
    if col in ["body", "title", "author", "tag"]:
        myMean = df[col].str.len().mean()
    elif col=="datetime":
        myMean = 0
    else:
        myMean = df[col].mean()
        
    means.append(myMean)
    
modes = []
for col in df.columns:

    myMode = df[col].mode()
    myMode = myMode[0]
        
    modes.append(myMode)

    
eda["Max"] = maxes
eda["Mins"] = mins
eda["Means"] = means
eda["Modes"] = modes

del([maxes, mins, means, modes, myMax, myMean, myMin, myMode])

#%% some histograms


ax = df['score'].plot.hist(bins=12, alpha=0.5)



#%% more specific EDA

# wanted to see the highest score. It's a net neutrality post.
# Second highest is a stephen hawking megathread 
max = df[df.score==df.score.max()]

# see avg score in topics
tag_scores = df.groupby('tag')['score'].mean()
tag_scores_med = df.groupby('tag')['score'].median().sort_values(ascending=False)
df.groupby('tag')['score'].mean().plot.barh()
#things tagged meta are much higher
# some topics def higher than others


df.groupby(pd.cut(df.a, 80))['inc'].mean().plot()

# author post count
authorcount = df.groupby('author').size().sort_values(ascending=False)
authorcount = authorcount.to_frame(name='author_post_count')
authorcount.iloc[0]=0 # I want to remove counts for Deleted users
authorcount.iloc[1]=0 # i want to neutralize the mods' counts, too
df['authorcount'] = df['author'].replace(authorcount.index, authorcount['author_post_count'])

#nonlinear relationships.
# need to take the 0 posts out when I do this
# 12 looks like an outlier, is it? yep, it's this one person 
author_postcount_scores = df.groupby('authorcount')['score'].mean()
author_count_scores_med = df.groupby('authorcount')['score'].median()
df_i = rawdf[df.author == 'inquilinekea'] # yeah it's this one author

# Length of title
df['title_len'] = df['title'].str.len()
title_len_scores = df.groupby(pd.cut(df.title_len, 10))['score'].mean()
title_len_scores = df.groupby(pd.cut(df.title_len, 10))['score'].median()
df.groupby(pd.cut(df.title_len, 10))['score'].median().plot.barh()
title_check = df[df.title_len > 270]
title_check = df[(df.title_len < 270) & (df.title_len >241)]
df.groupby(pd.qcut(df.title_len, 10))['score'].median().plot.barh()

# Length of post
df['body_len'] = df['body'].str.len().fillna(0)
body_len_scores = df.groupby(pd.cut(df.body_len, 10))['score'].mean()
body_len_scores = df.groupby(pd.cut(df.body_len, 40))['score'].median()
body_len_scoresq = df.groupby(pd.qcut(df.body_len, 40, duplicates='drop'))['score'].mean()
df.groupby(pd.cut(df.body_len, 10, duplicates='drop'))['score'].median().plot.barh()
df.groupby(pd.qcut(df.body_len, 10, duplicates='drop'))['score'].mean().plot.barh()
title_check = df[df.title_len > 270]
title_check = df[(df.title_len < 270) & (df.title_len >241)]

# title has AMA
df['is_ama'] = df['title'].str.contains("AMA")
is_ama_med = df.groupby('is_ama')['score'].median().sort_values(ascending=False)

#megathread
df['is_megathread'] = df['title'].str.contains("Megathread", case=False)
df.groupby('is_megathread')['score'].median().sort_values(ascending=False).plot.barh()
is_megathread_med = df.groupby('is_ama')['score'].median().sort_values(ascending=False)

# title has question mark - I could refine this
df['contains_question'] = df['title'].str.contains("?", regex=False)
df.groupby('contains_question')['score'].median().sort_values(ascending=False).plot.barh()
df_no_q = df[df.contains_question==False] #they're all the mod, though

#capitalize your title
df['no_capitalization'] = df['title'].str.islower()
df.groupby('no_capitalization')['score'].median().sort_values(ascending=False).plot.barh()


#%% some parts of speech tagging




