#!/usr/bin/env python
# coding: utf-8

# # Automatic Questions Tagging System
# ## Project Description:
# 
# 
# Sites like Quora and Stackoverflow, which are especially created to have questions and answers for its users, frequently ask users to provide five words along with their question so that they may be easily categorized. However, people occasionally offer incorrect tags, making it difficult for other users to explore. As a result, they want an automatic question tagging system that can recognize accurate and relevant tags for a user-submitted topic.
# 
# ## Project Dataset:
# Here is the link for the dataset : https://www.kaggle.com/stackoverflow/stacksample 
# The text of 10% of the questions and answers from the Stack Overflow programming Q&A website is included in this dataset.
# 
# This is divided into 3 tables:
# 
# 1. For all non-deleted Stack Overflow questions with an Id that is a multiple of 10, Questions provides the title, body, creation date, closed date (if applicable), score, and owner ID.
# 2. Each of the answers to these questions has a body, a creation date, a score, and an owner ID. The ParentId column references the Questions table.
# 3. Each of these questions has its own set of tags, which are listed under Tags.
# 

# In[71]:


import pandas as pd
import numpy as np
import string
import csv
import os


# ## Load Dataset into Sqlite Database

# In[72]:


from IPython.display import display, HTML
import sqlite3
from sqlite3 import Error

connec=sqlite3.connect("dbfile.db",True)
cursor=connec.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS TAGS(Id INT,Tag TEXT)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS QUESTIONS(
    Id INT,
    OwnerUserId INT,
    CreationDate TEXT,
    ClosedDate TEXT,
    Score INT,
    Title TEXT,
    Body TEXT)''')

question_read_csv = pd.read_csv('~/Downloads/archive 2/Questions.csv', encoding='latin-1',nrows=100000)
tags_read_csv =pd.read_csv('~/Downloads/archive 2/Tags.csv', encoding='latin-1',nrows=100000)


for index,row in tags_read_csv.iterrows():
    cursor.execute('INSERT INTO TAGS(Id,Tag) VALUES (?,?)',list((row['Id'],row['Tag']
                                                                          )))
for index,row in question_read_csv.iterrows():
    cursor.execute('INSERT INTO QUESTIONS(Id,OwnerUserId,CreationDate,ClosedDate,Score,Title,Body) VALUES (?,?,?,?,?,?,?)',list((row['Id']
                                                                         ,row['OwnerUserId'],
                                                                         row['CreationDate'],
                                                                         row['ClosedDate'],row['Score'],
                                                                         row['Title'],row['Body'],
                                                                          )))
connec.commit()


# In[73]:


def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn

        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows

conn = create_connection('dbfile.db')


# In[74]:


sql_statement = "select * from TAGS;"
df_tags = pd.read_sql_query(sql_statement, conn)
display(df_tags.head(5))


# In[75]:


sql_statement = "select * from QUESTIONS;"
df_ques = pd.read_sql_query(sql_statement, conn)
display(df_ques.head(5))


# In[76]:


tmp = df_tags['Tag'].value_counts()
tags_set =  set(tmp[tmp>100].index.tolist())
df_tags = df_tags[df_tags['Tag'].apply(lambda x : x in tags_set)]
gb = df_tags.groupby(by=['Id'],sort=False)
s = gb.apply(lambda grp: [grp['Tag'].values.tolist()])
df_tags = pd.DataFrame(s.to_dict()).T
df_tags.reset_index(inplace=True)
df_tags.columns = ['Id', "Tag"]


# In[77]:


df_ques


# In[78]:


df_ques.drop(columns=[ 'OwnerUserId', 'CreationDate', 'ClosedDate','Score'],axis = 1,inplace=True)


# In[79]:


df_final = df_ques.merge(df_tags, on='Id')


# In[80]:


df_final


# ## Concatenate both Title and Body columns

# In[81]:


df_final['Text'] =  df_final[['Title', 'Body']].agg('.'.join, axis=1)
df_final.drop(columns=['Title','Body','Id'],axis = 1,inplace=True)


# In[82]:


df_final


# ## Clean the Text column - Remove punctuations,lowercase,remove stop words

# In[83]:


def remove_punc(text):
    for punctuation in string.punctuation:
        text_cleaned = text.replace(punctuation,'')
    return text_cleaned


# In[84]:


df_final['Text'] = df_final['Text'].astype(str)
df_final['Text'] = df_final['Text'].apply(remove_punc)
df_final['Text'] = df_final['Text'].str.lower()
df_final['Text'] = df_final['Text'].str.split()
df_final['Text'].head()


# In[85]:


import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def word_lem(text_cleaned):
    text_lemmatized =[lemmatizer.lemmatize(i) for i in text_cleaned]
    return text_lemmatized
df_final['Text'] = df_final['Text'].apply(lambda x: word_lem(x))


# In[86]:


import sys
get_ipython().system('{sys.executable} -m pip install spacy')
get_ipython().system('{sys.executable} -m spacy download en')
import spacy
spacy_load = spacy.load('en_core_web_sm')
from spacy.lang.en import English
stopwords = spacy_load.Defaults.stop_words
df_final['Text'] = df_final['Text'].apply(lambda x:[word for word in x if not word in stopwords])
df_final['Text'].head()                                   


# In[87]:


from sklearn.feature_extraction.text import TfidfVectorizer
df_final['Text']= df_final['Text'].astype(str)
vectorizer = TfidfVectorizer(max_features = 100)
text_vectorized = vectorizer.fit_transform(df_final['Text'].str.lower())


# In[88]:


tags = df_final['Tag'].values.tolist()
final_tags = []
for tag in tags:
    final_tags.append([str(i) for i in tag])


# In[89]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
tags_enc = mlb.fit_transform(final_tags)


# In[ ]:





# ## Split the dataset into train and test data

# In[90]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(text_vectorized.toarray(),tags_enc,test_size=0.2)
print(x_train.shape)
print(y_train.shape)


# ## Model Building

# ### Support Vector Classification

# In[53]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
pred_svc = clf.predict(x_test)


# ### Logistic Regression

# In[48]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression()
clf_log = OneVsRestClassifier(log_classifier).fit(x_train, y_train)
pred_log = clf_log.predict(x_test)


# ### Random Forest Classification

# In[62]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
clf_rf = OneVsRestClassifier(rf_classifier).fit(x_train, y_train)
pred_rf = clf_rf.predict(x_test)


# ## Evaluation Metrics - Jaccard Score and Hamming Loss

# In[56]:


from sklearn.metrics import hamming_loss
def jacard_score(y_true,y_pred):
    jacard_calculated = np.minimum(y_true,y_pred).sum(axis=0) / np.maximum(y_true,y_pred).sum(axis=0)
    return jacard_calculated.mean()*100


# ## Results

# ### Support Vector Classfication 

# In[61]:


print("Jacard Score: {}".format(jacard_score(y_test, pred_svc)))
print("Hamming loss: {}".format(hamming_loss(y_test, pred_svc)))


# ### Logistic Regression

# In[60]:


print("Jacard Score: {}".format(jacard_score(y_test, pred_log)))
print("Hamming loss: {}".format(hamming_loss(y_test, pred_log)))


# ### Random Forest Classification

# In[64]:


print("Jacard Score: {}".format(jacard_score(y_test, pred_rf)))
print("Hamming loss: {}".format(hamming_loss(y_test, pred_rf)))


# In[ ]:




