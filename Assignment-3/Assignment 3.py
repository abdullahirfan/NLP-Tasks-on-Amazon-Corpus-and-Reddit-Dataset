
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sc
import gensim as g
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import csv
from sklearn.naive_bayes import MultinomialNB


# In[2]:


import csv
with open("training_pos.csv") as f:
    reader = csv.reader(f)
    stopwords_train_pos = []
    for row in reader:
        stopwords_train_pos.append(row)
        
with open("training_neg.csv") as f:
    reader = csv.reader(f)
    stopwords_train_neg = []
    for row in reader:
        stopwords_train_neg.append(row)


# In[3]:


for j in range(len(stopwords_train_pos)):
    for i in range(len(stopwords_train_pos[j])):
        stopwords_train_pos[j][i]=stopwords_train_pos[j][i].replace("['","")
        stopwords_train_pos[j][i]=stopwords_train_pos[j][i].replace("'","")
        #stopwords_train_pos[j][i]=stopwords_train_pos[j][i].replace(" '","")
        stopwords_train_pos[j][i]=stopwords_train_pos[j][i].replace("']","")
        stopwords_train_pos[j][i]=stopwords_train_pos[j][i].strip()


# In[4]:


for j in range(len(stopwords_train_neg)):
    for i in range(len(stopwords_train_neg[j])):
        stopwords_train_neg[j][i]=stopwords_train_neg[j][i].replace("['","")
        stopwords_train_neg[j][i]=stopwords_train_neg[j][i].replace("'","")
        #stopwords_train_neg[j][i]=stopwords_train_neg[j][i].replace(" '","")
        stopwords_train_neg[j][i]=stopwords_train_neg[j][i].replace("']","")
        stopwords_train_neg[j][i]=stopwords_train_neg[j][i].replace("]","")
        stopwords_train_neg[j][i]=stopwords_train_neg[j][i].strip()


# In[16]:


corpus=stopwords_train_neg+stopwords_train_pos


# In[18]:


from gensim.models import Word2Vec
print('Training Word2Vec on Combined corpus of positive and negative words (with stopwwords)')
word2vec = Word2Vec(corpus, min_count=3)
print('Words similar to Good:')
word2vec.wv.most_similar("good", topn=20)
print('Words similar to Bad:')
word2vec.wv.most_similar("bad", topn=20)
