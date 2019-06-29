
import io
import string
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import scipy as sc
import sys



# # Assignment-1: Making the files again to restart the assignment

# In[ ]:


if __name__ == "__main__":
    

	files=[sys.argv[1],sys.argv[2]]
	 
	pos=[]
	neg=[]
	def load_data(file):
		for i in range(len(file)):
			if i==0:
				with open(file[i],"r") as p:
					sentences = [j for j in p.read().split('\n') if j]
					for sentence in sentences:
						pos.append(nltk.word_tokenize(sentence))
			else:
				with open(file[i],"r") as n:
					sentences = [z for z in n.read().split('\n') if z]
					for sentence in sentences:
						neg.append(nltk.word_tokenize(sentence))
		print(pos[:2])
		print(neg[:2])


	# In[ ]:


	load_data(files)


	# ### Removing punctuations and lowercase

	# In[ ]:


	def remove_punctuations(ls):
		punctuations = set(string.punctuation) - {'.',"'"}

		punct_removed = [] 
		for sent in ls:
			temp = []
			for i in sent:
				if i.lower() not in punctuations:
					temp.append(i.lower())
			punct_removed.append(temp)      
		return punct_removed


	# In[ ]:


	pos_stopwords= remove_punctuations(pos)
	neg_stopwords=remove_punctuations(neg)


	# ### Making lists without stopwords

	# In[ ]:


	def without_stopword_list(ls):
		sp = set(stopwords.words('english'))
		ls_without_sw=[]
		
		for sent in ls:
			temp = []
			for i in sent:
				if i.lower() not in sp:
					temp.append(i.lower())
			ls_without_sw.append(temp)
		return ls_without_sw


	# In[ ]:


	pos_without_stopwords=without_stopword_list(pos_stopwords)
	neg_without_stopwords=without_stopword_list(neg_stopwords)


	# ### Data Splitting

	# In[ ]:


	def data_split(ls):
		split_1 = int(0.8 * len(ls))
		split_2 = int(0.9 * len(ls))
		
		train = ls[:split_1]
		val = ls[split_1:split_2]
		test = ls[split_2:]
		
		return train,test,val 


	# In[ ]:


	train_pos,test_pos,val_pos= data_split(pos_stopwords)
	train_pos_without_stopwords,test_pos_without_stopwords,val_pos_without_stopwords=data_split(pos_without_stopwords)

	train_neg,test_neg,val_neg= data_split(neg_stopwords)
	train_neg_without_stopwords,test_neg_without_stopwords,val_neg_without_stopwords=data_split(neg_without_stopwords)


	# ### Saving Files

	# In[ ]:


	np.savetxt("train_pos.csv", train_pos, delimiter=",", fmt='%s')
	np.savetxt("test_pos.csv", test_pos, delimiter=",", fmt='%s')
	np.savetxt("val_pos.csv", val_pos, delimiter=",", fmt='%s')
	np.savetxt("train_no_stopword_pos.csv", train_pos_without_stopwords, delimiter=",", fmt='%s')
	np.savetxt("test_no_stopword_pos.csv", test_pos_without_stopwords, delimiter=",", fmt='%s')
	np.savetxt("val_no_stopword_pos.csv", val_pos_without_stopwords, delimiter=",", fmt='%s')

	np.savetxt("train_neg.csv", train_neg, delimiter=",", fmt='%s')
	np.savetxt("test_neg.csv", test_neg, delimiter=",", fmt='%s')
	np.savetxt("val_neg.csv", val_neg, delimiter=",", fmt='%s')
	np.savetxt("train_no_stopword_neg.csv", train_neg_without_stopwords, delimiter=",", fmt='%s')
	np.savetxt("test_no_stopword_neg.csv", test_neg_without_stopwords, delimiter=",", fmt='%s')
	np.savetxt("val_no_stopword_neg.csv", val_neg_without_stopwords, delimiter=",", fmt='%s')