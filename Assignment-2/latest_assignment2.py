import numpy as np
import scipy as sc
import gensim as g
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn import model_selection, naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer
import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, metrics
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    

	files=[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]]

	def load_data(file):
		def retreive_from_csv(words):
			ls = [i for i in words]
			return ls
		
		f = open(file)
		lists=[]
	 
		for j in f:
			lists.append((retreive_from_csv(eval(j))))
		 
		return lists


	# In[9]:
	print('Loading files')

	train_pos_list = load_data(files[0])
	train_neg_list=load_data(files[1])

	val_pos_list = load_data(files[2])
	val_neg_list=load_data(files[3])

	test_pos_list=load_data(files[4])
	test_neg_list=load_data(files[5])

	print('Done:  Files Loaded')

	# ### Defining training, testing and val data
	print('\n')
	print('Defining training, testing and val data')
	# In[10]:


	x_train=train_pos_list+train_neg_list
	x_test=test_pos_list+test_neg_list
	x_val=val_pos_list+val_neg_list

	x_train=[" ".join(sent) for sent in x_train]
	x_test=[" ".join(sent) for sent in x_test]
	x_val=[" ".join(sent) for sent in x_val]

	print('Done:  Defining training, testing and val data')

	# ### Defining outcomes
	print('\n')
	print('Defining outcomes')
	# In[46]:


	y_train=[]
	y_test=[]
	y_val=[]
	for row in range(len(train_pos_list)):
		y_train.append(1)
		 

	for row in range(len(train_neg_list)):
		y_train.append(0)
	   

	for row in range(len(test_pos_list)):
		y_test.append(1)

	for row in range(len(test_neg_list)):
		y_test.append(0)
		 

	for row in range(len(val_pos_list)):
		y_val.append(1)
		

	for row in range(len(val_neg_list)):
		y_val.append(0)
	
	print('Done:  Defining outcomes')
	
	# ### Vectorizing and Developing best model for unigrams
	print('\n')
	print('Vectorizing for unigrams')
	# In[57]:


	count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 1))
	count_vect.fit(x_train)
		
		
	x_train_unigram = count_vect.transform(x_train)
	x_valid_unigram = count_vect.transform(x_val)
	x_test_unigram = count_vect.transform(x_test)
	print('Done : Vectorization done for unigrams')


	# In[74]:

	print('\n')
	print("Developing baseline model unigram")
	baseline_unigram = naive_bayes.MultinomialNB(alpha=1)
	baseline_unigram.fit(x_train_unigram, y_train)
	predictions_unigram=baseline_unigram.predict(x_test_unigram)
	print("Accuracy of baseline unigrams :",round(metrics.accuracy_score(y_test, predictions_unigram),4))
	print('Done : Developing baseline model unigram')

	# In[83]:

	print('\n')
	print("Performance tuning on validation")

	# Instantiate model 
	grid_model = naive_bayes.MultinomialNB()

	param_grid = { 
		'alpha':  [2,5,10,20,30],
	}
	print('\n')
	print('Ignore Error:')
	# Train the model on training data using GridSearch
	gridsearch_unigram = GridSearchCV(estimator=grid_model, param_grid=param_grid)
	print('\n')
	
	gridsearch_unigram.fit(x_valid_unigram, y_val)
	print("Best Parameters at: ",gridsearch_unigram.best_params_)
	print("Done: Performance tuning on validation")
	

	# In[3]:

	print('\n')
	print('Training model on best fit parameter for unigram @ ',gridsearch_unigram.best_params_ )
	a = gridsearch_unigram.best_params_.get("alpha", "")

	best_unigram = naive_bayes.MultinomialNB(alpha=a)
	best_unigram.fit(x_train_unigram, y_train)
	predictions_unigram_best=best_unigram.predict(x_test_unigram)
	print("Accuracy of Best model on unigrams :",round(metrics.accuracy_score(y_test, predictions_unigram_best),4))
	print('Done: Training model on best fit parameter for unigram')

	# In[ ]:


	# ### Vectorizing and Developing best model for bigrams
	print('\n')
	print('Vectorizing for bigrams')
	# In[57]:


	count_vect = CountVectorizer(analyzer='word', ngram_range = (2, 2))
	count_vect.fit(x_train)
		
		
	x_train_bigram = count_vect.transform(x_train)
	x_valid_bigram = count_vect.transform(x_val)
	x_test_bigram = count_vect.transform(x_test)
	print('Done: Vectorization done for bigrams')


	# In[33]:

	print('\n')
	print("Developing baseline model bigram")
	baseline_bigram = naive_bayes.MultinomialNB(alpha=1)
	baseline_bigram.fit(x_train_bigram, y_train)
	predictions_bigram=baseline_bigram.predict(x_test_bigram)
	print("Accuracy of baseline bigrams :",round(metrics.accuracy_score(y_test, predictions_bigram),4))
	print("Done: Developing baseline model bigram")

	# In[58]:

	print('\n')
	print("Performance tuning on validation")

	# Instantiate model 
	grid_model = naive_bayes.MultinomialNB()

	 
	# Train the model on training data using GridSearch
	print('\n')
	print('Ignore Error:')
	gridsearch_bigram = GridSearchCV(estimator=grid_model, param_grid=param_grid)
	print('\n')
	gridsearch_bigram.fit(x_valid_bigram, y_val)
	print("Best Parameters at: ",gridsearch_bigram.best_params_)
	print("Done: Performance tuning on validation")
	

	# In[60]:
	print('\n')
	print('Training model on best fit parameter for bigram @ ',gridsearch_bigram.best_params_ )
	a = gridsearch_bigram.best_params_.get("alpha", "")

	best_bigram = naive_bayes.MultinomialNB(alpha=a)
	best_bigram.fit(x_train_bigram, y_train)
	predictions_bigram_best=best_bigram.predict(x_test_bigram)
	print("Accuracy of Best model on bigrams :",round(metrics.accuracy_score(y_test, predictions_bigram_best),4))
	print('Done: Training model on best fit parameter for bigram')


	# In[ ]:

	print('\n')
	# ### Vectorizing and Developing best model for unigram + bigrams
	print('Vectorizing Unigram + Bigrams')
	# In[57]:


	count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 2))
	count_vect.fit(x_train)
		
		
	x_train_unigram_bigram = count_vect.transform(x_train)
	x_valid_unigram_bigram = count_vect.transform(x_val)
	x_test_unigram_bigram = count_vect.transform(x_test)
	print('Vectorization done for unigram + bigrams')


	# In[33]:

	print('\n')
	print("Developing baseline model unigram + bigrams")
	baseline_unigram_bigram = naive_bayes.MultinomialNB(alpha=1)
	baseline_unigram_bigram.fit(x_train_unigram_bigram, y_train)
	predictions_unigram_bigram=baseline_unigram_bigram.predict(x_test_unigram_bigram)
	print("Accuracy of baseline unigram + bigrams :",round(metrics.accuracy_score(y_test, predictions_unigram_bigram),4))
	print("Done: Developing baseline model unigram + bigrams")

	# In[58]:

	print('\n')
	print("Performance tuning on validation")

	# Instantiate model 
	grid_model = naive_bayes.MultinomialNB()

	 
	# Train the model on training data using GridSearch
	print('\n')
	print('Ignore Error:')
	gridsearch_unigram_bigram = GridSearchCV(estimator=grid_model, param_grid=param_grid)
	print('\n')
	gridsearch_unigram_bigram.fit(x_valid_unigram_bigram, y_val)
	print("Best Parameters at: ",gridsearch_unigram_bigram.best_params_)
	print("Done: Performance tuning on validation")

	# In[60]:
	print('\n')
	print('Training model on best fit parameter for unigram + bigrams @ ',gridsearch_unigram_bigram.best_params_ )
	a = gridsearch_unigram_bigram.best_params_.get("alpha", "")

	best_unigram_bigram = naive_bayes.MultinomialNB(alpha=a)
	best_unigram_bigram.fit(x_train_unigram_bigram, y_train)
	predictions_unigram_bigram_best=best_unigram_bigram.predict(x_test_unigram_bigram)
	print("Accuracy of Best model on unigram + bigrams :",round(metrics.accuracy_score(y_test, predictions_unigram_bigram_best),4))
	print('Done: Training model on best fit parameter for unigram + bigrams')
	print('\n')
	print('\n')
	print('### End of Script ###')