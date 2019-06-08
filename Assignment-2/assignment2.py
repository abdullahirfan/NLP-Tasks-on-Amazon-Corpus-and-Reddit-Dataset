# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 18:15:24 2019

@author: Abdullah Irfan
"""

import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, metrics
from hypopt import GridSearch

if __name__ == "__main__":
    var_type_names = ["train_pos_path", "train_neg_path", "valid_pos_path", \
                  "valid_neg_path", "test_pos_path", "test_neg_path"]
    
    labels = ["train_labels", "validation_labels", "test_labels"]
    
    set_types = ['training_set', 'validation_set', 'test_set']
    
    files = {key:[] for key in var_type_names}
    label_lists = {key:[] for key in labels}
    data_list = {key:[] for key in set_types}
    
    j = 0
    # Open each file and put the entries into a list
    for i in range(1, 7):
        input_path = sys.argv[i]
        
        fd = open(input_path)
        
        for x in fd:
            files[var_type_names[i - 1]].append(x)
            

            if (i % 2) == 0:
                label_lists[labels[j]].append(0) # For negative review labels 
            else:
                label_lists[labels[j]].append(1)  # For positive review labels
                
        if (i % 2) == 0:
            j += 1
   
        fd.close()

    print("Defined outcomes to each pos/neg file")
    
    print("Combining the positives and negative files for train, test, val")
    data_list[set_types[0]] = files[var_type_names[0]] + files[var_type_names[1]]
    data_list[set_types[1]] = files[var_type_names[2]] + files[var_type_names[3]]
    data_list[set_types[2]] = files[var_type_names[4]] + files[var_type_names[5]]
    
    print("Shuffling rows to stop trailing 0s and 1s for training")
    data_list[set_types[0]], label_lists[labels[0]] = shuffle(data_list[set_types[0]], label_lists[labels[0]])
    data_list[set_types[1]], label_lists[labels[1]] = shuffle(data_list[set_types[1]], label_lists[labels[1]])
    data_list[set_types[2]], label_lists[labels[2]] = shuffle(data_list[set_types[2]], label_lists[labels[2]])
    
    
    print("Making unigrams and vectorizing text into numbers")
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 1))
    count_vect.fit(data_list[set_types[0]] + data_list[set_types[1]])
    
    
    x_train = count_vect.transform(data_list[set_types[0]])
    x_valid = count_vect.transform(data_list[set_types[1]])
    x_test = count_vect.transform(data_list[set_types[2]])
    
    print("Training and Tuning Model using GridSearchCV for unigrams:")
    opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1,2,3,4,5]}])
 
    print("Optimum parameters found for unigrams")
    opt.fit(x_train, label_lists[labels[0]], x_valid, label_lists[labels[1]])
    print('Accuracy for unigrams: {:.3f}'.format(opt.score(x_test, label_lists[labels[2]])))
    
    
    print("Making bigrams and vectorizing text into numbers")
    count_vect = CountVectorizer(analyzer='word', ngram_range = (2, 2))
    count_vect.fit(data_list[set_types[0]] + data_list[set_types[1]])
    
    # Transform datasets to sparse matrix representations
    x_train = count_vect.transform(data_list[set_types[0]])
    x_valid = count_vect.transform(data_list[set_types[1]])
    x_test = count_vect.transform(data_list[set_types[2]])
    
    print("Training and Tuning Model using GridSearchCV for Bigrams:")
    opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1,2,3,4,5]}])
    print("Optimum parameters found for Bigrams")
    opt.fit(x_train, label_lists[labels[0]], x_valid, label_lists[labels[1]])
    print('Accuracy for Bigrams: {:.3f}'.format(opt.score(x_test, label_lists[labels[2]])))
    
    print("Making Unigrams + Bigrams and vectorizing text into numbers")
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 2))
    count_vect.fit(data_list[set_types[0]] + data_list[set_types[1]])
    
    x_train = count_vect.transform(data_list[set_types[0]])
    x_valid = count_vect.transform(data_list[set_types[1]])
    x_test = count_vect.transform(data_list[set_types[2]])
    
    print("Training and Tuning Model using GridSearchCV for Unigrams + Bigrams:")
    opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1,2,3,4,5]}])
    opt.fit(x_train, label_lists[labels[0]], x_valid, label_lists[labels[1]])
    print('Accuracy for Unigrams + Bigrams: {:.3f}'.format(opt.score(x_test, label_lists[labels[2]])))
    