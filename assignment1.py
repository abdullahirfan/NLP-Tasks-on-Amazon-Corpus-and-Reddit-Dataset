import sys
import numpy as np
import scipy as sc
import gensim as g
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from nltk.corpus import stopwords

if __name__ == "__main__":
    
    input_loc = open(sys.argv[1])
    file = input_loc.readlines()
    
       
    """
    Removed whitespaces from beginings and end
    """
    rwhite = [x.strip() for x in file] 
    
    """
    Make all lower case to have perfect match with stopword list
    """
    
    lower=[]
    def lowercase(lists): 
        lower=[x.lower() for x in lists]
        return lower
    
    wsplit =lowercase(rwhite)
   
    """
    Splitting happening here
    """
    def splitwords(lists): 
        split=([i for item in lists for i in item.split()]) 
        return split

    wsplit=(splitwords(wsplit))
    
    """
    Dealing with tokenization cases by adding whitespaces: 
    1. wasn't -> wasn ' t
    2. "pop embleshing" -> " pop emblishing "  
    """
    
    for i in range(len(wsplit)):
        wsplit[i]=wsplit[i].replace("'"," ' ")
        wsplit[i]=wsplit[i].replace("."," . ")
        wsplit[i]=wsplit[i].replace("-"," - ")
        wsplit[i]=wsplit[i].replace(","," . ")
        wsplit[i]=wsplit[i].replace('"',' " ')

    wsplit=(splitwords(wsplit))
    wsplit
    
    """
    Sample result of tokenization

    """
    for j in range(0,100):
        wsplit[j]
        
    print("Tokenization completed.")
    
    """   
    2. Removing Special Characters
    Note: Converted list to dataframe for splitting later using sklearn
    """
    
    df = pd.DataFrame({'col':wsplit})
    print(df.shape)
    # sample_tokenized_list = [["Hello", "World", "."], ["Good", "bye"]]

    """
    Removing special characters (Continued)
    """
    sp_df=df['col'].str.replace('[^\w\s]','')

    print("Special Characters removed.")
    
    """
    3. Stopwords Preprocessing
    """
    
    """
    3.1 With Stopwords Data Set
    """
    sp_df=pd.DataFrame(sp_df)
    
    """
    3.2 Without Stopwords Data Set
    """
    
    stop = stopwords.words('english')
    stopword=(sp_df['col']).apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    """
    Dropping empty string rows for both neg and pos 
    """
    stopword=pd.DataFrame(stopword)
    stopword['col'].replace('', np.nan, inplace=True)
    stopword.dropna(subset=['col'], inplace=True)
    
    """
    Verifying results without stopwords for neg and pos
    """
    print(stopword.head())
    
    print("Data with stopwords and without stopwords created.")
    
    """
    4. Data Splitting (Training(80%), Validation(10%) and Test (10%) set)
    """
    """
    A. With Stopwords
    """
    
    from sklearn.model_selection import train_test_split
    train_list, test_list = train_test_split(sp_df, test_size=0.1, random_state=1)
    train_list, val_list = train_test_split(train_list, test_size=0.111111, random_state=1)
    
    """
    B. Without stopwords
    """
    train_list_no_stopword, test_list_no_stopword = train_test_split(stopword, test_size=0.1, random_state=1)
    train_list_no_stopword, val_list_no_stopword = train_test_split(train_list_no_stopword, test_size=0.111111, random_state=1)
    
    print("Train/test/validation sets made.")
    
    print("Saving files.")
    
    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopword,delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,delimiter=",", fmt='%s')
    
    
    print("Task Complete.")