
# coding: utf-8

# In[2]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[3]:

df = pd.read_csv('drugsComTrain_raw.csv')


# In[4]:

df.info()


# In[5]:

df.head()


# In[6]:

df.describe()


# In[7]:

#find a correlation among the rating and useful count
df.plot.scatter(x='rating',y='usefulCount')


# In[8]:

#get the top 15 reviews in the list based on usefulness count
df.nlargest(15,'usefulCount')


# In[9]:

#see the most used drugs to treat conditions
df['drugName'].value_counts()


# In[10]:

#find the most common conditions that are treated
df['condition'].value_counts().nlargest(15)


# In[10]:

# checking out the test data
# commonCdf2 = df2['condition'].value_counts().nlargest(15)
# commonConditions = pd.DataFrame(commonCdf2)
# commonConditions


# In[11]:

#establish a dataframe to focus on birth control
birth_control_df = df[df.condition == 'Birth Control']


# In[12]:

birth_control_df.info()


# In[13]:

#locate each instance of birth control use in our dataframe
birth_control_df.head()


# In[14]:

#get the number of times each drug was used for birth control
birth_control_df['drugName'].value_counts().nlargest(15)


# In[15]:

#dataframe to show the average satifcation of using Etonogestrel for birth control(top choice of drug)
etonogestrelBCDf = birth_control_df[birth_control_df.drugName == 'Etonogestrel']
etonogestrelBCDf['rating'].mean()


# In[16]:

#find the top occurences of the drugs
df['drugName'].value_counts().nlargest(15)


# In[17]:

def condition_func(condition, df):
    if(type(condition) != str):
        raise ValueError("The input should be a string")
        
    condition_df = df.loc[df['condition'] == condition]
    print(condition_df['drugName'].value_counts().nlargest(15))
    
condition_func('Birth Control', df)


# In[18]:

#Looop through the drug names
for drug_name in df.drugName.unique():
    print(drug_name)


# In[19]:

pd.DatetimeIndex(start=min(df.date), end=max(df.date), freq='M')


# In[20]:

df.index = pd.to_datetime(df.date)
drug_df = pd.DataFrame(index=pd.DatetimeIndex(start=min(df.date), end=max(df.date), freq='M'))

for d_name in df.drugName.value_counts().nlargest(5).index:
    temp = df[df.drugName == d_name]
    drug_df[d_name] = temp.rating.resample('M').mean()

drug_df


# In[21]:

#Make the line graph of them
for c in drug_df.columns:
    drug_df[c].plot(title=c, ylim=(0, 10))
    plt.show()


# In[22]:

drug_df.plot(ylim=(0,10))
plt.show()


# In[23]:

#Make a graph of the mean rating of each year
df.rating.resample('A').mean().plot(ylim=(0,10))
plt.show()


# In[24]:

df[df.condition == "Birth Control"].review.nunique()


# In[25]:

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string


# In[53]:

def get_df(df, cond):
    review = df[df.condition == cond].review[:500]
    return pd.DataFrame({'text':review, 'label':cond})


# In[55]:

# we only take 500 reviews for each condition so the traning time will decrease

train_df = pd.DataFrame(columns=['text', 'label'])

condition_counts = df.condition.value_counts()

for cond in list(df.condition.unique()):
    if cond in condition_counts and condition_counts[cond] > 100:
        train_df = train_df.append(get_df(df, cond))

train_df.info()


# In[62]:

train_df['text'] = train_df.text.astype(str)
train_df['label'] = train_df.label.astype(str)


# In[63]:

train_df.label.value_counts()


# In[64]:

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], train_df['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# In[65]:

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_df['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[66]:

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train_df['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train_df['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train_df['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 


# In[67]:

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[ ]:

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)


# In[ ]:



