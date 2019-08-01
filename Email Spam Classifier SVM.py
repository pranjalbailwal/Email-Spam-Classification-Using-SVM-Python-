#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


# In[36]:


ham = [os.path.join("enron1/ham/", f) for f in os.listdir("enron1/ham/")]
spam = [os.path.join("enron1/spam/", f) for f in os.listdir("enron1/spam/")]

all_words = []
for email in ham:
    with open(email) as m:
        for line in m:
            words = line.split()
            all_words += words
for email in spam:
    with open(email, errors='ignore') as m:
        for line in m:
            words = line.split()
            all_words += words
dictionary = Counter(all_words)


# In[37]:


dict_list = list(dictionary)
for word in dict_list:
    if(word.isalpha()==False):
        del dictionary[word]
    elif(len(word)==1):
        del dictionary[word]
dictionary = dictionary.most_common(5000)


# In[46]:


def extractFeatures(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    feature_matrix = np.zeros((len(files), 5000))
    train_labels = np.zeros(len(files))
    docID = 0
    for file in files:
        with open(file, errors='ignore') as m:
            for line in m:
                words = line.split()
                for word in words:
                    wordID = 0
                    for i,d in enumerate(dictionary):
                        if(d[0] == word):
                            wordID = i
                            feature_matrix[docID, wordID] = words.count(word)
        tags = file.split('/')
        train_labels[docID] = 0
        label = tags[len(tags)-2]
        if(label == "spam"):
            train_labels[docID] = 1
        docID += 1
    return feature_matrix, train_labels
            


# In[47]:


ham_dir = "enron1/ham/"
spam_dir = "enron1/spam/"
ham_fm, ham_targets = extractFeatures(ham_dir)
spam_fm, spam_targets = extractFeatures(spam_dir)


# In[65]:


X = np.concatenate([ham_fm, spam_fm])
Y = np.concatenate([ham_targets, spam_targets])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.33, random_state = 42)


# In[74]:


parameters = [{'C':[1,10,100,1000], 'kernel': ['linear']}, {'C':[1,10,100,1000], 'gamma':[0.001, 0.01, 0.1, 1], 'kernel':['rbf']},]
clf = GridSearchCV(estimator=svm.SVC(), param_grid = parameters, cv = 3, n_jobs=-1)
clf.fit(X_train,Y_train)


# In[75]:


print("Best Score:", clf.best_score_)
print("Best C:", clf.best_estimator_.C)
print("Best Kernel:", clf.best_estimator_.kernel)
print("Best Gamma:", clf.best_estimator_.gamma)


# In[77]:


model = svm.SVC(kernel='rbf', C=10, gamma=0.001)
model.fit(X_train, Y_train)
predicted_targets = model.predict(X_test)
print("Accuracy :", accuracy_score(Y_test, predicted_targets)*100)

