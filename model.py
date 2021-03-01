import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#covt words to int;

def covert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0 }
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : covert_to_int(x))

y = dataset.iloc[:, -1]

##spliting training  and test set;

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with traing data
regressor.fit(X, y)

#saving model to disk

pickle.dump(regressor, open('model.pkl', 'wb'))

#loading model to compare the result
model = pickle.load(open('model.pkl', 'rb'))
