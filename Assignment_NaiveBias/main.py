import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from NaiveByes import NaiveBayes

csv = 'Sentiment_Analysis_task.csv'
column_names = ['labels', 'sentence']
data = pd.read_csv(csv, header=None, names=column_names)

l_data = data.labels
t_data = data.sentence

train_data, test_data, train_labels, test_labels = train_test_split(t_data, l_data, test_size=0.2)

nb=NaiveBayes(np.unique(train_labels)) #instantiate a NB class object
print ("---------------- Training In Progress --------------------")

nb.train(train_data,train_labels) #start tarining by calling the train function
print ('----------------- Training Completed ---------------------')

pclasses=nb.test(test_data) #get predcitions for test set

#check how many predcitions actually match original test labels
test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0])

print ("Test Set Examples: ",test_labels.shape[0]) # Outputs : Test Set Examples:  1502
print ("Test Set Accuracy: ",test_acc*100,"%") # Outputs : Test Set Accuracy:  93.8748335553 %

'''
print(len(train_data))
print(len(test_data))
print(len(test_lable))
print(len(train_lable))
'''
