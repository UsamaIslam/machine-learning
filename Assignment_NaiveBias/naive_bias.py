import pandas as pd
from sklearn.model_selection import train_test_split

csv = 'Sentiment_Analysis_task.csv'
column_names = ['labels', 'sentence']
data = pd.read_csv(csv, header=None, names=column_names)

l_data = data.labels
t_data = data.sentence

train_data, test_data, train_label, test_label = train_test_split(t_data, l_data, test_size=0.2)

'''
print(len(train_data))
print(len(test_data))
print(len(test_lable))
print(len(train_lable))
'''
