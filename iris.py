import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.neighbors import KNeighborsClassifier

dataframe = pd.read_csv('IRIS.csv')

def isContain(word,spc):
    for i in range(len(spc)):
        if word==spc[i]:
            return i
    return -1       


feautres = dataframe.iloc[:,[0,1,2,3]].values
species = list(dataframe['species'])
# print(species)
spc=[]
for i in range(len(species)):
    check = isContain(species[i],spc)
    if check==-1:
        text = species[i]
        species[i]=len(spc)
        spc.append(text)
    else:
        species[i]=check

# print(species)
# feautres

import random
index = [i for i in range(len(species))]
index = random.sample(index,len(index))
train_size = (len(index)*75)//100
train_idx = index[0:train_size]
test_idx = index[:-(len(index)*25)//100:-1]


X_train=[feautres[i] for i in train_idx]
Y_train=[species[i] for i in train_idx]

X_test=[feautres[i] for i in test_idx]
Y_test=[species[i] for i in test_idx]


model = KNeighborsClassifier()

model.fit(X_train,Y_train)
print('accuracy ',model.score(X_test,Y_test))

lebal = model.predict([[6,3,4.8,1.8]])
# print(lebal)
print(spc[lebal[0]])