
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
warnings.filterwarnings('ignore')
df=pd.read_csv('https://raw.githubusercontent.com/iam-Vivek/forest_fire_prediction/main/an.csv')
features = df[['temperature', 'humidity','sunlight' , 'oxygen']]
target = df['label']
labels = df['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
acc = []
model = []
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)
print(classification_report(Ytest,predicted_values))
filename = 'm1.pkl'
joblib.dump(RF, filename)