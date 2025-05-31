import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix,classification_report,average_precision_score,precision_recall_curve

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    df=pd.read_csv("dataset.csv")
    print("Loaded successfully")
except:
    print("error loading the dataset>>>>>")
    exit()

print("fraud and normal percentage....")
print(df['Class'].value_counts(normalize=True)*100)

# sns.countplot(x='Class', data=df)
# plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
# plt.show()
# print(df.head())
df=df.drop('Time',axis=1)
# print(df.head())
x=df.drop('Class',axis=1)
y=df['Class']

scaler=StandardScaler()
x['Amount']=scaler.fit_transform(x[['Amount']])
# print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
# print(x_test.shape)

smote=SMOTE(random_state=42)
x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)
# print(x_train.head())

isolation_forest=IsolationForest(random_state=42,contamination=y_train.sum()/len(y_train))
model=isolation_forest.fit(x_train)
# y_pred_raw=model.predict(x_test)
# y_pred=np.where(y_pred_raw==-1,1,0)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# lr=LogisticRegression()
# model=lr.fit(x_train_smote,y_train_smote)
# y_pred=model.predict(x_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

random_forest=RandomForestClassifier(n_estimators=10)
model=random_forest.fit(x_train_smote,y_train_smote)
y_pred=model.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))