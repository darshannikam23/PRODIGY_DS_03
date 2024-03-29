import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

bank = pd.read_csv("Churn_Modelling.csv")
#print(bank.info())
#print(bank)

bank = bank.drop(['RowNumber', 'Surname', 'Geography', 'Gender'], axis=1)

threshold_balance = bank["Balance"].mean()
threshold_active = bank["IsActiveMember"].mean()

def create_purchase_label(row):
	if row['IsActiveMember'] > threshold_active and row['Balance'] > threshold_balance:
		return 1
	else:
		return 0

bank['PurchaseLabel'] = bank.apply(create_purchase_label, axis=1)
print(bank[['Balance', 'IsActiveMember', 'PurchaseLabel']])
print((bank['PurchaseLabel']).sum())

y = bank['PurchaseLabel']
x = bank.drop(['PurchaseLabel'], axis=1)
print(x)
print(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtc.fit(X_train, y_train)
p=dtc.predict(X_test)
prediction=pd.DataFrame(p)
score=metrics.accuracy_score(prediction,y_test)
print("Score using Decision Tree", score)

print("Classification Report:\n", classification_report(y_test, p))
