import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]
# print("after one-hot encoding")
df = pd.get_dummies(data=df, prefix=cat_variables,columns=cat_variables)
# print(df.head())
features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing heart diseas  column from features list
# print(df.columns)
# print(features)
X_train, X_val, Y_train, Y_val = train_test_split(df[features], df['HeartDisease'],test_size=.8, train_size=.2, random_state=55,)

model  = DecisionTreeClassifier(max_depth=3,min_samples_split=30, random_state=55)
model.fit(X_train,Y_train)
prediction_val = model.predict(X_val) #val for cross validator set
accuracy_val = accuracy_score(prediction_val,Y_val)
print(accuracy_val)
