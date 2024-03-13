import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]
df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)
features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable

X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=55)

model = RandomForestClassifier(n_estimators=100,min_samples_split=10, max_depth=16, random_state=55)
# n_estimator is the  number of trees in the forest and it’s  a hyperparameter that we can tune to improve
model.fit(X_train,y_train)
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print('Validation Accuracy: ', accuracy)