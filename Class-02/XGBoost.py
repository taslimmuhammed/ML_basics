import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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

xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = 55)
xgb_model.fit(X_train,y_train, eval_set = [(X_val,y_val)], early_stopping_rounds = 10)
#  early stopping rounds is the count of rounds done after the model is saturated
# if model is ready at 5 rounds then 5+ early_stopping_rounds no of rounds will be performed
prediction = xgb_model.predict(X_val)
accuracy = accuracy_score(prediction,y_val)
print(accuracy)

