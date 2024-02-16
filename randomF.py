import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('D:/project\pythonProject/Regression/randomForst/diabetes3.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y , random_state=0, train_size = .75)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred = rf.predict(x_test[5:10])
print(x_test[5:10])
print(pred)