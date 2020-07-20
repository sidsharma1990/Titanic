# Titanic Main

#================ Final=====================
X_train = X, X_test = test_ds, y_train = y, y_test = gen_sub

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Kaggle
# dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
# test_ds = pd.read_csv('/kaggle/input/titanic/test.csv')
# gen_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# Importing the dataset
dataset = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')
gen_sub = pd.read_csv('gender_submission.csv')

# Dropping categorical data
dataset.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',
        inplace=True)
test_ds.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',
        inplace=True)
gen_sub.drop('PassengerId', axis=1, inplace=True)

# Value count
dataset.Sex.value_counts()
test_ds.Sex.value_counts()

# Survived by sex
dataset[["Sex", "Survived"]].groupby(dataset["Sex"], as_index = False).mean()

# dataset dep and indepen
X = dataset.drop('Survived', axis='columns')
y = dataset.Survived

# to check, in which column do we have missing value
X.columns[X.isna().any()]
test_ds.columns[test_ds.isna().any()]
X.isna().sum()
test_ds.isna().sum()

# filling NaN values
X.Age = X.Age.fillna(X.Age.median())
test_ds = test_ds.fillna({'Age':test_ds.Age.median(),
                          'Fare':test_ds.Fare.median()})

X.describe()

# creating dummy variables for categorical data
dummies = pd.get_dummies(X.Sex)
dummies1 = pd.get_dummies(test_ds.Sex)

X = pd.concat([X,dummies],axis='columns')
test_ds = pd.concat([test_ds,dummies1],axis='columns')

# droping sex column
X.drop(['Sex','female'],axis='columns',inplace=True)
test_ds.drop(['Sex','female'],axis='columns',inplace=True)

### Correlation=======================================================
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#===============================================================

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test_ds = sc.transform(test_ds)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
obj = RandomForestClassifier()
obj.fit(X,y)

# Prediction
y_pred = obj.predict(test_ds)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(confusion_matrix(gen_sub,y_pred))
print(classification_report(gen_sub,y_pred))
print (accuracy_score(gen_sub,y_pred))
print ("{0:.0%}".format(ac))
0.8492822966507177 (85%)
0.8253588516746412
print ('{.2'}.accuracy_score(gen_sub,y_pred))
ac = accuracy_score(gen_sub,y_pred)
###########
# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)

y_pred = classifier.predict(test_ds)

# Making the Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
cm = confusion_matrix(gen_sub, y_pred)
print(cm)
print (accuracy_score(gen_sub,y_pred))
ac = accuracy_score(gen_sub,y_pred)
print ("{0:.0%}".format(ac))
0.8995215311004785

############ ANN
 Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

X_train = X, X_test = test_ds, y_train = y, y_test = gen_sub

# Importing the dataset
dataset = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv')
gen_sub = pd.read_csv('gender_submission.csv')

# Dropping categorical data
dataset.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',
        inplace=True)
test_ds.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',
        inplace=True)
gen_sub.drop('PassengerId', axis=1, inplace=True)

# Value count
dataset.Sex.value_counts()
test_ds.Sex.value_counts()

# Survived bu sex
dataset[["Sex", "Survived"]].groupby(dataset["Sex"], as_index = False).mean()

# dataset dep and indepen
X = dataset.drop('Survived', axis='columns')
y = dataset.Survived

# to check, in which column do we have missing value
X.columns[X.isna().any()]
test_ds.columns[test_ds.isna().any()]
X.isna().sum()
test_ds.isna().sum()

# filling NaN values
X.Age = X.Age.fillna(X.Age.median())
test_ds = test_ds.fillna({'Age':test_ds.Age.median(),
                          'Fare':test_ds.Fare.median()})

X.describe()

# creating dummy variables for categorical data
dummies = pd.get_dummies(X.Sex)
dummies1 = pd.get_dummies(test_ds.Sex)

X = pd.concat([X,dummies],axis='columns')
test_ds = pd.concat([test_ds,dummies1],axis='columns')

# droping sex column
X.drop(['Sex','female'],axis='columns',inplace=True)
test_ds.drop(['Sex','female'],axis='columns',inplace=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test_ds = sc.transform(test_ds)
########==================================

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# relu is rectifier activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer, Sigmoid is the best probability function with 0 and 1 output
# Use softmax, if ouput is not binary, and have more than 2 outputs
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN=====================================================

# Compiling the ANN
# categorical_crossentropy for categorical output
# metrics can be multiple as a list
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X, y, batch_size = 32, epochs = 100)

# testing data on 1 member
ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))> 0.5)

# Part 4 - Making the predictions and evaluating the model==================
# Predicting the Test set results
y_pred = ann.predict(test_ds)
y_pred1 = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(gen_sub, y_pred1)
print(cm)
accuracy_score(gen_sub, y_pred1)
































