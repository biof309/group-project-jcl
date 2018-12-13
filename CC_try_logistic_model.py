import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

df = pd.read_csv('train.csv')
df.head()
df.info()

x = df.copy()
x = x.drop('Cover_Type', axis = 1)
x = x.drop('Id', axis = 1)

y = df['Cover_Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42,
                                                    stratify=True)
# specify models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=1000, max_depth=30, class_weight='balanced')
sv = svm.SVC()

# fit models
lr.fit(x_train, y_train)
rf.fit(x_train, y_train)
sv.fit(x_train, y_train)

# predict test set labels
lr_pred = lr.predict(x_test)
rf_pred = rf.predict(x_test)
sv_pred = sv.predict(x_test)

# evaluate model accuracies
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
sv_accuracy = accuracy_score(y_test, sv_pred)


