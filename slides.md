% Forest Cover Type Classification using SciKitLearn
% Cara, Jacob, Lorenzo
% December 13, 2018

# Deciding on a project

Formed a group and deciding to do something with machine learning

Explored Kaggle for potential projects

Came across a couple options including forest cover classification and classifying monsters (Halloween themed)

Decided on classification on forest cover type

# Task

Predict forest cover type

- Classify the predominant kind of tree cover

- Actual forest cover type for a 30x30 meter cell was determined by US Forest Service

- Independent variables obtained from US Geological Survey and USFS

# Importance

Real world data

Designing ways to predict forest cover type can help for future forest surveys

Forests are an important natural resource

- Play an important role in sustaining geochemical and bioclimatic processes

Knowing the most important variables in predicting cover type can lead to more efficient surveying

# Data

56 total variables

- ID
- Slope
- Elevation
- Aspect
- Slope
- Horizontal distance to hydrology
- Vertical distance to hydrology
- Hillshade at different times of day
- 4 different wilderness areas
- 40 different soil types

15120 rows of data

# Elevation differences

![Google Images](https://pixnio.com/free-images/2016/06/14/forest-hillside-725x483.jpg)

# Aspect and Hillshade

![Google Images](https://media.mnn.com/assets/images/2015/08/pine-creek-gorge-pa.jpg.838x0_q80.jpg)

# Exploratory Analysis

```python
# Imported pandas and investigated the data 
import pandas as pd

df = pd.read_csv('train.csv')

df.head()
df.info()
df.describe()
```
Learned basics about the data

# Exploratory Analysis

```python
import seaborn as sns

df1 = df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
          'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
          'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
          'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]

sns.pairplot(df1)

```

# Some correlations and violin plots



![Violin Plot 1](C:\Users\Jacob\PycharmProjects\group-project-jcl\src\visualization\Figure_1.png)



# Machine Learning with SciKitLearn

Decided to try to use SciKitLearn

Wanted to try multiple different classification models as this is a supervised learning project

Logistic regression, random forest, support vector machine, k nearest neighbor

# Importing Necessary Packages
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
```

# Testing Multiple Models
```python
df = pd.read_csv('train.csv')
x = df.copy()
x = x.drop('Cover_Type', axis = 1)
x = x.drop('Id', axis = 1)

y = df['Cover_Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)


# specify models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=1000, max_depth=30, class_weight='balanced')
sv = svm.SVC(kernel='rbf')

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

lr_accuracy 
rf_accuracy 
sv_accuracy 
```

# Results

Logistic Regression Accuracy Score: 

- 0.6761463844797179

Random Forest Accuracy Score:

- 0.859347442680776

Support Vector Machine Accuracy Score:

- 0.13734567901234568

# Testing a Model with Select Variables

Random Forest!

Changed variables in the model:

```python
df = pd.read_csv('train.csv')

x = df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]
y = df['Cover_Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=24)
rfc = RandomForestClassifier(n_estimators=1000, max_depth=100, class_weight='balanced')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
rf_accuracy = accuracy_score(y_test, y_pred)

rf_accuracy
```

New Accuracy Score with only 8 variables:

- 0.7908399470899471

Not as good of a model!

# Next Steps

Still have to troubleshoot and hone our models

Try different variables

Try combinations of variables

Generate confusion matrix

Need to use the test set on Kaggle and see how well our model preforms!

# Questions?

Thank you!