import pandas as pd
import sklearn
df = pd.read_csv('train.csv')
import numpy as np
import matplotlib as plt


df.head()
df.info()
df.describe()
'''
cover type is what were trying to predict
we can use other variables that we choose to predict cover type
'''


from sklearn.model_selection import train_test_split
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['Cover_Type'].values
X = df.drop('Cover_Type', axis=1).values


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

# This was interesting to see KNN for cover type and see how the accurate the model was

# Create arrays for features and target variable
X = df['Elevation']

y = df['Cover_Type']

print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

y = np.reshape(y, [-1, 1])

X = np.reshape(X, -1, 1)

print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

''' 
I cannot figure out how to reshape this and I am not sure if it is possible because it is (15120, ) and I think
needs to be (15120, 1) idk what to do yet. COME BACK TO THIS
'''

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)

# Fit the model to the data
reg.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X, y))

# Plot regression line
pd.plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

'''
the below script is what we need to do to set hyperparameters and compute metrics of a logistic regression for our data
the problem is that I dont know how to reshape the data properly so the tests fail at the end
'''


y = df[['Cover_Type']]

X = df[['Elevation']]

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=21)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


#building a rnadom forest:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
df = pd.read_csv('train.csv')

df.info()
df.copy()
df.describe()
x=df[['Elevation', 'Id', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]
y=df['Cover_Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=24)
rfc = RandomForestClassifier(n_estimators=1000, max_depth=100, class_weight='balanced')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
rf_accuracy = accuracy_score(y_test, y_pred)

rf_accuracy

import numpy as np

ct = df[['Cover_Type']]
elevation = df[['Elevation']]
corr_1 = np.corrcoef(ct, elevation)
corr_1
