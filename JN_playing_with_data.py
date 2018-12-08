import pandas as pd
import sklearn
df = pd.read_csv('train.csv')
import numpy as np

df.head()
df.info()

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



# Create arrays for features and target variable
X = df['Elevation']

y = df['Cover_Type']

print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

y = np.reshape(y, [-1, 1])

X = np.reshape(X, -1, 1)

print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))
# I cannot figure out how to reshape this and I am not sure if it is possible because it is (15120, ) and I think
# needs to be (15120, 1) idk what to do yet. COME BACK TO THIS

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
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


