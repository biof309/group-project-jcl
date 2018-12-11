import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.csv')
df.head()
df.info()

df1 = df[['Elevation', 'Aspect', 'Slope']]
df1.head()

sns.pairplot(df1)
plt.show()

x = df.copy()
x = x.drop('Cover_Type', axis = 1)
x = x.drop('Id', axis = 1)

y = df[['Cover_Type']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
