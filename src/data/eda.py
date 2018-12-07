import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df.head()
df.info()

df1 = df[['Elevation', 'Aspect', 'Slope']]
df1.head()

sns.pairplot(df1)
plt.show()
