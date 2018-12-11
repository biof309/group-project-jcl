import pandas as pd
import seaborn as sns

df = pd.read_csv('train.csv')
#df.head()
#df.info()

df1 = df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
          'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
          'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
          'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]
df1.head()

sns.pairplot(df1)

sns.violinplot(data = df, x = 'Cover_Type', y = 'Aspect')