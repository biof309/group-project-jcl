% Forrest Cover Type Classification using SciKitLearn
% Cara, Jacob, Lorenzo
% December 11, 2018

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

![](aspect_plot.png)





# Adding an image

![Tree Pic](http://www.mast-producing-trees.org/wp-content/uploads/2009/11/oak-hickory.jpg)


# Here is a slide with some code on it!
```python
# get a list of all of the letters in the alphabet 
[chr(65+i) for i in range(26)] #uppercase      
[chr(97+i) for i in range(26)] #lowercase
```