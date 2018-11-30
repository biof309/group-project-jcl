url = 'https://www.kaggle.com/c/forest-cover-type-kernels-only/download/train.csv'

from urllib.request import urlretrieve

urlretrieve(url, 'train.csv')
