---
date: 2020-3-01
title: Practice Classification
---


## Real or Not? NLP with Disaster Tweets

```
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import re
import string
import os

from string import punctuation
from collections import defaultdict
from nltk import FreqDist
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer as countVectorizer
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
```


```
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
submission = pd.read_csv("./sample_submission.csv")

print(train_df.head(10))

data = pd.concat([train_df, test_df], axis=0, sort=False)

print("Number of unique locations: ", data.location.nunique())


print("Missing values:")
data.isna().sum()
data.location.fillna("None", inplace=True)
data['text'] = data['text'].apply(lambda x: remove_URL(x))
data['text'] = data['text'].apply(lambda x: remove_punct(x))


```

       id keyword  ...                                               text target
    0   1     NaN  ...  Our Deeds are the Reason of this #earthquake M...      1
    1   4     NaN  ...             Forest fire near La Ronge Sask. Canada      1
    2   5     NaN  ...  All residents asked to 'shelter in place' are ...      1
    3   6     NaN  ...  13,000 people receive #wildfires evacuation or...      1
    4   7     NaN  ...  Just got sent this photo from Ruby #Alaska as ...      1
    5   8     NaN  ...  #RockyFire Update => California Hwy. 20 closed...      1
    6  10     NaN  ...  #flood #disaster Heavy rain causes flash flood...      1
    7  13     NaN  ...  I'm on top of the hill and I can see a fire in...      1
    8  14     NaN  ...  There's an emergency evacuation happening now ...      1
    9  15     NaN  ...  I'm afraid that the tornado is coming to our a...      1
    
    [10 rows x 5 columns]
    Number of unique locations:  4521
    Missing values:



```
vectorizer = feature_extraction.text.CountVectorizer()

train_v = vectorizer.fit_transform(train_df["text"])
test_v = vectorizer.transform(test_df["text"])

linear_classifier = linear_model.RidgeClassifier()
score = model_selection.cross_val_score(linear_classifier, train_v, train_df["target"], cv=3, scoring="f1")
print(score)
linear_classifier.fit(train_v, train_df["target"])
submission["target"] = linear_classifier.predict(test_v)
print(submission.head())
submission.to_csv('submission.csv', index=False)
```

    [0.59453669 0.56498283 0.64082434]
       id  target
    0   0       0
    1   2       1
    2   3       1
    3   9       0
    4  11       1

