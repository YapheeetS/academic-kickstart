---
date: 2020-4-20
title: Assignment3
---

# Haojin Liao 1001778275

## Data preprocess


```python
import re
import os
import numpy as np
from tqdm import tqdm
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')


class Data_preprocess(object):
    def __init__(self):
        pass

    # regular expression
    def rm_tags(self, text):
        re_tag = re.compile(r'<[^>]+>')
        return re_tag.sub('', text)


    def read_files(self, filetype):
        path = "./aclImdb/"
        file_list = []
        pos_num = 0
        neg_num = 0
        positive_path = path + filetype+"/pos/"
        for f in os.listdir(positive_path):
            file_list += [positive_path+f]
            pos_num += 1
        negative_path = path + filetype+"/neg/"
        for f in os.listdir(negative_path):
            file_list += [negative_path+f]
            neg_num += 1
        print('read', filetype, 'files:', len(file_list))
        print('pos_num: ', pos_num)
        print('neg_num: ', neg_num)
        all_labels = ([1] * pos_num + [0] * neg_num)
        all_texts = []
        for index, fi in tqdm(enumerate(file_list)):
            with open(fi, encoding='utf8') as file_input:
                filelines = file_input.readlines()
                if len(filelines) != 0:
                    text = filelines[0]
                    # remove < > tag
                    text = self.rm_tags(text)
                    # lower case
                    text = text.lower()
                    # tokenize
                    words = word_tokenize(text)
                    # topwords
                    words = [w for w in words if w not in stopwords.words('english')]
                    # # Stemming
                    words = [PorterStemmer().stem(w) for w in words]
                    all_texts.append(words)
                else:
                    print('empty index: ', index)
                    all_texts.append([''])
#             if index == 20:
#                 break

        return all_texts, all_labels


data_preprocess = Data_preprocess()
x_train, y_train = data_preprocess.read_files('train')
x_test, y_test = data_preprocess.read_files('test')

train_index = [i for i in range(len(x_train))]
test_index = [i for i in range(len(x_test))]

random.shuffle(train_index)
random.shuffle(test_index)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train[train_index]
y_train = y_train[train_index]
x_test = x_test[test_index]
y_test = y_test[test_index]


```

    [nltk_data] Downloading package punkt to /home/haojin/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /home/haojin/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    6it [00:00, 59.93it/s]

    read train files: 25000
    pos_num:  12500
    neg_num:  12500


    25000it [11:37, 35.82it/s]
    5it [00:00, 42.39it/s]

    read test files: 25000
    pos_num:  12500
    neg_num:  12500


    15841it [07:10, 36.76it/s]

    empty index:  15835


    18847it [08:33, 50.65it/s]

    empty index:  18839


    25000it [11:21, 36.70it/s]


## Extract feature words


```python
import math, collections

labels = [0, 1]

def mutual_info(N, Nij, Ni_, N_j):
    return Nij * 1.0 / N * math.log(N * (Nij + 1) * 1.0 / (Ni_ * N_j)) / math.log(2)

def label2id(label):
    for i in range(len(labels)):
        if label == labels[i]:
            return i

def id2label(i):
    if i <= 2:
        return labels[i]
    else:
        return 0
    
def doc_dict():
    return [0] * len(labels)

def count_for_cates(train_x, train_y, featureFile='./bayes_feature.txt'):
    doccount = [0] * len(labels)
    wordcount = collections.defaultdict(lambda: doc_dict())

    n = 0
    class_count = [0, 0]

    while (n < len(train_x)):

        index1 = label2id(train_y[n])

        class_count[index1] += 1
        words = train_x[n]
        for word in words:
            wordcount[word][index1] += 1
            doccount[index1] += 1
        n += 1

    # print('wordcount:', wordcount)
    print('Word count ：', len(wordcount))
    print('doc count Number of words per category:', doccount)

    print('Extract feature words')
    midict = collections.defaultdict(lambda: doc_dict())
    N = sum(doccount)
    for k, vs in wordcount.items():
        for i in range(len(vs)):
            N11 = vs[i]
            N10 = sum(vs) - N11
            N01 = doccount[i] - N11
            N00 = N - N11 - N10 - N01
            mi = mutual_info(N, N11, N10 + N11, N01 + N11) + mutual_info(N, N10, N10 + N11, N00 + N10) + mutual_info(N, N01, N01 + N11, N01 + N00) + mutual_info(
                N, N00, N00 + N10, N00 + N01)
            midict[k][i] = mi

    fwords = set()
    for i in range(len(doccount)):
        keyf = lambda x: x[1][i]
        sortedDict = sorted(midict.items(), key=keyf, reverse=True)
        for j in range(100):
            fwords.add(sortedDict[j][0])
    out = open(featureFile, 'w', encoding='utf-8', errors='ignore')
    out.write(str(doccount) + '\n')
    for fword in fwords:
        out.write(fword + '\n')
    out.close()
    return class_count

class_count = count_for_cates(x_train, y_train)


```

    Word count ： 108646
    doc count Number of words per category: [1899087, 1953045]
    Extract feature words


## Train Naive_Bayes


```python
def load_feature_words(featureFile):
    f = open(featureFile, encoding='utf-8', errors='ignore')
    doccounts = eval(f.readline())
    features = set()
    for line in f:
        features.add(line.strip())
    f.close()
    return doccounts, features

def train_bayes(class_count, featurefile, modelfile, x_train, y_train):
    
    doccounts, features = load_feature_words(featurefile)
    print(doccounts)
    wordcount = collections.defaultdict(lambda: doc_dict())
    tcount = [0] * len(doccounts)
    
    for index, words in enumerate(x_train):
        
        index1 = label2id(y_train[index])
        
        for word in words:
            if word in features:
                tcount[index1] += 1
                wordcount[word][index1] += 1
    print('tcount: ', tcount)
    print('wordcount: ', wordcount)
    outmodel = open(modelfile, 'w', encoding='utf-8')
    print('save model')
    for k, v in wordcount.items():
        if k == '':
            continue
        scores = [v[i] * 1.0 / len(wordcount) * (class_count[i]/sum(class_count)) for i in range(len(v))]
#         scores = [(v[i] + 1) * 1.0 / (tcount[i] + len(wordcount)) * (class_count[i]/sum(class_count)) for i in range(len(v))]
        outmodel.write(k + '\t' + str(scores) + '\n')
    outmodel.close()
    
train_bayes(class_count,'./bayes_feature.txt','./bayes_model.txt', x_train, y_train)
```

    [1899087, 1953045]
    tcount:  [362722, 330106]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f18c4186560>, {'movi': [27993, 21947], ',': [131804, 144074], "n't": [19959, 13420], 'enjoy': [1466, 2797], 'love': [2751, 5962], 'year': [2551, 3765], 'superb': [99, 549], 'could': [5684, 3673], 'play': [3565, 5009], 'perfect': [367, 1338], 'great': [2637, 6351], 'well': [3833, 5833], 'noth': [2927, 1283], 'look': [5646, 4032], 'world': [1369, 2395], 'beauti': [820, 2393], 'oh': [934, 324], 'famili': [1218, 2116], 'life': [2214, 3885], 'annoy': [943, 282], 'also': [3496, 5433], 'tri': [3808, 2491], 'act': [5120, 3318], 'favorit': [297, 1101], 'reason': [1964, 1165], 'dull': [664, 141], 'anyth': [1843, 1059], 'would': [7663, 5710], '2': [1295, 667], 'today': [313, 922], '...': [7242, 4884], '?': [11342, 4743], 'bad': [7139, 1846], 'even': [7665, 5021], 'wast': [1998, 193], 'plot': [4137, 2469], 'wonder': [1309, 2282], "'m": [2878, 1878], 'amaz': [367, 1162], 'suck': [591, 124], 'least': [1989, 1092], 'poor': [1443, 407], 'best': [2039, 4226], 'suppos': [1451, 455], 'terribl': [1544, 321], 'brilliant': [263, 885], 'like': [12020, 10138], 'guy': [2658, 1559], 'instead': [1433, 698], 'worst': [2433, 244], 'dumb': [491, 131], 'thing': [4619, 3435], 'decent': [813, 335], 'excel': [419, 1781], 'disappoint': [1200, 591], 'alway': [1138, 2067], 'pathet': [403, 63], 'minut': [2458, 1049], 'bore': [1941, 512], 'save': [1164, 521], 'perform': [1842, 3559], 'ridicul': [894, 219], 'avoid': [685, 228], 'unless': [507, 135], 'cheap': [667, 202], 'poorli': [618, 67], 'highli': [267, 828], 'seri': [1186, 2045], 'role': [1578, 2569], 'crap': [820, 143], 'wors': [1189, 209], 'unfunni': [244, 17], 'script': [2196, 1037], 'lack': [1214, 590], 'horribl': [1190, 195], 'unbeliev': [428, 113], 'excus': [401, 99], 'stupid': [1514, 294], 'aw': [1536, 188], 'young': [1298, 2323], 'laughabl': [439, 49], 'delight': [87, 461], 'money': [1594, 701], 'badli': [538, 95], 'fantast': [153, 674], 'lame': [638, 83], 'joke': [1074, 493], 'touch': [328, 917], 'garbag': [363, 67], 'fail': [1077, 391], 'heart': [397, 948], 'mess': [616, 148], 'gore': [697, 269], 'redeem': [391, 71], 'pointless': [453, 43], 'insult': [353, 68], 'embarrass': [431, 88], 'zombi': [902, 345], 'stewart': [74, 372], 'victoria': [13, 216]})
    save model


## Predict on text data


```python
def load_model(modelfile):
    print('loading model')
    f = open(modelfile, encoding='utf-8', errors='ignore')
    scores = {}
    for line in f:
        word, counts = line.strip().rsplit('\t', 1)
        scores[word] = eval(counts)
    f.close()
    return scores

def predict(featurefile, modelfile, test_x, test_y):
    doccounts, features = load_feature_words(featurefile)
    docscores = [math.log(count * 1.0 / sum(doccounts)) for count in doccounts]
    scores = load_model(modelfile)
    rcount = 0
    doccount = 0
    print('Use the test set to validate the model')

    predict_y = []

    n = 0
    while (n < len(test_x)):
        words = test_x[n]
        index1 = label2id(test_y[n])
        prevalues = list(docscores)
        for word in words:
            if word in features:
                for i in range(len(prevalues)):
                    prevalues[i] += math.log(scores[word][i])
        m = max(prevalues)
        pindex = prevalues.index(m)

        predict_y.append(id2label(pindex))

        if pindex == index1:
            rcount += 1
        doccount += 1
        n += 1
    print('Test text volume: %d, Predict the correct amount of categories: %d, Naive Bayes classifier accuracy: %f' % (doccount, rcount, rcount * 1.0 / doccount))
    
predict('./bayes_feature.txt', './bayes_model.txt', x_test, y_test)
```

    loading model
    Use the test set to validate the model
    Test text volume: 25000, Predict the correct amount of categories: 19387, Naive Bayes classifier accuracy: 0.775480


## Bayes model using Smoothing


```python
def load_feature_words(featureFile):
    f = open(featureFile, encoding='utf-8', errors='ignore')
    doccounts = eval(f.readline())
    features = set()
    for line in f:
        features.add(line.strip())
    f.close()
    return doccounts, features

def train_bayes(class_count, featurefile, modelfile, x_train, y_train):
    
    doccounts, features = load_feature_words(featurefile)
    print(doccounts)
    print(features)
    wordcount = collections.defaultdict(lambda: doc_dict())
    tcount = [0] * len(doccounts)
    
    for index, words in enumerate(x_train):
        
        index1 = label2id(y_train[index])
        
        for word in words:
            if word in features:
                tcount[index1] += 1
                wordcount[word][index1] += 1
    print('tcount: ', tcount)
    print('wordcount: ', wordcount)
    outmodel = open(modelfile, 'w', encoding='utf-8')
    print('save model')
    for k, v in wordcount.items():
        if k == '':
            continue
        scores = [(v[i] + 1) * 1.0 / (tcount[i] + len(wordcount)) * (class_count[i]/sum(class_count)) for i in range(len(v))]
        outmodel.write(k + '\t' + str(scores) + '\n')
    outmodel.close()
    
train_bayes(class_count,'./bayes_feature.txt','./bayes_model.txt', x_train, y_train)
```

    [1899087, 1953045]
    {'highli', 'plot', 'also', 'lack', 'money', 'unless', 'worst', 'bad', 'stewart', "'m", 'terribl', 'gore', 'fantast', 'decent', 'annoy', 'disappoint', 'guy', 'fail', 'wonder', 'minut', 'young', 'even', 'pathet', 'unbeliev', 'seri', 'suck', 'lame', 'alway', 'wors', 'look', 'stupid', 'brilliant', 'play', 'excus', 'laughabl', 'anyth', '2', 'victoria', 'beauti', ',', 'poorli', 'noth', 'today', 'wast', 'delight', 'embarrass', 'crap', 'garbag', 'would', '...', 'redeem', 'oh', 'tri', 'reason', 'dull', '?', 'horribl', 'touch', 'save', 'best', 'well', 'excel', 'year', 'favorit', 'heart', 'pointless', 'dumb', 'like', 'love', 'script', 'zombi', 'aw', 'amaz', 'badli', 'perform', 'joke', 'perfect', 'could', 'instead', 'ridicul', 'avoid', 'cheap', 'suppos', 'role', 'mess', 'movi', 'enjoy', 'least', 'superb', 'act', 'famili', 'poor', "n't", 'unfunni', 'insult', 'great', 'bore', 'world', 'life', 'thing'}
    tcount:  [362722, 330106]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f18c4186cb0>, {'movi': [27993, 21947], ',': [131804, 144074], "n't": [19959, 13420], 'enjoy': [1466, 2797], 'love': [2751, 5962], 'year': [2551, 3765], 'superb': [99, 549], 'could': [5684, 3673], 'play': [3565, 5009], 'perfect': [367, 1338], 'great': [2637, 6351], 'well': [3833, 5833], 'noth': [2927, 1283], 'look': [5646, 4032], 'world': [1369, 2395], 'beauti': [820, 2393], 'oh': [934, 324], 'famili': [1218, 2116], 'life': [2214, 3885], 'annoy': [943, 282], 'also': [3496, 5433], 'tri': [3808, 2491], 'act': [5120, 3318], 'favorit': [297, 1101], 'reason': [1964, 1165], 'dull': [664, 141], 'anyth': [1843, 1059], 'would': [7663, 5710], '2': [1295, 667], 'today': [313, 922], '...': [7242, 4884], '?': [11342, 4743], 'bad': [7139, 1846], 'even': [7665, 5021], 'wast': [1998, 193], 'plot': [4137, 2469], 'wonder': [1309, 2282], "'m": [2878, 1878], 'amaz': [367, 1162], 'suck': [591, 124], 'least': [1989, 1092], 'poor': [1443, 407], 'best': [2039, 4226], 'suppos': [1451, 455], 'terribl': [1544, 321], 'brilliant': [263, 885], 'like': [12020, 10138], 'guy': [2658, 1559], 'instead': [1433, 698], 'worst': [2433, 244], 'dumb': [491, 131], 'thing': [4619, 3435], 'decent': [813, 335], 'excel': [419, 1781], 'disappoint': [1200, 591], 'alway': [1138, 2067], 'pathet': [403, 63], 'minut': [2458, 1049], 'bore': [1941, 512], 'save': [1164, 521], 'perform': [1842, 3559], 'ridicul': [894, 219], 'avoid': [685, 228], 'unless': [507, 135], 'cheap': [667, 202], 'poorli': [618, 67], 'highli': [267, 828], 'seri': [1186, 2045], 'role': [1578, 2569], 'crap': [820, 143], 'wors': [1189, 209], 'unfunni': [244, 17], 'script': [2196, 1037], 'lack': [1214, 590], 'horribl': [1190, 195], 'unbeliev': [428, 113], 'excus': [401, 99], 'stupid': [1514, 294], 'aw': [1536, 188], 'young': [1298, 2323], 'laughabl': [439, 49], 'delight': [87, 461], 'money': [1594, 701], 'badli': [538, 95], 'fantast': [153, 674], 'lame': [638, 83], 'joke': [1074, 493], 'touch': [328, 917], 'garbag': [363, 67], 'fail': [1077, 391], 'heart': [397, 948], 'mess': [616, 148], 'gore': [697, 269], 'redeem': [391, 71], 'pointless': [453, 43], 'insult': [353, 68], 'embarrass': [431, 88], 'zombi': [902, 345], 'stewart': [74, 372], 'victoria': [13, 216]})
    save model


## Compare the smoothing one with the original one on test data


```python
def load_model(modelfile):
    print('loading model')
    f = open(modelfile, encoding='utf-8', errors='ignore')
    scores = {}
    for line in f:
        word, counts = line.strip().rsplit('\t', 1)
        scores[word] = eval(counts)
    f.close()
    return scores

def predict(featurefile, modelfile, test_x, test_y):
    doccounts, features = load_feature_words(featurefile)
    docscores = [math.log(count * 1.0 / sum(doccounts)) for count in doccounts]
    scores = load_model(modelfile)
    rcount = 0
    doccount = 0
    print('Use the test set to validate the model')

    predict_y = []

    n = 0
    while (n < len(test_x)):
        words = test_x[n]
        index1 = label2id(test_y[n])
        prevalues = list(docscores)
        for word in words:
            if word in features:
                for i in range(len(prevalues)):
                    prevalues[i] += math.log(scores[word][i])
        m = max(prevalues)
        pindex = prevalues.index(m)

        predict_y.append(id2label(pindex))

        if pindex == index1:
            rcount += 1
        doccount += 1
        n += 1
    print('Test text volume: %d, Predict the correct amount of categories: %d, Naive Bayes classifier accuracy: %f' % (doccount, rcount, rcount * 1.0 / doccount))
    
predict('./bayes_feature.txt', './bayes_model.txt', x_test, y_test)
```

    loading model
    Use the test set to validate the model
    Test text volume: 25000, Predict the correct amount of categories: 20269, Naive Bayes classifier accuracy: 0.810760


## Top 10 words


```python
scores = load_model('./bayes_model.txt')
# print(scores)
# keyf = lambda x: x[1][i]
label_0 = sorted(scores.items(), key = lambda kv:(kv[1][0] - kv[1][1]), reverse=True)
print('top 10 words for negative class: ')
for index, item in enumerate(label_0):
    if index < 10:
        print(item[0])
label_1 = sorted(scores.items(), key = lambda kv:(kv[1][1] - kv[1][0]), reverse=True)
print('top 10 words for positive class: ')
for index, item in enumerate(label_1):
    if index < 10:
        print(item[0])
```

    loading model
    top 10 words for negative class: 
    ?
    n't
    bad
    movi
    worst
    even
    ...
    wast
    could
    noth
    top 10 words for positive class: 
    ,
    great
    love
    best
    well
    also
    perform
    life
    play
    beauti


## K-Fold Validation


```python
from sklearn.model_selection import RepeatedKFold

kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=30)

for train_index, dev_index in kf.split(x_train):
    # print('train_index', train_index, 'test_index', test_index)
    train_x, train_y = x_train[train_index], y_train[train_index]
    dev_x, dev_y = x_train[dev_index], y_train[dev_index]
    
    train_bayes(class_count,'./bayes_feature.txt','./bayes_model.txt', train_x, train_y)
    predict('./bayes_feature.txt', './bayes_model.txt', dev_x, dev_y)
    
```

    [1899087, 1953045]
    {'highli', 'plot', 'also', 'lack', 'money', 'unless', 'worst', 'bad', 'stewart', "'m", 'terribl', 'gore', 'fantast', 'decent', 'annoy', 'disappoint', 'guy', 'fail', 'wonder', 'minut', 'young', 'even', 'pathet', 'unbeliev', 'seri', 'suck', 'lame', 'alway', 'wors', 'look', 'stupid', 'brilliant', 'play', 'excus', 'laughabl', 'anyth', '2', 'victoria', 'beauti', ',', 'poorli', 'noth', 'today', 'wast', 'delight', 'embarrass', 'crap', 'garbag', 'would', '...', 'redeem', 'oh', 'tri', 'reason', 'dull', '?', 'horribl', 'touch', 'save', 'best', 'well', 'excel', 'year', 'favorit', 'heart', 'pointless', 'dumb', 'like', 'love', 'script', 'zombi', 'aw', 'amaz', 'badli', 'perform', 'joke', 'perfect', 'could', 'instead', 'ridicul', 'avoid', 'cheap', 'suppos', 'role', 'mess', 'movi', 'enjoy', 'least', 'superb', 'act', 'famili', 'poor', "n't", 'unfunni', 'insult', 'great', 'bore', 'world', 'life', 'thing'}
    tcount:  [289522, 263780]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f18c417bcb0>, {'movi': [22297, 17518], ',': [105399, 115297], "n't": [15815, 10741], 'enjoy': [1163, 2254], 'love': [2175, 4713], 'year': [2031, 3014], 'superb': [86, 449], 'could': [4548, 2961], 'play': [2786, 3969], 'perfect': [299, 1081], 'great': [2146, 5005], 'well': [3044, 4670], 'noth': [2345, 1057], 'look': [4531, 3165], 'world': [1092, 1913], 'beauti': [640, 1925], 'reason': [1536, 911], 'dull': [544, 109], 'tri': [3056, 1980], 'anyth': [1464, 833], 'would': [6109, 4596], '2': [1042, 545], 'wonder': [1042, 1822], 'act': [4089, 2639], 'terribl': [1229, 247], 'best': [1606, 3328], 'brilliant': [205, 725], 'also': [2765, 4314], 'like': [9647, 8049], 'even': [6183, 3980], 'guy': [2140, 1244], '...': [5814, 3922], 'instead': [1142, 550], 'least': [1608, 873], 'amaz': [283, 930], 'excel': [323, 1407], 'disappoint': [964, 453], 'worst': [1957, 194], '?': [9005, 3779], 'alway': [916, 1622], 'plot': [3295, 1979], "'m": [2306, 1508], 'pathet': [323, 51], 'minut': [1961, 862], 'bad': [5653, 1483], 'bore': [1511, 410], 'wast': [1556, 154], 'ridicul': [690, 181], 'avoid': [552, 184], 'cheap': [520, 168], 'thing': [3651, 2729], 'poorli': [501, 57], 'highli': [203, 673], 'wors': [952, 166], 'suppos': [1145, 368], 'unless': [405, 111], 'unfunni': [201, 15], 'script': [1769, 841], 'lack': [1000, 453], 'life': [1730, 3091], 'horribl': [959, 153], 'famili': [950, 1721], 'perform': [1459, 2862], 'unbeliev': [341, 94], 'excus': [320, 77], 'favorit': [242, 889], 'stupid': [1208, 227], 'dumb': [403, 96], 'suck': [479, 103], 'aw': [1244, 150], 'seri': [953, 1667], 'young': [1011, 1875], 'today': [249, 747], 'delight': [69, 365], 'money': [1281, 584], 'role': [1271, 2030], 'lame': [515, 69], 'decent': [655, 253], 'joke': [850, 398], 'touch': [264, 725], 'laughabl': [339, 38], 'badli': [429, 80], 'crap': [649, 106], 'garbag': [308, 52], 'fail': [869, 313], 'annoy': [741, 223], 'save': [921, 411], 'poor': [1151, 330], 'mess': [479, 120], 'oh': [764, 246], 'gore': [580, 206], 'redeem': [321, 54], 'heart': [316, 763], 'fantast': [120, 552], 'insult': [292, 51], 'embarrass': [342, 71], 'pointless': [363, 39], 'stewart': [60, 305], 'zombi': [756, 273], 'victoria': [9, 194]})
    save model
    loading model
    Use the test set to validate the model
    Test text volume: 5000, Predict the correct amount of categories: 4071, Naive Bayes classifier accuracy: 0.814200
    [1899087, 1953045]
    {'highli', 'plot', 'also', 'lack', 'money', 'unless', 'worst', 'bad', 'stewart', "'m", 'terribl', 'gore', 'fantast', 'decent', 'annoy', 'disappoint', 'guy', 'fail', 'wonder', 'minut', 'young', 'even', 'pathet', 'unbeliev', 'seri', 'suck', 'lame', 'alway', 'wors', 'look', 'stupid', 'brilliant', 'play', 'excus', 'laughabl', 'anyth', '2', 'victoria', 'beauti', ',', 'poorli', 'noth', 'today', 'wast', 'delight', 'embarrass', 'crap', 'garbag', 'would', '...', 'redeem', 'oh', 'tri', 'reason', 'dull', '?', 'horribl', 'touch', 'save', 'best', 'well', 'excel', 'year', 'favorit', 'heart', 'pointless', 'dumb', 'like', 'love', 'script', 'zombi', 'aw', 'amaz', 'badli', 'perform', 'joke', 'perfect', 'could', 'instead', 'ridicul', 'avoid', 'cheap', 'suppos', 'role', 'mess', 'movi', 'enjoy', 'least', 'superb', 'act', 'famili', 'poor', "n't", 'unfunni', 'insult', 'great', 'bore', 'world', 'life', 'thing'}
    tcount:  [290476, 263757]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f1871154e60>, {',': [105725, 115069], 'great': [2127, 5125], 'well': [3091, 4620], 'noth': [2296, 1003], 'look': [4480, 3196], 'world': [1103, 1936], 'could': [4553, 2917], 'beauti': [693, 1884], 'enjoy': [1157, 2209], "n't": [15949, 10771], 'love': [2183, 4829], 'oh': [735, 266], 'year': [1995, 2991], 'famili': [969, 1693], 'movi': [22292, 17528], 'life': [1792, 3106], 'annoy': [734, 234], 'also': [2788, 4339], 'tri': [3071, 1974], 'act': [4074, 2674], 'favorit': [240, 860], 'reason': [1582, 945], 'dull': [507, 118], 'anyth': [1458, 852], 'would': [6107, 4578], 'play': [2901, 4058], '2': [1033, 540], 'today': [243, 730], '...': [5845, 3880], '?': [9062, 3778], 'bad': [5674, 1475], 'even': [6142, 4038], 'wast': [1588, 155], 'plot': [3336, 1930], 'wonder': [1066, 1835], "'m": [2302, 1538], 'amaz': [298, 898], 'suck': [473, 108], 'least': [1607, 874], 'poor': [1163, 326], 'best': [1620, 3393], 'suppos': [1130, 373], 'terribl': [1245, 261], 'brilliant': [218, 689], 'worst': [1993, 189], 'dumb': [401, 104], 'like': [9634, 8086], 'guy': [2098, 1279], 'thing': [3705, 2735], 'decent': [656, 264], 'alway': [924, 1665], 'pathet': [318, 50], 'minut': [1952, 858], 'bore': [1581, 415], 'save': [936, 409], 'perform': [1455, 2859], 'unless': [405, 104], 'seri': [926, 1672], 'role': [1265, 1999], 'crap': [669, 108], 'excel': [335, 1408], 'wors': [938, 164], 'unfunni': [199, 11], 'avoid': [536, 178], 'script': [1786, 808], 'lack': [954, 490], 'perfect': [294, 1067], 'horribl': [963, 159], 'stupid': [1231, 233], 'aw': [1282, 157], 'young': [1083, 1841], 'instead': [1145, 555], 'laughabl': [359, 39], 'delight': [67, 380], 'money': [1264, 557], 'badli': [439, 68], 'fantast': [129, 524], 'superb': [73, 459], 'lame': [507, 66], 'disappoint': [931, 472], 'excus': [314, 69], 'garbag': [291, 57], 'poorli': [492, 51], 'fail': [867, 309], 'joke': [878, 396], 'heart': [315, 732], 'highli': [228, 673], 'ridicul': [720, 169], 'touch': [255, 738], 'mess': [487, 110], 'gore': [565, 225], 'cheap': [529, 158], 'redeem': [319, 55], 'pointless': [372, 33], 'unbeliev': [343, 93], 'insult': [286, 58], 'embarrass': [334, 66], 'zombi': [734, 279], 'stewart': [55, 297], 'victoria': [12, 161]})
    save model
    loading model
    Use the test set to validate the model
    Test text volume: 5000, Predict the correct amount of categories: 4084, Naive Bayes classifier accuracy: 0.816800
    [1899087, 1953045]
    {'highli', 'plot', 'also', 'lack', 'money', 'unless', 'worst', 'bad', 'stewart', "'m", 'terribl', 'gore', 'fantast', 'decent', 'annoy', 'disappoint', 'guy', 'fail', 'wonder', 'minut', 'young', 'even', 'pathet', 'unbeliev', 'seri', 'suck', 'lame', 'alway', 'wors', 'look', 'stupid', 'brilliant', 'play', 'excus', 'laughabl', 'anyth', '2', 'victoria', 'beauti', ',', 'poorli', 'noth', 'today', 'wast', 'delight', 'embarrass', 'crap', 'garbag', 'would', '...', 'redeem', 'oh', 'tri', 'reason', 'dull', '?', 'horribl', 'touch', 'save', 'best', 'well', 'excel', 'year', 'favorit', 'heart', 'pointless', 'dumb', 'like', 'love', 'script', 'zombi', 'aw', 'amaz', 'badli', 'perform', 'joke', 'perfect', 'could', 'instead', 'ridicul', 'avoid', 'cheap', 'suppos', 'role', 'mess', 'movi', 'enjoy', 'least', 'superb', 'act', 'famili', 'poor', "n't", 'unfunni', 'insult', 'great', 'bore', 'world', 'life', 'thing'}
    tcount:  [290183, 264898]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f18c41e7ef0>, {'movi': [22418, 17473], ',': [105446, 115540], "n't": [15990, 10747], 'enjoy': [1162, 2268], 'love': [2243, 4797], 'year': [2059, 3063], 'superb': [79, 450], 'could': [4495, 2959], 'play': [2847, 4051], 'perfect': [288, 1078], 'oh': [757, 259], 'famili': [992, 1703], 'life': [1752, 3144], 'annoy': [771, 233], 'also': [2791, 4278], 'tri': [3069, 1979], 'act': [4101, 2689], 'favorit': [239, 865], 'today': [262, 722], '...': [5975, 3954], '?': [9057, 3851], 'bad': [5752, 1504], 'even': [6066, 4059], 'wast': [1608, 150], 'plot': [3264, 1974], 'wonder': [1053, 1801], "'m": [2297, 1491], 'amaz': [298, 934], 'suck': [447, 98], 'least': [1568, 865], 'poor': [1138, 328], 'best': [1632, 3415], 'suppos': [1184, 369], 'great': [2109, 5077], 'terribl': [1219, 252], 'brilliant': [217, 693], 'beauti': [646, 1895], 'like': [9617, 8162], 'look': [4508, 3254], 'guy': [2146, 1229], 'instead': [1141, 545], 'well': [3022, 4770], 'would': [6187, 4642], 'worst': [1894, 188], 'dumb': [392, 106], 'world': [1085, 1929], 'thing': [3711, 2763], 'decent': [666, 283], 'excel': [355, 1427], 'disappoint': [963, 488], 'alway': [893, 1636], 'reason': [1583, 944], 'pathet': [334, 51], 'minut': [1975, 818], 'bore': [1536, 399], 'save': [889, 417], 'perform': [1489, 2816], 'ridicul': [726, 173], 'avoid': [546, 182], 'unless': [416, 107], 'cheap': [524, 164], 'poorli': [472, 55], 'highli': [212, 652], 'seri': [933, 1612], 'noth': [2334, 1022], 'role': [1252, 2120], 'crap': [647, 123], 'wors': [942, 167], 'unfunni': [192, 15], 'anyth': [1455, 847], '2': [1041, 533], 'unbeliev': [346, 93], 'excus': [324, 84], 'lack': [961, 460], 'stupid': [1224, 248], 'aw': [1217, 152], 'young': [1037, 1864], 'laughabl': [347, 44], 'badli': [427, 79], 'fantast': [124, 540], 'lame': [498, 70], 'joke': [854, 393], 'touch': [257, 755], 'money': [1279, 553], 'garbag': [293, 53], 'fail': [837, 328], 'script': [1750, 855], 'heart': [310, 746], 'mess': [518, 124], 'delight': [70, 367], 'gore': [552, 218], 'dull': [533, 107], 'pointless': [356, 31], 'redeem': [297, 62], 'insult': [286, 58], 'horribl': [959, 156], 'embarrass': [362, 77], 'zombi': [722, 273], 'stewart': [63, 290], 'victoria': [11, 171]})
    save model
    loading model
    Use the test set to validate the model
    Test text volume: 5000, Predict the correct amount of categories: 4059, Naive Bayes classifier accuracy: 0.811800
    [1899087, 1953045]
    {'highli', 'plot', 'also', 'lack', 'money', 'unless', 'worst', 'bad', 'stewart', "'m", 'terribl', 'gore', 'fantast', 'decent', 'annoy', 'disappoint', 'guy', 'fail', 'wonder', 'minut', 'young', 'even', 'pathet', 'unbeliev', 'seri', 'suck', 'lame', 'alway', 'wors', 'look', 'stupid', 'brilliant', 'play', 'excus', 'laughabl', 'anyth', '2', 'victoria', 'beauti', ',', 'poorli', 'noth', 'today', 'wast', 'delight', 'embarrass', 'crap', 'garbag', 'would', '...', 'redeem', 'oh', 'tri', 'reason', 'dull', '?', 'horribl', 'touch', 'save', 'best', 'well', 'excel', 'year', 'favorit', 'heart', 'pointless', 'dumb', 'like', 'love', 'script', 'zombi', 'aw', 'amaz', 'badli', 'perform', 'joke', 'perfect', 'could', 'instead', 'ridicul', 'avoid', 'cheap', 'suppos', 'role', 'mess', 'movi', 'enjoy', 'least', 'superb', 'act', 'famili', 'poor', "n't", 'unfunni', 'insult', 'great', 'bore', 'world', 'life', 'thing'}
    tcount:  [289194, 265604]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f1871154dd0>, {'movi': [22145, 17737], ',': [105131, 115844], "n't": [16015, 10816], 'enjoy': [1173, 2233], 'love': [2222, 4765], 'year': [2016, 3016], 'superb': [75, 417], 'could': [4543, 2926], 'play': [2810, 4042], 'perfect': [284, 1056], 'great': [2055, 5123], 'well': [3068, 4645], 'noth': [2356, 1036], 'look': [4521, 3299], 'world': [1115, 1908], 'beauti': [634, 1941], 'oh': [742, 255], 'famili': [985, 1721], 'life': [1762, 3133], 'annoy': [775, 222], 'also': [2793, 4448], 'tri': [2969, 2025], 'act': [4091, 2661], 'favorit': [234, 884], 'reason': [1606, 944], 'dull': [544, 113], 'anyth': [1523, 859], 'would': [6059, 4543], '2': [1031, 515], 'today': [242, 750], '...': [5628, 3867], '?': [9151, 3755], 'bad': [5699, 1469], 'even': [6088, 4023], 'wast': [1605, 162], 'plot': [3334, 1989], 'wonder': [1028, 1878], "'m": [2276, 1512], 'amaz': [287, 940], 'suck': [473, 91], 'least': [1578, 874], 'poor': [1142, 324], 'best': [1643, 3379], 'suppos': [1183, 360], 'like': [9555, 8179], 'guy': [2119, 1248], 'instead': [1145, 571], 'worst': [1915, 209], 'dumb': [383, 108], 'thing': [3737, 2777], 'decent': [651, 261], 'excel': [322, 1461], 'disappoint': [969, 475], 'brilliant': [210, 739], 'alway': [888, 1687], 'save': [954, 434], 'perform': [1498, 2898], 'ridicul': [720, 177], 'avoid': [565, 185], 'unless': [412, 106], 'cheap': [540, 156], 'poorli': [501, 50], 'highli': [209, 652], 'pathet': [324, 53], 'seri': [968, 1601], 'role': [1252, 2103], 'crap': [648, 113], 'wors': [970, 171], 'unfunni': [195, 14], 'minut': [1959, 825], 'script': [1739, 840], 'lack': [988, 478], 'horribl': [937, 155], 'unbeliev': [335, 86], 'excus': [323, 84], 'young': [1036, 1867], 'laughabl': [349, 38], 'delight': [70, 373], 'money': [1242, 581], 'terribl': [1251, 256], 'badli': [402, 76], 'fantast': [120, 538], 'lame': [525, 63], 'joke': [846, 398], 'stupid': [1195, 236], 'touch': [276, 745], 'bore': [1562, 412], 'fail': [843, 303], 'heart': [316, 773], 'aw': [1188, 150], 'redeem': [312, 53], 'pointless': [361, 36], 'gore': [536, 213], 'mess': [499, 125], 'garbag': [275, 56], 'embarrass': [346, 71], 'insult': [282, 53], 'zombi': [729, 288], 'stewart': [55, 330], 'victoria': [13, 174]})
    save model
    loading model
    Use the test set to validate the model
    Test text volume: 5000, Predict the correct amount of categories: 4061, Naive Bayes classifier accuracy: 0.812200
    [1899087, 1953045]
    {'highli', 'plot', 'also', 'lack', 'money', 'unless', 'worst', 'bad', 'stewart', "'m", 'terribl', 'gore', 'fantast', 'decent', 'annoy', 'disappoint', 'guy', 'fail', 'wonder', 'minut', 'young', 'even', 'pathet', 'unbeliev', 'seri', 'suck', 'lame', 'alway', 'wors', 'look', 'stupid', 'brilliant', 'play', 'excus', 'laughabl', 'anyth', '2', 'victoria', 'beauti', ',', 'poorli', 'noth', 'today', 'wast', 'delight', 'embarrass', 'crap', 'garbag', 'would', '...', 'redeem', 'oh', 'tri', 'reason', 'dull', '?', 'horribl', 'touch', 'save', 'best', 'well', 'excel', 'year', 'favorit', 'heart', 'pointless', 'dumb', 'like', 'love', 'script', 'zombi', 'aw', 'amaz', 'badli', 'perform', 'joke', 'perfect', 'could', 'instead', 'ridicul', 'avoid', 'cheap', 'suppos', 'role', 'mess', 'movi', 'enjoy', 'least', 'superb', 'act', 'famili', 'poor', "n't", 'unfunni', 'insult', 'great', 'bore', 'world', 'life', 'thing'}
    tcount:  [291513, 262385]
    wordcount:  defaultdict(<function train_bayes.<locals>.<lambda> at 0x7f18c41e7ef0>, {'movi': [22820, 17532], ',': [105515, 114546], "n't": [16067, 10605], 'enjoy': [1209, 2224], 'love': [2181, 4744], 'year': [2103, 2976], 'superb': [83, 421], 'could': [4597, 2929], 'play': [2916, 3916], 'perfect': [303, 1070], 'great': [2111, 5074], 'well': [3107, 4627], 'noth': [2377, 1014], 'look': [4544, 3214], 'world': [1081, 1894], 'beauti': [667, 1927], 'oh': [738, 270], 'famili': [976, 1626], 'life': [1820, 3066], 'annoy': [751, 216], 'also': [2847, 4353], 'tri': [3067, 2006], 'act': [4125, 2609], 'favorit': [233, 906], 'reason': [1549, 916], 'dull': [528, 117], 'anyth': [1472, 845], 'would': [6190, 4481], '2': [1033, 535], 'today': [256, 739], '...': [5706, 3913], '?': [9093, 3809], 'bad': [5778, 1453], 'even': [6181, 3984], 'wast': [1635, 151], 'plot': [3319, 2004], 'wonder': [1047, 1792], "'m": [2331, 1463], 'amaz': [302, 946], 'suck': [492, 96], 'least': [1595, 882], 'poor': [1178, 320], 'best': [1655, 3389], 'suppos': [1162, 350], 'terribl': [1232, 268], 'brilliant': [202, 694], 'like': [9627, 8076], 'guy': [2129, 1236], 'instead': [1159, 571], 'worst': [1973, 196], 'dumb': [385, 110], 'thing': [3672, 2736], 'decent': [624, 279], 'excel': [341, 1421], 'disappoint': [973, 476], 'alway': [931, 1658], 'pathet': [313, 47], 'minut': [1985, 833], 'bore': [1574, 412], 'save': [956, 413], 'perform': [1467, 2801], 'ridicul': [720, 176], 'avoid': [541, 183], 'unless': [390, 112], 'cheap': [555, 162], 'poorli': [506, 55], 'highli': [216, 662], 'seri': [964, 1628], 'role': [1272, 2024], 'crap': [667, 122], 'script': [1740, 804], 'lack': [953, 479], 'horribl': [942, 157], 'unbeliev': [347, 86], 'excus': [323, 82], 'stupid': [1198, 232], 'aw': [1213, 143], 'laughabl': [362, 37], 'delight': [72, 359], 'money': [1310, 529], 'wors': [954, 168], 'badli': [455, 77], 'fantast': [119, 542], 'joke': [868, 387], 'touch': [260, 705], 'young': [1025, 1845], 'garbag': [285, 50], 'lame': [507, 64], 'heart': [331, 778], 'mess': [481, 113], 'fail': [892, 311], 'gore': [555, 214], 'redeem': [315, 60], 'pointless': [360, 33], 'unfunni': [189, 13], 'insult': [266, 52], 'embarrass': [340, 67], 'zombi': [667, 267], 'stewart': [63, 266], 'victoria': [7, 164]})
    save model
    loading model
    Use the test set to validate the model
    Test text volume: 5000, Predict the correct amount of categories: 4095, Naive Bayes classifier accuracy: 0.819000

