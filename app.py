# -*- coding: utf-8 -*- 

s1 = 'line Ä, 궯, 奠 end' 
s2 = 'line Ä, 궯, end' 
print(type(s1),s1)

s1 = 'line Ä, 궯, 奠'
print(type(s1),s1)

import pymorphy2
morph = pymorphy2.MorphAnalyzer()



import vk
import time
import pickle
import collections

import sys, codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

from nltk.stem.snowball import RussianStemmer
russian_stemmer = RussianStemmer()

import creds



#profiles = vkapi('users.get', user_id=790781)
#print(vkapi.getServerTime())


def get_group_messages(group_id):
    vkapi = vk.API(creds.APP_ID, 'gennad.zlobin@gmail.com', creds.APP_SECRET, timeout=10)

    offset = 0

    group_id = int(group_id)
    if group_id > 0:
        group_id = -group_id
        
    #m = vkapi('wall.get', owner_id=-37009309, count=100, offset=offset).items()
    m = vkapi('wall.get', owner_id=group_id, count=100, offset=offset)
    #import ipdb; ipdb.set_trace()

    count = int(m['count'])
    offset += 100

    for i in m['items']:
        yield i['text']

    while offset < count:
        m = vkapi('wall.get', owner_id=group_id, count=100, offset=offset)
        #_ = int(next(m)[1])

        for i in m['items']:
            yield i['text']
        offset += 100
        time.sleep(3)


def save_group_messages(group_id, filename):
    data = [i for i in get_group_messages(group_id) if i.strip()]
    #import ipdb; ipdb.set_trace()
    output = open(filename, 'wb')
    pickle.dump(data, output)

RASHKA_GROUP_ID = 37009309
#save_group_messages(RASHKA_GROUP_ID, 'vatnik')

ANDROID_INSIDER_GROUP_ID = 47433299
#save_group_messages(ANDROID_INSIDER_GROUP_ID, 'android')

from nltk.corpus import stopwords


def bag_of_non_stopwords(words, stopfile='russian'):
    badwords = stopwords.words(stopfile)
    return bag_in_words_not_in_set(words, badwords)

def bag_of_words(words):
    dict = {}
    for word in words:
        #normal_form = morph.parse(word)[0].normal_form
        normal_form = russian_stemmer.stem(word)
        dict[normal_form] = True

    return dict

    #return dict([(word, True) for word in words])


def bag_in_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))



from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def bag_of_bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


def label_feats_from_corpus(corp, label, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    #for label in corp.categories:
    #    for fileid in corp.fileids(categories=[label]):
    res = []
    for msg in corp:
        feats = feature_detector(msg)
        #label_feats[label].append(feats)
        res.append(feats)
    return res


def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []

    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff ]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])

    return train_feats, test_feats



def analyze():
    # coding: utf-8
    import pickle
    import nltk

    android = pickle.load(open('android', 'rb'))
    vatnik = pickle.load(open('vatnik', 'rb'))

    punkt_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    android = [punkt_tokenizer.tokenize(sentence) for sentence in android]
    vatnik = [punkt_tokenizer.tokenize(sentence) for sentence in vatnik]

    #android = [i[:-1] for i in android]
    #vatnik = [i[:-1] for i in vatnik]

    android = [item.lower() for sublist in android for item in sublist]
    vatnik = [item.lower() for sublist in vatnik for item in sublist]


    from nltk.tokenize import RegexpTokenizer

    regexp_tokenizer = RegexpTokenizer(r'\w+')
    regexp_tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')

    #white_space_tokenizer = nltk.tokenize.WhitespaceTokenizer()

    android = [regexp_tokenizer.tokenize(i) for i in android]
    vatnik = [regexp_tokenizer.tokenize(i) for i in vatnik]


    android_feats = label_feats_from_corpus(android, 'android')
    vatnik_feats = label_feats_from_corpus(vatnik, 'vatnik')






    #bag_android = bag_of_words(android)
    #bag_vatnik = bag_of_words(vatnik)

    dct = dict(
        vatnik=vatnik_feats,
        android=android_feats
    )

    train_feats, test_feats = split_label_feats(dct)

    import ipdb; ipdb.set_trace()
    print (len(train_feats))
    print (len(test_feats))

    from nltk.classify import NaiveBayesClassifier
    nb_classifier = NaiveBayesClassifier.train(train_feats)
    import ipdb; ipdb.set_trace()

    print (nb_classifier.labels())
    from nltk.classify.util import accuracy
    print ('accuracy', accuracy(nb_classifier, test_feats))







    a = 1


analyze()









"""
output = open('vatnik', 'r')
txt = pickle.load(output)
import ipdb; ipdb.set_trace()
a = 1
"""

