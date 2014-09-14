# -*- coding: utf-8 -*- 

s1 = 'line Ä, 궯, 奠 end' 
s2 = 'line Ä, 궯, end' 
print(type(s1),s1)

import nltk

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

from nltk.tokenize import RegexpTokenizer
regexp_tokenizer = RegexpTokenizer(r'\w+')


def get_group_messages(group_id):
    vkapi = vk.API(creds.APP_ID, 'gennad.zlobin@gmail.com', creds.APP_SECRET, timeout=10)

    offset = 0

    group_id = int(group_id)
    if group_id > 0:
        group_id = -group_id
        
    m = vkapi('wall.get', owner_id=group_id, count=100, offset=offset)

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
    output = open(filename, 'wb')
    pickle.dump(data, output)


from nltk.corpus import stopwords


def bag_of_non_stopwords(words, stopfile='russian'):
    badwords = stopwords.words(stopfile)
    return bag_in_words_not_in_set(words, badwords)

def bag_of_words(words):
    dict = {}
    for word in words:
        if len(word) < 4: continue
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



punkt_tokenizer = nltk.tokenize.PunktSentenceTokenizer()


def analyze():
    # coding: utf-8
    import pickle
    import nltk

    dct = dict(
        android=pickle.load(open('android', 'rb'))[:400],
        vatnik=pickle.load(open('vatnik', 'rb'))[:400],
        kinomania=pickle.load(open('kinomania', 'rb'))[:400],
        vk_science=pickle.load(open('vk_science', 'rb'))[:400]
    )


    for key, sentences in dct.items():
        dct[key] = [punkt_tokenizer.tokenize(sentence) for sentence in sentences]
        dct[key] = [item.lower() for sublist in dct[key] for item in sublist]
        dct[key] = [regexp_tokenizer.tokenize(i) for i in dct[key]]
        dct[key] = label_feats_from_corpus(dct[key], key)


    train_feats, test_feats = split_label_feats(dct)

    import ipdb; ipdb.set_trace()
    print (len(train_feats))
    #print (len(test_feats))

    from nltk.classify import NaiveBayesClassifier
    nb_classifier = NaiveBayesClassifier.train(train_feats)


    print (nb_classifier.labels())
    from nltk.classify.util import accuracy
    print ('accuracy', accuracy(nb_classifier, test_feats))
    nb_classifier.show_most_informative_features(20)
    return nb_classifier


classifier = analyze()

RASHKA_GROUP_ID = 37009309
#save_group_messages(RASHKA_GROUP_ID, 'vatnik')

ANDROID_INSIDER_GROUP_ID = 47433299
#save_group_messages(ANDROID_INSIDER_GROUP_ID, 'android')

KINOMANIA_GROUP_ID = 43215063
#save_group_messages(KINOMANIA_GROUP_ID, 'kinomania')

VK_SCIENCE_GROUP_ID = 29559271
#save_group_messages(VK_SCIENCE_GROUP_ID, 'vk_science')

def get_list_of_feed():
    vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')
    result = vkapi('newsfeed.get', filters='post')
    lst = []
    for msg in result['items']:
        lst.append(msg['text'])
    return lst

lst = get_list_of_feed()


def classify_feed(lst, classifier):

    #lst = [punkt_tokenizer.tokenize(sentence) for sentence in lst]
    #lst = [item.lower() for sublist in lst for item in sublist]
    lst = [item.lower() for item in lst]
    lst = [regexp_tokenizer.tokenize(i) for i in lst]


    for i in lst:
        print (classifier.classify(bag_of_words(i)))
        print (i)


classify_feed(lst, classifier)


