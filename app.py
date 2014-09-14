# -*- coding: utf-8 -*- 


import nltk


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
import os

from nltk.tokenize import RegexpTokenizer
regexp_tokenizer = RegexpTokenizer(r'\w+')



publics = {
    'auto': [
        'typical_motorcyclist',
        'moto_msk'

    ],
    'sport': [
        'ligatv',
        'olympicsrus',
        'sportexpress'
    ],
    'travel': [
        'natgeoru',
        'goodtravels'
    ],

    'hightech': [
        'icommunity',
        'androidinsider',
        'habr'
    ],
    'science': [
        'obrazovach',
        'nakedsci',
        'postnauka',
    ],
    'city': [
        'moyamsk',
        'piter',
        'blognsk'
    ],
    'english': [
        'beginenglish_ru',
        'luv_english',
        'english_is_fun'
    ],
    'politics_pro_ukr': [
        'westernclub',
        'public_rushka'
    ],
    'news': [
        '1tvnews',
        'kpru',
        'ria'
    ],
    'movies': [
        'hd_kino_mania'
    ]
}






def get_group_messages(group_id, limit):
    #vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')
    #vkapi = vk.API(app_id=creds.APP_ID,  access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications', user_login='gennad.zlobin@googlemail.com')

    vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')

    offset = 0

    screen_name = group_id

    try:
        group_id = vkapi('groups.getById', group_id=group_id)[0]['id']
    except vk.api.VkAPIMethodError as e:
        #import ipdb; ipdb.set_trace()
        print (screen_name, group_id)

    if isinstance(group_id, int) and group_id > 0:
        group_id = -group_id

    try:
        m = vkapi('wall.get', owner_id=group_id, count=100, offset=offset)
    except vk.api.VkAPIMethodError as e:
        #import ipdb; ipdb.set_trace()
        print (screen_name, group_id)
        print (123)

    count = int(m['count'])
    offset += 100

    for i in m['items']:
        yield i['text']

    while offset < min(count, limit):
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




def cache_publics():
    for name, screen_names in publics.items():
        if os.path.exists(name):
            continue

        messages = []
        #import ipdb; ipdb.set_trace()

        for screen_name in screen_names:
            limit = 300

            for i in get_group_messages(screen_name, limit):
                if i.strip():
                    messages.append(i)

        output = open(name, 'wb')
        pickle.dump(messages, output)

cache_publics()



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

    dct = {}

    for name, screen_names in publics.items():
        pickled = pickle.load(open(name, 'rb'))[:400],
        #import ipdb; ipdb.set_trace()
        dct[name] = pickled[0]


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

vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')


classifier = analyze()


def get_list_of_feed():
    vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')
    result = vkapi('newsfeed.get', filters='post', count=100)
    lst = []
    for msg in result['items']:
        lst.append((msg['text'], msg['source_id']))


    result = vkapi('newsfeed.get', filters='post', count=100, new_offset=100)
    for msg in result['items']:
        lst.append((msg['text'], msg['source_id']))

    result = vkapi('newsfeed.get', filters='post', count=100, new_offset=200)
    for msg in result['items']:
        lst.append((msg['text'], msg['source_id']))

    return lst




def classify_feed(lst, classifier):

    #lst = [punkt_tokenizer.tokenize(sentence) for sentence in lst]
    #lst = [item.lower() for sublist in lst for item in sublist]
    lst = [(msg.lower(), source_id) for msg, source_id in lst]
    lst = [(regexp_tokenizer.tokenize(i), source_id) for i, source_id in lst]

    res = collections.defaultdict(list)


    for i, source_id in lst:
        if not i:
            continue

        category = classifier.classify(bag_of_words(i))
        print (category)
        print (i)

        res[source_id].append(category)

    print (res)
    import ipdb; ipdb.set_trace()
    for i, j in res.items():
        time.sleep(1)
        screen_name = vkapi('groups.getById', group_id=abs(i))[0]['screen_name']
        print (screen_name, max(j))

lst = get_list_of_feed()
classify_feed(lst, classifier)

