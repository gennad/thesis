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

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords, reuters
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

import collections
from nltk import metrics
import pickle
import ipdb

from nltk.classify import MultiClassifierI
from nltk.classify.megam import config_megam



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

vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')



punkt_tokenizer = nltk.tokenize.PunktSentenceTokenizer()


def get_user_messages(screen_name_or_user_id, limit):
    offset = 0

    try:
        m = vkapi('wall.get', owner_id=screen_name_or_user_id, count=100, offset=offset)
    except vk.api.VkAPIMethodError as e:
        #print (screen_name, group_id)
        #print (123)
        return
    except vk.api.VkAPIMethodError:
        return

    count = int(m['count'])
    offset += 100

    for i in m['items']:
        yield i['text']
        #print( i['text'])

    while offset < min(count, limit):
        m = vkapi('wall.get', owner_id=screen_name_or_user_id, count=100, offset=offset)
        #_ = int(next(m)[1])
        for i in m['items']:
            yield i['text']
            #print (i['text'])
        offset += 100
        time.sleep(3)


def get_group_messages(group_id, limit):
    vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')


    offset = 0

    screen_name = group_id

    try:
        group_id = vkapi('groups.getById', group_id=group_id)[0]['id']
    except vk.api.VkAPIMethodError as e:
        #print (screen_name, group_id)
        return

    if isinstance(group_id, int) and group_id > 0:
        group_id = -group_id

    try:
        m = vkapi('wall.get', owner_id=group_id, count=100, offset=offset)
    except vk.api.VkAPIMethodError as e:
        #print (screen_name, group_id)
        #print (123)
        return
    except vk.api.VkAPIMethodError:
        return

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

        for screen_name in screen_names:
            limit = 300

            for i in get_group_messages(screen_name, limit):
                if i.strip():
                    messages.append(i)

        output = open(name, 'wb')
        pickle.dump(messages, output)




def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = metrics.precision(refsets[label], testsets[label])
        recalls[label] = metrics.recall(refsets[label], testsets[label])

    return precisions, recalls




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




def bag_of_bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


#def label_feats_from_corpus(corp, label, feature_detector=bag_of_words):
def label_feats_from_corpus(corp, label=None, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    #for label in corp.categories:
    #    for fileid in corp.fileids(categories=[label]):
    res = []
    if not isinstance(corp, list):
        for label in corp.categories():
            for fileid in corp.fileids(categories=[label]):
                feats = feature_detector(corp.words(fileids=[fileid]))
                label_feats[label].append(feats)
        return label_feats

    for msg in corp:
        feats = feature_detector(msg)
        #label_feats[label].append(feats)
        res.append(feats)
    return res


def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)



"""
def label_feats_from_corpus(corp, feature_detector=bag_of_bigrams_words):
    label_feats = collections.defaultdict(list)
    import ipdb; ipdb.set_trace()
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
                feats = feature_detector(corp.words(fileids=[fileid]))
                label_feats[label].append(feats)
    return label_feats
"""


def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []

    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, [label]) for feat in feats[:cutoff ]])
        test_feats.extend([(feat, [label]) for feat in feats[cutoff:]])

    return train_feats, test_feats



def get_feats2():

    dct = {}

    for name, screen_names in publics.items():
        pickled = pickle.load(open(name, 'rb'))[:400],
        dct[name] = pickled[0]  # [1] contains nothing


    for key, sentences in dct.items():
        dct[key] = [punkt_tokenizer.tokenize(sentence) for sentence in sentences]
        dct[key] = [item.lower() for sublist in dct[key] for item in sublist]
        dct[key] = [regexp_tokenizer.tokenize(i) for i in dct[key]]
        dct[key] = label_feats_from_corpus(dct[key], label=key)


    train_feats, test_feats = split_label_feats(dct)
    return train_feats, test_feats


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
    for i, j in res.items():
        time.sleep(1)
        screen_name = vkapi('groups.getById', group_id=abs(i))[0]['screen_name']
        print (screen_name, max(j))





def get_friend_groups(user_id=4356905):
    vkapi = vk.API(app_id=creds.APP_ID, user_login='gennad.zlobin@googlemail.com', user_password=creds.USER_PASSWORD, access_token=creds.APP_SECRET, timeout=10, scope='offline,friends,wall,groups,notifications')
    group_ids = vkapi('groups.get', user_id=user_id)
    
    res = collections.defaultdict(list)

    for group_id in group_ids['items']:
        for i in get_group_messages(group_id, 100):
            if i.strip():
                res[group_id].append(i)
    return res


#res = get_friend_groups()




#print (len(train_feats))
#print (len(test_feats))

"""
nb_classifier = NaiveBayesClassifier.train(train_feats)
print (nb_classifier.labels())
print('accuracy nb=', accuracy(nb_classifier, test_feats))
nb_classifier.show_most_informative_features(20)

from nltk.classify import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
print('accuracy dt=', accuracy(dt_classifier, test_feats))

"""

from nltk.classify import MaxentClassifier
"""
me_classifier = MaxentClassifier.train(train_feats, algorithm='iis', trace=0, max_iter=1, min_lldelta=0.5)
print ('me=', accuracy(me_classifier, test_feats))
me_classifier.show_most_informative_features(n=10)
"""




def bag_of_words_in_set(words, wordset):
    return bag_of_words(set(words) & wordset)




# todo f_measure



from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, words in labelled_words.items():
        for word in words:
            word = word.lower()
            word = russian_stemmer.stem(word)

            word_fd[word] += 1
            label_word_fd[label][word] += 1
            #label_word_fd[label].inc(word)

    n_xx = label_word_fd.N()
    high_info_words = set()

    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        words_scores = collections.defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            words_scores[word] = score

        bestwords = [word for word, score in words_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)
    return high_info_words



def reuters_high_info_words(score_fn=BigramAssocMeasures.chi_sq):
    """
    labeled_words = []
    for label in reuters.categories():
        labeled_words.append((label, reuters.words(categories=[label])))

    return high_information_words(labeled_words, score_fn=score_fn)
    """

    categories_to_features = collections.defaultdict(dict)
    categories_to_words = collections.defaultdict(list)
    categories_to_high_info_words = collections.defaultdict(set)
    categories_to_messages = collections.defaultdict(list)

    for name, screen_names in publics.items():
        pickled = pickle.load(open(name, 'rb'))[:400],
        list_of_messages = pickled[0] # [1] contains nothing

        for msg in list_of_messages:
            categories_to_messages[name].append(msg)

            list_of_sentences = punkt_tokenizer.tokenize(msg)
            list_of_words_list = [regexp_tokenizer.tokenize(sentence) for sentence in list_of_sentences]

            for list_of_words in list_of_words_list:
                categories_to_words[name].extend(list_of_words)

            list_of_words_dicts = label_feats_from_corpus(list_of_words_list)

            for dct in list_of_words_dicts:
                categories_to_features[name].update(dct)


    return high_information_words(categories_to_words, score_fn=score_fn)


def reuters_train_test_feats(feature_detector=bag_of_words):


    categories_to_features = collections.defaultdict(list)
    categories_to_words = collections.defaultdict(list)
    categories_to_high_info_words = collections.defaultdict(set)
    categories_to_messages = collections.defaultdict(list)

    for name, screen_names in publics.items():
        pickled = pickle.load(open(name, 'rb'))[:400],
        list_of_messages = pickled[0] # [1] contains nothing

        for msg in list_of_messages:
            categories_to_messages[name].append(msg)

            list_of_sentences = punkt_tokenizer.tokenize(msg)
            list_of_words_list = [regexp_tokenizer.tokenize(sentence) for sentence in list_of_sentences]

            for list_of_words in list_of_words_list:
                categories_to_words[name].extend(list_of_words)

            list_of_words_dicts = label_feats_from_corpus(list_of_words_list)

            #for dct in list_of_words_dicts:
            #    categories_to_features[name].update(dct)

            for dct in list_of_words_dicts:
                categories_to_features[name].append(dct)

    labeled_words = []

    dct = {}

    high_info_words = high_information_words(categories_to_words)

    #for category, list_of_words in categories_to_words.items():
    #    categories_to_high_info_words[category] = high_information_words(list_of_words)

    #return high_information_words(labeled_words, score_fn=score_fn)


    train, test = split_label_feats(categories_to_features)

    return train, test


    train_feats = []
    test_feats = []


def train_binary_classifiers(trainf, labelled_feats, labelset):
    pos_feats = collections.defaultdict(list)
    neg_feats = collections.defaultdict(list)
    classifiers = {}

    for feat, labels in labelled_feats:
        for label in labels:
            pos_feats[label].append(feat)

        for label in labelset - set(labels):
            neg_feats[label].append(feat)

    for label in labelset:
        postrain = [(feat, label) for feat in pos_feats[label]]
        negtrain = [(feat, '!%s' % label) for feat in neg_feats[label]]
        classifiers[label] = trainf(postrain + negtrain)

    return classifiers



class MultiBinaryClassifier(MultiClassifierI):
    def __init__(self, *label_classifiers):
        self._label_classifiers = dict(label_classifiers)
        self._labels = sorted(self._label_classifiers.keys())

    def labels(self):
        return self._labels

    def classify(self, feats):
        lbls = set()

        for label, classifier in self._label_classifiers.items():
            if classifier.classify(feats) == label:
                lbls.add(label)

        return lbls





def multi_metrics(multi_classifier, test_feats):
    mds = []
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feat, labels) in enumerate(test_feats):
        for label in labels:
            refsets[label].add(i)

        guessed = multi_classifier.classify(feat)

        for label in guessed:
            testsets[label].add(i)

        mds.append(metrics.masi_distance(set(labels), guessed))

    avg_md = sum(mds) / float(len(mds))
    precisions = {}
    recalls = {}

    for label in multi_classifier.labels():
        precisions[label] = metrics.precision(refsets[label], testsets[label])
        recalls[label] = metrics.recall(refsets[label], testsets[label])

    return precisions, recalls, avg_md




if __name__ == '__main__':

    st = str('/usr/local/bin/megam')
    import ipdb; ipdb.set_trace()
    config_megam(st)


    rwords = reuters_high_info_words()
    featdet = lambda words: bag_of_words_in_set(words, rwords)
    multi_train_feats, multi_test_feats = reuters_train_test_feats(featdet)

    trainf = lambda train_feats: MaxentClassifier.train(train_feats, algorithm='megam', trace=0, max_iter=10)
    #labelset = set(reuters.categories())
    labelset = set(list(publics.keys()))
    classifiers = train_binary_classifiers(trainf, multi_train_feats, labelset)

    len(classifiers)


    multi_classifier = MultiBinaryClassifier(*classifiers.items())

    multi_precisions, multi_recalls, avg_md = multi_metrics(multi_classifier, multi_test_feats)
    print (avg_md)

    print (multi_precisions)
    print (multi_recalls)

    ipdb.set_trace()


    cache_publics()
    train_feats, test_feats = get_feats2()





    from classification import train_binary_classifiers







    from nltk.corpus import movie_reviews

    labels = movie_reviews.categories()
    labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
    high_info_words = set(high_information_words(labeled_words))
    feat_det = lambda words: bag_of_words_in_set(words, high_info_words)
    lfeats = label_feats_from_corpus(movie_reviews, feature_detector=feat_det)
    train_feats, test_feats = split_label_feats(lfeats)


    nb_classifier = NaiveBayesClassifier.train(train_feats)
    print ('high info nb', accuracy(nb_classifier, test_feats) )
    nb_prec, nb_rec = precision_recall(nb_classifier, test_feats)

    print ('prec', nb_prec)
    print ('rec', nb_rec)

    me_class = MaxentClassifier.train(train_feats, algorithm='megam', trace=0, max_iter=10)
    print ('high info nb', accuracy(me_class, test_feats))
    nb_prec, nb_rec = precision_recall(me_class, test_feats)

    a = [(i, 1) for i in get_user_messages('4356905', 100)]
    classify_feed(a, me_classifier)

    lst = get_list_of_feed()
    classify_feed(lst, me_classifier)







    me_classifier = MaxentClassifier.train(train_feats, algorithm='megam', trace=0, max_iter=10)
    print ('me=', accuracy(me_classifier, test_feats))
    me_classifier.show_most_informative_features(n=10)


    p,r=precision_recall(me_classifier, test_feats)


# add high information words
# tweak decision trees
# voting
# multiple binary classifiers
