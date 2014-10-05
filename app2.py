import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

from app import get_corpus_for_sp
import vk


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."



class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')




###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)

    from app import get_friends, get_groups, get_group_messages

    friends_ids = get_friends(790781)
    for friend_id in friends_ids:
        most_common = []
        try:
            group_ids = get_groups(friend_id)
        except vk.VkAPIMethodError:
            continue

        for group_id in group_ids:
            msgs = get_group_messages(group_id, limit=100)

            vv = vectorizer.transform(msgs)
            predicted = clf.predict(vv)
            from collections import Counter
            res = Counter(predicted)
            #print ('user_id=', friend_id, ' group_id=', group_id, ' counter=', res)
            mc = res.most_common()
            try:
                first_mc = mc[0][0]
                most_common.append(first_mc)
                most_common.append(first_mc)
            except IndexError:
                pass

            try:
                second_mc = mc[1][0]
                most_common.append(second_mc)
            except IndexError:
                pass
        ccc = Counter(most_common)
        import ipdb; ipdb.set_trace()

        print ('user_id=', friend_id, ' counter=', ccc)




    """
    from app import get_user_messages
    lst = [(i, 1) for i in get_user_messages('4356905', 500)]
    lst = [i[0] for i in lst]

    vv = vectorizer.transform(lst)
    predicted = clf.predict(vv)

    for i in range(len(predicted)):
        print (lst[i])
        print ('>>>>>>>>>>>>>>>class ', predicted[i])

    import ipdb; ipdb.set_trace()
    a = 1

    # Feed analyze start
    from app import get_list_of_feed
    lst = get_list_of_feed()
    lst = [i[0] for i in lst]

    vv = vectorizer.transform(lst)
    predicted = clf.predict(vv)

    for i in range(len(predicted)):
        print (lst[i])
        print ('>>>>>>>>>>>>>>>class ', predicted[i])
    """


    # Feed analyze end






    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    #import ipdb; ipdb.set_trace()
    print ('PRECISION ', metrics.precision_score(y_test, pred))
    print ('ACCURACY ', metrics.accuracy_score(y_test, pred))

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time






if __name__ == '__main__':
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--report",
                  action="store_true", dest="print_report",
                  help="Print a detailed classification report.")
    op.add_option("--chi2_select",
                  action="store", type="int", dest="select_chi2",
                  help="Select some number of features using a chi-squared test")
    op.add_option("--confusion_matrix",
                  action="store_true", dest="print_cm",
                  help="Print the confusion matrix.")
    op.add_option("--top10",
                  action="store_true", dest="print_top10",
                  help="Print ten most discriminative terms per class"
                       " for every classifier.")
    op.add_option("--all_categories",
                  action="store_true", dest="all_categories",
                  help="Whether to use all categories or not.")
    op.add_option("--use_hashing",
                  action="store_true",
                  help="Use a hashing vectorizer.")
    op.add_option("--n_features",
                  action="store", type=int, default=2 ** 16,
                  help="n_features when using the hashing vectorizer.")
    op.add_option("--filtered",
                  action="store_true",
                  help="Remove newsgroup information that is easily overfit: "
                       "headers, signatures, and quoting.")

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    #print(__doc__)
    #op.print_help()
    #print()





    res = get_corpus_for_sp()

    pairs = []
    for category, posts_list in res.items():
        for post in posts_list:
            pairs.append((category, post))

    import random
    random.shuffle(pairs)

    data, target = [], []
    for category, post in pairs:
        target.append(category)
        data.append(post)


    SPLIT_PERC = 0.75
    split_size = int(len(data) * SPLIT_PERC)
    train_data = data[:split_size]
    test_data = data[split_size:]
    train_categories = target[:split_size]
    test_categories = target[split_size:]


    data_train_size_mb = size_mb(train_data)
    data_test_size_mb = size_mb(test_data)

    #import ipdb; ipdb.set_trace()
    """
    print("%d documents - %0.3fMB (training set)" % (
        len(train_data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(test_data), data_test_size_mb))
    """
    #print("%d categories" % len(categories))
    #print()

    # split a training set and a test set
    y_train, y_test = train_categories, test_categories






    import ipdb; ipdb.set_trace()
    print("Extracting features from the training dataset using a sparse vectorizer")

    t0 = time()
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=opts.n_features)
        X_train = vectorizer.transform(train_data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(train_data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(test_data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        print("done in %fs" % (time() - t0))
        print()


    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())

    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                                dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty)))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet")))

    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))



    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    results.append(benchmark(L1LinearSVC()))


    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()




# Feed analyze start
def get_analyzed_feed():
    from app import get_list_of_feed
    lst = get_list_of_feed()
    lst = [i[0] for i in lst]

    clf = MultinomialNB(alpha=.01)

    #print('_' * 80)
    #print("Training: ")
    #print(clf)
    t0 = time()


    res = get_corpus_for_sp()

    pairs = []
    for category, posts_list in res.items():
        for post in posts_list:
            pairs.append((category, post))

    import random
    random.shuffle(pairs)


    data, target = [], []
    for category, post in pairs:
        target.append(category)
        data.append(post)


    SPLIT_PERC = 0.75
    split_size = int(len(data) * SPLIT_PERC)
    train_data = data[:split_size]
    test_data = data[split_size:]
    train_categories = target[:split_size]
    test_categories = target[split_size:]

    y_train, y_test = train_categories, test_categories


    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
    X_train = vectorizer.fit_transform(train_data)


    clf.fit(X_train, y_train)
    train_time = time() - t0
    #print("train time: %0.3fs" % train_time)

    t0 = time()
    X_test = vectorizer.transform(test_data)
    pred = clf.predict(X_test)

    from app import get_friends, get_groups, get_group_messages

    vv = vectorizer.transform(lst)
    predicted = clf.predict(vv)

    for i in range(len(predicted)):
        txt = lst[i]
        class_ = predicted[i]
        yield (txt, class_)
