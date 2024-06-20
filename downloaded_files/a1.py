'''
Sebastian Ballesteros
'''

'''
'''

############################## IMPORTS  ########################################

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize

from matplotlib import pyplot as plt
import math
import pandas
import codecs
import numpy as np
import string
import sys
import argparse

########################### GLOBAL VARIABLES ##################################

lemmatize = False
smoothing = 1
stem = False
min_frequency = 2
train_percentage = 0.8
test_percentage = 0.2
cross_val_splits = 10
verbose = False

########################### GLOBAL CONSTANTS ##################################

SENTENCES_COUNT = 10662
NUM_MODELS = 5
DATA_DIRECTORY = r'./rt-polaritydata'

###########################  HELPER FUNCTIONS ##################################

def read_file(filename):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as file:
             return file.readlines()

def tokenize(sentence):
    sentence_without_punctuation = (''.join([word for word in sentence if word not in string.punctuation]))
    sentence_tokenized = nltk.word_tokenize(sentence_without_punctuation)
    return sentence_tokenized

def stem(sentences):
    stemmer = Porterstemmer()

    sentences_stemmed = []
    for sentence in sentences:
        sentences_stemmed.append(''.join([stemmer.stem(word) for word in sentence]))
    return sentences_stemmed

def lemmatize(sentences):
    lemmatizer = WordNetlemmatizer()

    sentences_lemmatized = []
    for sentence in sentences:
        sentences_lemmatized.append(''.join([lemmatizer.lemmatize(word) for word in sentence]))
    return sentences_lemmatized

def remove_stop_words(tokenized_sentencees):
    stop_words = stopwords.words('english')

    new_tokenized_sentences = []
    for sentence in tokenized_sentencees:
        without_stop_words = [word for word in sentence if word not in stop_words]
        new_tokenized_sentences.append(without_stop_words)
    return new_tokenized_sentences

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', action='store_true', dest='lemmatize', default=False,
                    help='Lemmatize in preprocess')

    parser.add_argument('-s', action='store_true', dest='stem', default=False,
                    help='Stem in preprocess')

    parser.add_argument('-tr', action='store', default=0.8,
                    dest='train_percentage',
                    help='Set the train percentage')

    parser.add_argument('-te', action='store', default=0.2,
                        dest='test_percentage',
                        help='Set the train percentage')

    parser.add_argument('-sm', action='store', default=1,
                    dest='smoothing',
                    help='Set the smoothing value')

    parser.add_argument('-minf', action='store', default=1,
                    dest='min_f',
                    help='Set the minimum frequency for features')

    parser.add_argument('-cross', action='store', default=10,
                    dest='cross_val',
                    help='Set the cross validation split iterations')

    parser.add_argument('-v', action='store_true', dest='verbose', default=False,
                        help='Print the confusion matrix of each iteration')

    results = parser.parse_args()
    global lemmatize
    lemmatize = results.lemmatize
    global smoothing
    smoothing  = results.smoothing
    global stem
    stem = results.stem
    global min_frequency
    min_frequency = results.min_f
    global train_percentage
    train_percentage = results.train_percentage
    global test_percentage
    test_percentage = results.test_percentage
    global cross_val_splits
    cross_val_splits = results.cross_val
    global verbose
    verbose = results.verbose

###########################  MAIN FUNCTIONS ##################################

def load_data():
    negative_sentences = read_file('./rt-polaritydata/rt-polarity.neg')
    positive_sentences = read_file('./rt-polaritydata/rt-polarity.pos')

    y_negative = np.zeros((len(negative_sentences), 1))
    y_positive = np.ones((len(positive_sentences), 1))

    y = np.concatenate((y_negative, y_positive))
    x = np.concatenate((negative_sentences, positive_sentences))

    return(x, y)


def preprocess(x, y, train_index, test_index):

    #lemmatize or stem
    if(lemmatize):
        x = lemmatize(x)
    elif(stem):
        x = stem(x)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #extract features and remove infrequently occurring words
    tokenizer = CountVectorizer(min_df=min_frequency, tokenizer=nltk.word_tokenize, stop_words='english')
    x_counts = tokenizer.fit_transform(x_train)

    # Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
    fooTfmer = TfidfTransformer()
    x_train_vec = fooTfmer.fit_transform(x_counts)

    #print(x)

    x_test_vec = tokenizer.transform(x_test)

    return x_train_vec, x_test_vec, y_train, y_test


def train_naive_bayes(x_train, y_train, x_test):
    naive_bayes = MultinomialNB(alpha=smoothing)

    naive_bayes.fit(x_train, y_train.ravel())

    naive_bayes.score(x_train, y_train)

    y_pred = naive_bayes.predict(x_test)

    return y_pred

def test_naive_bayes(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    if(verbose):
        print('-----------Naive Bayes-------------')
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_logistic_regression(x_train, y_train, x_test):
    logistic_regression = LogisticRegression()

    logistic_regression.fit(x_train, y_train.ravel())

    logistic_regression.score(x_train, y_train)

    y_pred = logistic_regression.predict(x_test)

    return y_pred

def test_logistic_regression(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    if(verbose):
        print('-------Logistic Regression--------')
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_support_vector(x_train, y_train, x_test):
    support_vector = LinearSVC()

    support_vector.fit(x_train, y_train.ravel())

    support_vector.score(x_train, y_train)

    y_pred = support_vector.predict(x_test)

    return y_pred

def test_support_vector(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    if(verbose):
        print('-----Support Vector Machines------')
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_bernoulli(x_train, y_train, x_test):
    bernoulli = BernoulliNB()

    bernoulli.fit(x_train, y_train.ravel())

    bernoulli.score(x_train, y_train)

    y_pred = bernoulli.predict(x_test)

    return y_pred

def test_bernoulli(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    if(verbose):
        print('-------------Bernoulli------------')
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def train_random():
    y_pred = np.random.randint(2, size=(math.ceil(SENTENCES_COUNT * test_percentage)))
    return y_pred

def test_random(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    if(verbose):
        print('---------------Random-------------')
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Confusion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    return accuracy

def plot_results(models_accuracy):
    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(16,9))

    x1 = np.arange(len(models_accuracy))

    ax1.bar(x1, [x[1] for x in models_accuracy])

    # Place values on top of bars
    for i, v in enumerate(models_accuracy):
        #(x, y, text)

        ax1.text(i - 0.25, 40, v[0])
        ax1.text(i - 0.25, v[1] + 2, str(v[1]) + '%')

    ax1.set_ylabel('Average accuracy (%)')
    ax1.set_title('Models ran in {} iteration'.format(cross_val_splits))
    ax1.set_ylim([0, 100])

    plt.show()

def main():
    x, y = load_data()

    #Run cross validation
    cross_val = ShuffleSplit(n_splits=cross_val_splits, test_size=test_percentage)

    naive_bayes_accuracy = []
    lregression_accuracy = []
    svector_accuracy = []
    bernoulli_accuracy = []
    random_accuracy = []

    #Preprocess, train and test for each model with all the cross-val splits
    for train_index, test_index in cross_val.split(x):

        x_train, x_test, y_train, y_test = preprocess(x, y, train_index, test_index)

        y_pred = train_naive_bayes(x_train, y_train, x_test)
        accuracy =test_naive_bayes(y_test, y_pred)
        naive_bayes_accuracy.append(accuracy)

        y_pred = train_logistic_regression(x_train, y_train, x_test)
        accuracy = test_logistic_regression(y_test, y_pred)
        lregression_accuracy.append(accuracy)

        y_pred = train_support_vector(x_train, y_train, x_test)
        accuracy = test_support_vector(y_test, y_pred)
        svector_accuracy.append(accuracy)

        y_pred = train_bernoulli(x_train, y_train, x_test)
        accuracy = test_bernoulli(y_test, y_pred)
        bernoulli_accuracy.append(accuracy)

        y_pred = train_random()
        accuracy = test_random(y_test, y_pred)
        random_accuracy.append(accuracy)

    # Compute average of all runs
    models_accuracy = []
    models_accuracy.append(('Naive Bayes', sum(naive_bayes_accuracy)/len(naive_bayes_accuracy)))
    models_accuracy.append(('Logistic Regression', sum(lregression_accuracy)/len(lregression_accuracy)))
    models_accuracy.append(('Support Vector', sum(svector_accuracy)/len(svector_accuracy)))
    models_accuracy.append(('Bernoulli', sum(bernoulli_accuracy)/len(bernoulli_accuracy)))
    models_accuracy.append(('Random', sum(random_accuracy)/len(random_accuracy)))

    #Plot the results of the average of each model
    plot_results(models_accuracy)

if __name__ == "__main__":
    parse_args()
    main()
