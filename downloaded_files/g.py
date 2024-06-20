from utils import *
from global_data import *

from sklearn.pipeline import Pipeline

logging.info("Problem g")

method_arr = [TruncatedSVD(n_components=50, n_iter=10, random_state=17),
              NMF(n_components=50, random_state=17)]
method_name = ["LSI", "NMF"]

for ai, method in enumerate(method_arr):
    pipeline_g = Pipeline([
        ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', method),
        ('toarr', SparseToDenseArray()),
        ('clf', GaussianNB()),
    ])
    pipeline_g.fit(train_data.data, train_label)
    pred_test = pipeline_g.predict(test_data.data)
    pred_test_prob = pipeline_g.predict_proba(test_data.data)[:, 1]
    print("-" * 70)
    print("Using " + method_name[ai] +
          " and Gaussian Naive Bayes")
    analyze(test_label, pred_test_prob, pred_test, CAT, 2)

pipeline_g2 = Pipeline([
    ('vect', CountVectorizer(min_df=MIN_DF, stop_words=ENGLISH_STOP_WORDS, tokenizer=stem_and_tokenize)),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', method_arr[1]),
    ('clf', MultinomialNB()),
])
pipeline_g2.fit(train_data.data, train_label)
pred_test = pipeline_g2.predict(test_data.data)
pred_test_prob = pipeline_g2.predict_proba(test_data.data)[:, 1]
print("-" * 70)
print("Using NVM and Multinomial Naive Bayes")
analyze(test_label, pred_test_prob, pred_test, CAT, 2)

logging.info("finished Problem g")
