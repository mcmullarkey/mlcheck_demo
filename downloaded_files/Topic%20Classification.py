import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Set random seed
np.random.seed(500)

# Read data using pandas
Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\NLP Project\corpus_small.csv", encoding='latin-1')

# Preprocessing: Remove blank rows, change to lowercase, tokenize
Corpus.dropna(subset=['text'], inplace=True)
Corpus['text'] = Corpus['text'].str.lower()
Corpus['text'] = Corpus['text'].apply(nltk.word_tokenize)

# Preprocessing: Remove stop words, perform word lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()
tag_map = defaultdict(lambda: nltk.corpus.wordnet.NOUN)
tag_map['J'] = nltk.corpus.wordnet.ADJ
tag_map['V'] = nltk.corpus.wordnet.VERB
tag_map['R'] = nltk.corpus.wordnet.ADV
stop_words = set(nltk.corpus.stopwords.words('english'))

def process_text(text):
    final_words = []
    for word, tag in nltk.pos_tag(text):
        if word.isalpha() and word not in stop_words:
            word_final = lemmatizer.lemmatize(word, tag_map[tag[0]])
            final_words.append(word_final)
    return ' '.join(final_words)

Corpus['text_final'] = Corpus['text'].apply(process_text)

# Split the data into training and test sets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['label'], test_size=0.3)

# Label encode the target variable
Encoder = model_selection.LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.transform(Test_Y)

# Vectorize the words using TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Train and test the classifiers
models = [
    ('Naive Bayes', naive_bayes.MultinomialNB()),
    ('SVM', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
]

for name, model in models:
    model.fit(Train_X_Tfidf, Train_Y)
    predictions = model.predict(Test_X_Tfidf)
    accuracy = accuracy_score(predictions, Test_Y) * 100
    print(f"{name} accuracy score: {accuracy:.2f}%")
