# Sample Script to use our trained model in a real-time Environment
# download weights form below link and place the weights in the directory of the this python file before running this scripts
# https://drive.google.com/drive/folders/1YB4Ya5IgwgigS7wvbNXDox0f4wUjGLTf?usp=sharing

import pickle
from sklearn.ensemble import RandomForestClassifier

loaded_model = pickle.load(open("RandomForestClassifier_WineVariety", 'rb'))
tf_idf = pickle.load(open("tfidf_Vectorizer_wineRatingDescription", 'rb'))

while True:
    s = input().strip()
    X = tf_idf.transform([s])
    prediction = loaded_model.predict(X)
    print(prediction)