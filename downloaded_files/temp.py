from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
# Load the Random Forest CLassifier model
filename = 'model_cyber_bullying.sav'
model = joblib.load(open(filename, 'rb'))
tfidf_vectorizer = joblib.load('vectorizer_cyber_bullying.sav')
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route(('/home'))
def home():   
    print("hello") 
    return render_template("index.html")


@app.route(("/equ"),methods = ['POST','GET'])
def result():   
    if request.method == 'POST':
        # output = request.form
        # name = int(output["name"])
        arg0 = request.form["arg0"]
        # if isinstance(arg0, list):
        #     arg0 = arg0[0]

        # data = [[arg0]]
        arg0_vectorized = tfidf_vectorizer.transform([arg0])

        # Predict using the model
        my_prediction = model.predict(arg0_vectorized)
        
        print("\n Testing the phrase: ", arg0)
        # v1=vectorizer.transform([arg0])
        # text_to_predict = np.array(arg0)

        # Preprocess the text using the vectorizer
        # text_to_predict = vectorizer.transform([text_to_predict])

        # Predict the class label
        # ans = list(clf.predict(v1))[0]



        # print("\n Testing the phrase: ", 'i am a normal girl')
        # v1=vectorizer.transform(['i am a normal girl'])

        # print("Is it considered bullying?",ans)

        filename = 'vectorizer_cyber_bullying.sav'
        # joblib.dump(tfidf_vectorizer, filename)
        # arg2 = int(output["arg2"])
        # ans = arg0
        # anss = str(ans)
    return render_template("index.html",name = my_prediction[0])   

if __name__ == "__main__":
    app.run(debug=True)