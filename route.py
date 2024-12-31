from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import datetime
import os

app = Flask(__name__)

# Debug statement to check template search path
print("Template search paths:", app.jinja_loader.searchpath)

# Loading model
model_path = os.getcwd() + r'/model'
try:
    classifier = joblib.load(model_path + r'/classifier.pkl')
except KeyError as e:
    print(f"KeyError while loading the model: {e}")
    print("Ensure joblib and scikit-learn versions match.")
    raise
except FileNotFoundError:
    print("Model file not found. Please check the path.")
    raise
except Exception as e:
    print(f"An error occurred: {e}")
    raise

def predictfunc(review):    
    prediction = classifier.predict(review)
    if prediction[0] == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'      
    return prediction[0], sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        result = request.form
        content = request.form['review']
        print(f"Review received: {content}")
        review = pd.Series(content)
        prediction, sentiment = predictfunc(review)      
    return render_template("predict.html", pred=prediction, sent=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
