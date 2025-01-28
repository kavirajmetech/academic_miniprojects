from flask import Flask, request, render_template
import joblib

# Load the model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        vectorized_message = vectorizer.transform([message])
        prediction = model.predict(vectorized_message)[0]
        result = "Spam" if prediction == 1 else "Ham"
        return render_template('index.html', prediction=result, user_input=message)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

