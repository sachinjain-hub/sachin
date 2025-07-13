from flask import Flask, render_template, request,jsonify
import pickle
import os
import numpy as np
os.environ.get("PORT", 5000)
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    return jsonify({'predicted_price': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
