from flask import Flask, jsonify, render_template, url_for
import xgboost as xgb
from keras.models import load_model
import joblib
import pickle
import sklearn
import numpy as np
from flask import request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('home.html', name='Trung', date='11/01/2022')

@app.route("/predict", methods=['POST'])
def predict():
    #load lib scaler & model
    XScaler = joblib.load('XScaler.gz')
    yScaler = joblib.load('yScaler.gz')
    modelNN = load_model('modelNN.h5')

    #input
    input = request.get_json()['input']
    
    #scaler input
    inputScale = XScaler.transform([input])
    
    predict = modelNN.predict(inputScale)
    
    price = yScaler.inverse_transform(predict)[0][0]
    print(price)
    return jsonify({
        'code': 0,
        'price': str(price)
    })

# @app.route("/tes", methods=['GET'])
# def tes():
#     #load lib scaler & model
#     mdxg = xgb.Booster().load_model("xg.txt")
#     ip = [5,368.7,13.0,28.0,11.0,7.0,5.0,2]
#     print(mdxg)
#     return jsonify({'code': 0})

if __name__ == "__main__":
    app.run()