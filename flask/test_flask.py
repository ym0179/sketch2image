import flask
from flask import Flask, request, render_template, jsonify, url_for
from tensorflow.keras.models import load_model
import numpy as np
import imageio
import sys
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
import matplotlib.pyplot as plt
import cv2
import os

app=Flask(__name__, static_url_path="/static", static_folder="static")

# 메인페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

# 데이터 예측 처리
@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method=='POST' :

        file=request.files['img']
        if not file : return render_template('index.html')
        
        # sketch=request.files['img']
        # sketch.save("./flask_project/static/in.png")

        img=imageio.imread(file)
        img=cv2.resize(img, (256, 256))
        img=img_to_array(img)
        img = (img - 127.5) / 127.5
        img = expand_dims(img, 0)
        print(img.shape)

        vgg16=load_model('./flask_project/bear_strawberry_softmax.h5')
        strawberry=format(vgg16.predict(img)[0][0] * 100, '.1f')
        teddybear=format(vgg16.predict(img)[0][1] * 100, '.1f')


        model = load_model('./flask_project/model_031200.h5')
        predict=model.predict(img)

        predict = (predict + 1) / 2.0
        plt.imshow(predict[0])
        plt.axis('off')
        plt.savefig('./flask_project/static/out.png')
        return render_template("index.html", fake_img='out.png', name=strawberry, label=teddybear)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['v'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

if __name__ == '__main__' :
    app.run(host="127.0.0.1", port="8080")