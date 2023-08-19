# fix windows registry stuff
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

from flask import Flask, render_template, request, redirect, url_for, flash
import json
import os
import os.path
from werkzeug.utils import secure_filename
import time

import torch
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Allows PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Greek Character classes
classes = ['Alpha', 'Beta', 'Chi', 'Delta', 'Epsilon', 'Eta', 'Gamma', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Omega', 'Omicron', 'Phi', 'Pi', 'Psi' ,'Rho', 'LunateSigma', 'Tau', 'Theta', 'Upsilon', 'Xi', 'Zeta']


# Loading YOLO v5 model
seg = torch.hub.load('ultralytics/yolov5', 'custom', path='static/models/best_yolo.pt')

# Loading ResNet model trained on AL-ALL version 2a
infer = load_model("static/models/alv2_cxe")

def Segment(img_path):
    im = Image.open(img_path)
    results = seg(im, size=640)
    df = results.pandas().xyxy[0]
    df.rename(columns = {'confidence':'yolo_confidence'}, inplace = True)
    #df = df[df['yolo_confidence'] > 0.5]
    df["width"] = abs(df["xmax"] - df["xmin"])
    df["height"] = abs(df["ymax"] - df["ymin"])
    df["xc"] = df["xmin"] + (df["width"] / 2)
    df["yc"] = df["ymin"] + (df["height"] / 2)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)
    df["xc"] = df["xc"].astype(int)
    df["yc"] = df["yc"].astype(int)
    df = df.drop(['name', 'class', 'xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    segdat = df.to_json("static/data/current.json")
    segdat = df.to_csv("static/data/current.csv")
    print("saved to Desktop")
    return segdat

app = Flask(__name__)
app.secret_key = '1234567890'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/segmentation', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        dt = time.strftime('%Y%m%d_%H%M%S_')
        full_name = dt + secure_filename(f.filename)
        fragid = full_name[:-4]
        imgext = full_name[-4:]
        full_path = "static/data/current" + imgext

        f.save(full_path)
        seg = Segment(full_path)
        
        return render_template('segmentation.html', fid=fragid, imgtype=imgext)
        #return redirect('/segmentation.html')
    else:
        return redirect(url_for('home'))





@app.route('/linefinder/<string:userinfo>', methods=['GET', 'POST'])
def linefinder(userinfo):
    readin=userinfo
    with open('static/outdata.json', 'w') as f:
        f.write(readin)
        f.close()
    
    return render_template('/linefinder.html')



@app.route('/inference', methods=['GET', 'POST'])
def inference():
    return render_template('inference.html')


if __name__=='__main__':
    app.run(debug = True)
