from flask import Flask, render_template, request, redirect, url_for, flash
import json
import os.path
from werkzeug.utils import secure_filename
import datetime
import re

import torch
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt

# Allows PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("Loading model...")
#seg = torch.hub.load('ultralytics/yolov5', 'custom', path='/static/models/best_yolo.pt')
print("Model loaded.")

def Segment(img_path):
    im = Image.open(img_path)
    results = seg(im, size=640)
    df = results.pandas().xyxy[0]
    df.rename(columns = {'confidence':'yolo_confidence'}, inplace = True)
    df = df[df['yolo_confidence'] > 0.5]
    df["width"] = abs(df["xmax"] - df["xmin"])
    df["height"] = abs(df["ymax"] - df["ymin"])
    df["xc"] = df["xmin"] + (df["width"] / 2)
    df["yc"] = df["ymin"] + (df["height"] / 2)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)
    df["xc"] = df["xc"].astype(int)
    df["yc"] = df["yc"].astype(int)
    df = df.drop(['name', 'class', 'xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    segdat = df.to_json()
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
        # dt = str(datetime.datetime.now())
        # dt = dt.split(".")[0]
        # dt = re.sub('[^a-zA-Z0-9\n\.]', '_', dt)
        dt = "TODAY"
        full_name = dt + '_' + secure_filename(f.filename)
        full_path = '/mnt/c/Users/matth/Desktop/new_app/static/uploads/' + full_name
        rel_path = 'static/uploads/' + full_name
        segs = 'temporary placeholder for segmentation data from yolo'
        f.save(full_path)

        #segs = Segment(full_path)
        segs = 'temporary placeholder for segmentation data from yolo'

        return render_template('segmentation.html', relpath=rel_path, ydata=segs)
    else:
        return redirect(url_for('home'))

if __name__=='__main__':
    app.run(debug = True)
