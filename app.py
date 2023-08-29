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
import cv2
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from myfunctions import *

# Allows PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Greek Character classes
classes = ['Alpha', 'Beta', 'Chi', 'Delta', 'Epsilon', 'Eta', 'Gamma', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Omega', 'Omicron', 'Phi', 'Pi', 'Psi' ,'Rho', 'LunateSigma', 'Tau', 'Theta', 'Upsilon', 'Xi', 'Zeta']


# Loading YOLO v5 model
seg = torch.hub.load('ultralytics/yolov5', 'custom', path='static/models/best_yolo.pt')

# Loading ResNet model trained on AL-ALL version 2a
infer = load_model("static/models/alv2_cxe")

# def Segment(img_path):
#     im = Image.open(img_path)
#     results = seg(im, size=640)
#     df = results.pandas().xyxy[0]
#     df.rename(columns = {'confidence':'yolo_confidence'}, inplace = True)
#     #df = df[df['yolo_confidence'] > 0.5]
#     df["width"] = abs(df["xmax"] - df["xmin"])
#     df["height"] = abs(df["ymax"] - df["ymin"])
#     df["xc"] = df["xmin"] + (df["width"] / 2)
#     df["yc"] = df["ymin"] + (df["height"] / 2)
#     df["width"] = df["width"].astype(int)
#     df["height"] = df["height"].astype(int)
#     df["xc"] = df["xc"].astype(int)
#     df["yc"] = df["yc"].astype(int)
#     df = df.drop(['name', 'class', 'xmin', 'ymin', 'xmax', 'ymax'], axis=1)
#     segdat = df.to_json("static/data/current.json")
#     segdat = df.to_csv("static/data/current.csv")
#     print("saved to Desktop")
#     return segdat

def yolo(pth):
    img_path = pth
    im = Image.open(img_path)
    results = seg(im, size=640)
    df = results.pandas().xyxy[0]
    df.rename(columns = {'confidence':'yolo_confidence'}, inplace = True)
    df = df.drop(['name', 'class'], axis=1)
    df["width"] = abs(df["xmax"] - df["xmin"])
    df["height"] = abs(df["ymax"] - df["ymin"])
    df["xc"] = df["xmin"] + (df["width"] / 2)
    df["yc"] = df["ymin"] + (df["height"] / 2)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)
    df["xc"] = df["xc"].astype(int)
    df["yc"] = df["yc"].astype(int)
    df["xmin"] = df["xmin"].astype(int)
    df["ymin"] = df["ymin"].astype(int)
    df["xmax"] = df["xmax"].astype(int)
    df["ymax"] = df["ymax"].astype(int)
    return df

def makeLines(df):
    xc = df['xc']
    yc = df['yc']
    w = df['width']
    h = df['height']
    xmin = df['xmin']
    xmax = df['xmax']
    ymin = df['ymin']
    ymax = df['ymax']
    ycon = df['yolo_confidence']
    lines, X, rad, df, nPts = SegmentLines( df )
    #print(lines)
    #print("****************")
    lines = list(reversed(lines))
    for i in range(len(lines)):
        lines[i] = list(reversed(lines[i]))
    #print(lines)
    nxc = []
    nyc = []
    nw = []
    nh = []
    nl = []
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            nxc.append(xc[lines[i][j]])
            nyc.append(yc[lines[i][j]])
            nw.append(w[lines[i][j]])
            nh.append(h[lines[i][j]])
            nl.append(i)
    df3 = pd.DataFrame(list(zip(xmin, ymin, xmax, ymax, nxc, nyc, nw, nh, nl, ycon)), columns=['xmin', 'ymin', 'xmax', 'ymax', 'xc', 'yc', 'width', 'height', 'line', 'yolo_confidence'])  
    return df3, len(lines)

def inferFrag(df):
    fragment = cv2.imread("static/data/current.jpg")
    chars = []
    conf = []
    x = []
    for i in range(len(df)):
        #cropped = fragment[(df['ymin'][i]):(df['ymax'][i]), (df['xmin'][i]):(df['xmax'][i])] ### DON'T USE THIS VERSION, X/Y MIN/MAX NOT CALCULATED CORRECTLY IN DATA FILES ###
        cropped = fragment[(int(df['yc'][i] - (df['height'][i] / 2))):(int(df['yc'][i] + (df['height'][i] / 2))), (int(df['xc'][i] - (df['width'][i] / 2))):(int(df['xc'][i] + (df['width'][i] / 2)))]
        res = cv2.resize(cropped, (70, 70), interpolation=cv2.INTER_AREA)
        x = []
        x.append(res)
        X = np.array(x)
        preds = infer.predict(X, verbose=0)
        imclass = np.argmax(preds[0])
        char = classes[imclass]
        prob = preds[0][imclass]
        chars.append(char)
        conf.append(prob)
    df['char'] = chars
    df['resnet_confidence'] = conf

    return df

def saveLinesImg(df, ll):
    tog = 0
    thickness = 1
    colorspace = [(55, 131, 255), (77, 233, 76), (255, 140, 0), (255, 238, 0), (246, 0, 0)]
    img = cv2.imread("static/data/current.jpg")[..., ::-1]
    image = img.copy()
    plt.figure()
    i = 1
    for i in range(ll):
        df2 = df[df['line'] == i]
        df2 = df2.reset_index()
        #print(df2)
        #print("******************************", len(df2))
        for j in range(len(df2)):
            # top left
            start_point = (int(df2['xc'][j] - (df2['width'][j] / 2)), int(df2['yc'][j] - (df2['height'][j] / 2)))
            # bttm right 
            end_point = (int(df2['xc'][j] + (df2['width'][j] / 2)), int(df2['yc'][j] + (df2['height'][j] / 2)))
            image = cv2.rectangle(image, start_point, end_point, colorspace[tog], thickness)
        tog += 1
        if tog >= len(colorspace):
            tog = 0
    plt.imshow(image)
    plt.savefig("static/data/yololocs_test.png")
    plt.show()

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
        #seg = Segment(full_path)
        seg = yolo(full_path)
        seg.to_csv("static/data/yolodata.csv")
        seg.to_json("static/data/yolodata.json")
        
        return render_template('segmentation.html', fid=fragid, imgtype=imgext)
        #return redirect('/segmentation.html')
    else:
        return redirect(url_for('home'))

@app.route('/linefinder/<string:userinfo>', methods=['GET', 'POST'])
def linefinder(userinfo):
    readin=userinfo
    with open('static/data/outdata.json', 'w') as f:
        f.write(readin)
        f.close()
    df =pd.read_json('static/data/outdata.json')
    df.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'xc', 'yc', 'width', 'height', 'yolo_confidence']
    df.to_csv('static/data/outdata.csv')
    df2, ll = makeLines(df)
    df2.to_csv('static/data/locs_lines.csv')
    df2.to_json('static/data/locs_lines.json')
    ndf = inferFrag(df2)
    ndf.to_csv("static/data/locs_lines_chars.csv")
    ndf.to_json("static/data/locs_lines_chars.json", orient='records')
    saveLinesImg(df2, ll)
    return render_template('linefinder.html')



@app.route('/showlines', methods=['GET', 'POST'])
def inference():
    return render_template('showlines.html')


if __name__=='__main__':
    app.run(debug = True)
