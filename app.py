from flask import Flask, render_template, request, redirect, url_for, flash
import json
import os.path
from werkzeug.utils import secure_filename
import datetime
import re

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

        return render_template('segmentation.html', relpath=rel_path, ydata=segs)
    else:
        return redirect(url_for('home'))