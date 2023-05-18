import cv2
import os
import gc
import torch
import time
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
from threading import Thread
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, session, redirect
from werkzeug.utils import secure_filename

print("Import test Passed!")

print("loading model . . . .")
seg = torch.hub.load('ultralytics/yolov5', 'custom', path='static/models/best_yolo.pt')#, force_reload=True)

print("Model loaded.")
