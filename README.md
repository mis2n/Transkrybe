# Transkrybe

This repository hosts a prototype web application for AI-assisted transcription of ancient Greek papyri. The software, written in Python and JavaScript, utilizes custom models to locate and classify Greek characters on 
highly damaged papyrus fragments. The tool is designed to assist professional papyrologist in their important work. The software utilizes the Flask library as well as Fabric.js for the creation of the interactive 
user interface. This project is a work in progress and is highly experimental. known bugs exist, primarily surrounding synchronous/asynchronous tasks. Future iterations of this software will likely incorporate 
additional model and tools with the goal of creating an all-in-one suite of software tools for professional papyrology.  

<h3>flask app anaconda env setup steps</h3>h3
___________________________________________________
<br>
<ul>
<li>install anaconda and/or - 'conda update conda'</ul>
<li>conda create --name flaskenv python=3.7</ul>
<li>conda activate flaskenv</ul>
<li>\path\to\anaconda\env\python.exe -m pip install --upgrade pip</ul>
<li>pip install flask</ul>
<li>pip install -r yolo5_requirements.txt</ul>
<li>pip install tensorflow</ul>
<ul><br>
        to test installation, in the repo directory run: python import_test.py
<br>
if import test passes and model loads, you're good to go
<br>
Initialize the Flask server by running:<br>
                        python app.py
<br>
You can then access the app in your local browser at:<br>
                        localhost:5000
