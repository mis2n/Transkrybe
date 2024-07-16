# Transkrybe

flask app anaconda env setup steps
___________________________________________________
install anaconda and/or - conda update conda
conda create --name flaskenv python=3.7
conda activate flaskenv
\path\to\anaconda\env\python.exe -m pip install --upgrade pip
pip install flask
pip install -r yolo5_requirements.txt
pip install tensorflow

        to test installation: python import_test.py

if import test passes and model loads, you're good to go

run server by:
                        python app.py
access app in browser at:
                        localhost:5000
