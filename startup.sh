#!/bin/bash

export FLASK_APP=app
export FLASK_ENV=development
export FLASK_DEBUG=1
flask run &
cmd.exe /C start http://localhost:5000
