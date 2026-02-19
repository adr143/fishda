#!/bin/bash

# Navigate to working directory
cd /home/admin/Desktop/Tilapia_Web || exit

# Activate virtual environment
source venv/bin/activate

# Run the Flask app
python app.py >> flask.log 2>&1
