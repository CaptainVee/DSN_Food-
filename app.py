from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from PIL import Image
import os
import sys
from good import *


app = Flask(__name__)
#UPLOAD_FOLDER = '/Users/CAPTIAN VEE/Mask_RCNN/uploads'
#DETECTION_FOLDER = '/home/shubham-sakha/project/Repo/custom_object/FLask/static/detections'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      # create a secure filename
      filename = secure_filename(f.filename)
      print(filename)
      # save file to /static/uploads
      #filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      basepath = os.path.dirname(__file__)
      file_path = os.path.join(
      basepath, 'uploads', filename)
      f.save(file_path)
      print(file_path)
 
      classes, pic_names = predict(file_path, filename)
      if classes == None:
        return render_template("about.html", file=filename, pic=pic_names)
      else:
        return render_template("uploaded.html", file=filename, my_classes=classes, pic=pic_names)     


if __name__ == '__main__':
   app.run(debug=True)

#if __name__ == '__main__':
   #app.run(port='0.0.0.0', debug=8080)
