import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,make_response
from textblob import TextBlob
import numpy as np
import cv2
import Image
from gridfs import GridFS  
from pymongo import MongoClient  
import json
__author__ = 'AboorvaDevarajan'
# Initializing the Mongo Database.
mongo_client = MongoClient('mongodb://localhost:27017/') 
db = mongo_client['TestDB'] 
grid_fs = GridFS(db)

imageProcessServer = Flask(__name__)
imageProcessServer.config['UPLOAD_FOLDER'] = 'uploads/'
imageProcessServer.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in imageProcessServer.config['ALLOWED_EXTENSIONS']

@imageProcessServer.route('/')
@imageProcessServer.route('/index')
def index():
    return render_template('index.html')

@imageProcessServer.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(imageProcessServer.config['UPLOAD_FOLDER'], filename))

	with grid_fs.new_file(filename=filename) as fp:
        	fp.write(file)
        	file_id = fp._id
    	if grid_fs.find({"_id":file_id}) is not None:
        	print(url_for('uploaded_file',filename="d"+filename))
		return redirect(url_for('uploaded_file',filename="d"+filename))
    	else:
        	return json.dumps({'status': 'Error occurred while saving file.'}), 500

	img = cv2.imread('uploads/'+filename)
	dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	im = Image.fromarray(dst)
	im.save('uploads/d_'+filename)

@imageProcessServer.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(imageProcessServer.config['UPLOAD_FOLDER'],
                               filename)
if __name__ == "__main__":
    imageProcessServer.run(debug=True)
