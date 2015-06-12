import goslate
import os
from matplotlib import pyplot,pylab as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
from textblob import TextBlob
import numpy as np
import cv2
import Image
import scipy;


senti = Flask(__name__)
senti.config['UPLOAD_FOLDER'] = 'uploads/'
senti.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg','png','JPG'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in senti.config['ALLOWED_EXTENSIONS']

@senti.route('/')
@senti.route('/index')
def index():
    return render_template('index.html')

@senti.route('/multi', methods=['POST'])
def multi():
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            filename = file.filename
            print(file.filename)
            file.save(os.path.join(senti.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
            
    file1 = uploaded_files[0]
    file2 = uploaded_files[1]
    
    img_rgb = cv2.imread('uploads/'+file1.filename)
    print('uploads/'+file1.filename)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('uploads/'+file2.filename,0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
      cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


    im = Image.fromarray(img_rgb)
    im.save('uploads/tm'+filename)
    
    print(type(img_rgb))
    
    return redirect(url_for('uploaded_file',filename="tm"+filename))


@senti.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
	filename = file.filename
        file.save(os.path.join(senti.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread('uploads/'+filename)

        if 'Denoise Image' in request.form['submit']:
		dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
		im = Image.fromarray(dst)
		im.save('uploads/d'+filename)
       	 	return redirect(url_for('uploaded_file',filename="d"+filename))
	elif 'Detect Face' in request.form['submit']:
		faceCascade = cv2.CascadeClassifier('static/xml/haarcascade_frontalface_alt.xml')
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		im = Image.fromarray(img)
		im.save('uploads/face'+filename)
		return redirect(url_for('uploaded_file',filename="face"+filename))
	elif 'Smoothen Image' in request.form['submit']:
		kernel = np.ones((5,5),np.float32)/25
		dst = cv2.filter2D(img,-1,kernel)
		im = Image.fromarray(dst)
		im.save('uploads/smooth'+filename)
		return redirect(url_for('uploaded_file',filename="smooth"+filename))
	elif 'Detect Blue Colour' in request.form['submit']:
		# Convert BGR to HSV
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# define range of blue color in HSV
		lower_blue = np.array([110,50,50])
		upper_blue = np.array([130,255,255])
		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		# Bitwise-AND mask and original image\
		res = cv2.bitwise_and(img,img, mask= mask)
		im = Image.fromarray(res)
		im.save('uploads/blue'+filename)
		return redirect(url_for('uploaded_file',filename="blue"+filename))
	elif 'Image Resize' in request.form['submit']:
	   	small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
	   	resized_image = cv2.resize(image, (100, 50)) 
      		small = scipy.misc.imresize(image, 0.5)
      		print(type(small))
      		return ''
@senti.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(senti.config['UPLOAD_FOLDER'],
                               filename)

@senti.route('/api/v1/sentiment/<message>')
def sentiment(message):
    text = TextBlob(message)
    response = {'polarity' : text.polarity , 'subjectivity' : text.subjectivity}
    return jsonify(response)

@senti.route('/translate')
def translate():
    textVal = request.args.get('textVal', 0, type=str)
    langVal = request.args.get('langVal', 0, type=str)
    gs = goslate.Goslate()
    return jsonify(result=gs.translate(textVal,langVal))

if __name__ == "__main__":
    senti.run(debug=True)
