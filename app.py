from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: 'healthy-Spiral', 
1: 'parkinson-Spiral'
           }
verboses_names = {
0: 'healthy-wave', 
1: 'parkinson-wave'
           }


model = load_model('spiral.h5')
models = load_model('wave.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]
    
def predicts_labels(img_path):
	test_image = image.load_img(img_path, target_size=(196,196))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 196,196,3)

	predict_x=models.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verboses_names[classes_x[0]]
 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)
    
@app.route("/indexs", methods=['GET', 'POST'])
def indexs():
	return render_template("indexs.html")


@app.route("/submits", methods = ['GET', 'POST'])
def gets_outputs():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predicts_labels(img_path)

	return render_template("predictions.html", prediction = predict_result, img_path = img_path)    
    

@app.route("/performance")
def performance():
	return render_template('performance.html')
@app.route("/performances")
def performances():
	return render_template('performances.html')    

@app.route("/chart")
def chart():
	return render_template('chart.html') 

	
if __name__ =='__main__':
	app.run(debug = True)


	

	


