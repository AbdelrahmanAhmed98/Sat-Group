from flask import Flask,render_template,request
import joblib
from helpers.dummies import *

app=Flask(__name__)

model=joblib.load('Mo_model.h5')
scaler=joblib.load('Mo_scaler.h5')

@app.route('/',methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/predict',methods=['Get'])
def predict():
	all_data=request.args
	
	Country=all_data['Country']
	Month=all_data['Month']

	data=[Country,Month]
	final_data=scaler.transform([data])
	pred=model.predict(final_data)
	return str(pred)
	




if __name__=='__main__':
	app.run()