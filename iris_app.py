from flask import Flask,render_template,request
import numpy as np 
import pickle

with open ('iris-model.pickle','rb') as f:
    model=pickle.load(f)

app=Flask(__name__)

@app.route('/')
def app_home():
    return render_template('index.html')

@app.route('/prediction',methods=['post'])
def prediction():
    sepalL=float(request.form.get('sepalL'))
    sepalW=float(request.form.get('sepalW'))
    petalL=float(request.form.get('petalL'))
    petalW=float(request.form.get('petalW'))

    input_elements=np.array([sepalL,sepalW,petalL,petalW])
    output=model.predict([input_elements])[0]

    if output==0:
        output='iris-setosa'
    elif output==1:
        output='iris-versicolor'
    
    elif output==2:
        output='iris-verginica'

    return render_template('after.html',result=output)
    
if __name__=='__main__':
    app.run(debug=True,port=8080,host='0.0.0.0')



