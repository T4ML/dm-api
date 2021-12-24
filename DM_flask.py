from flask import Flask, request,jsonify,render_template
import numpy as np
import pickle
import os

import pandas as pd
from preprocessing import preprocessing


BASEDIR = "C:/JupyterNotebook/PROJECT/dm-api/"
os.chdir(BASEDIR)


MODELDIR = "models/"
model = pickle.load(open(MODELDIR+'RandomForestSimple.pickle','rb'))
# scaler3 = pickle.load(open(MODELDIR+'stdX3_scaler.pickle','rb'))


app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

# def predict():
#     cgpa = request.form.get('cgpa')
#     iq = request.form.get('iq')
#     profile_score = request.form.get('profile_score')
#     input_query = np.array([[cgpa,iq,profile_score]])
#     result = model.predict(input_query)[0]
#     return jsonify({'placement':str(result)})
col_names = ["T_AGE","T_INCOME","T_MARRY","T_HEIGHT","T_WEIGHT","T_BMI","T_DRINK","T_DRDU","T_TAKFQ","T_TAKAM","T_RICEFQ","T_RICEAM","T_WINEFQ","T_WINEAM","T_SOJUFQ","T_SOJUAM","T_BEERFQ","T_BEERAM","T_HLIQFQ","T_HLIQAM","T_SMOKE","T_SMDUYR","T_SMDUMO","T_SMAM","T_PSM","T_EXER"]

@app.route('/',methods=['POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        T_AGE = int(request.form["T_AGE"])
        T_INCOME = int(request.form["T_INCOME"])
        T_MARRY = int(request.form["T_MARRY"])
        T_HEIGHT = int(request.form["T_HEIGHT"])
        T_WEIGHT = int(request.form["T_WEIGHT"])
        # T_BMI = float(request.form["T_BMI"])
        T_DRINK = int(request.form["T_DRINK"])
        T_DRDU = int(request.form["T_DRDU"])
        T_TAKFQ = int(request.form["T_TAKFQ"])
        T_TAKAM = int(request.form["T_TAKAM"])
        T_RICEFQ = int(request.form["T_RICEFQ"])
        T_RICEAM = int(request.form["T_RICEAM"])
        T_WINEFQ = int(request.form["T_WINEFQ"])
        T_WINEAM = int(request.form["T_WINEAM"])
        T_SOJUFQ = int(request.form["T_SOJUFQ"])
        T_SOJUAM = int(request.form["T_SOJUAM"])
        T_BEERFQ = int(request.form["T_BEERFQ"])
        T_BEERAM = int(request.form["T_BEERAM"])
        T_HLIQFQ = int(request.form["T_HLIQFQ"])
        T_HLIQAM = int(request.form["T_HLIQAM"])
        T_SMOKE = int(request.form["T_SMOKE"])
        T_SMDUYR = int(request.form["T_SMDUYR"])
        T_SMDUMO = int(request.form["T_SMDUMO"])
        T_SMAM = int(request.form["T_SMAM"])
        T_PSM = int(request.form["T_PSM"])
        T_EXER = int(request.form["T_EXER"])
        T_BMI= float(T_WEIGHT/(T_HEIGHT**2))
        
        #get prediction
        input_data = [T_AGE,T_INCOME,T_MARRY,T_HEIGHT,T_WEIGHT,T_BMI,T_DRINK,T_DRDU,T_TAKFQ,T_TAKAM,T_RICEFQ,T_RICEAM,T_WINEFQ,T_WINEAM,T_SOJUFQ,T_SOJUAM,T_BEERFQ,T_BEERAM,T_HLIQFQ,T_HLIQAM,T_SMOKE,T_SMDUYR,T_SMDUMO,T_SMAM,T_PSM,T_EXER]
        

        
        # simple model used 3 cols
        # input_df = pd.DataFrame(scaler3.transform(np.array(input_cols).reshape(1,-1)), columns =col_names )
        # input_df = pd.DataFrame(np.array(scaler3.transform(input_cols)).reshape(-1,1), columns =["T_AGE", "T_INCOME", "T_BMI"] )
        
        processed_df = preprocessing(input_data)
        
        
        prediction = model.predict(processed_df)
        prob = round((model.predict_proba(processed_df)[0] * 100)[1],2)
        output = round(prediction[0], 2)
        if prediction == 0:
            res = "당뇨 음성"
        else:
            res = "당뇨 양성"
        return render_template("index.html", prediction_text='당뇨 여부 : {}, 확률 : {}%'.format(res,prob))

if __name__ == '__main__':
    app.run(debug=True,port = 8080)

