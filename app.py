import pickle
import math
import numpy as np
import pandas as pd
import catboost as cat
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from math import sqrt
from flask import Flask, escape, request, url_for, redirect, render_template, request



#run_with_ngrok(app)
model = pickle.load(open('catboost_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route('/prediction', methods = ['POST','GET'])
def prediction():
  if request.method == 'POST':
    Location = request.form['location']
    Network_Availability = request.form['network_available']
    District = request.form['district']
    Zip_Code = request.form['zip_code']


    Op = ['GP', 'Robi-Airtel', 'Banglalink', 'Teletalk']
    model_input = []
    score = []

    ele2 = Location
    ele3 = Network_Availability
    ele4 = District
    ele5 = Zip_Code


    for i in range(len(Op)):
      ele1 = Op[i]
      model_input.clear()
      model_input.append(ele1)
      model_input.append(ele2)
      model_input.append(ele3)
      model_input.append(ele4)
      model_input.append(ele5)


      Operator = model_input[0]
      Upazila_or_Thana = model_input[1]
      Active_Network_Available    = model_input[2]
      District = model_input[3]
      Area_Zip_Code = model_input[4]

      p = model.predict(model_input)

      score.append(p)

      temp = p - math.floor(p)

      if (temp > 0.5):
        p = math.ceil(p)
      elif (temp == 0.5):
        p = p
      else:
        p = math.floor(p)

      #print(f'Operator : {Operator} {Active_Network_Available} score in Area : {Upazila_or_Thana} (Out of 100) is : {round(p,2)}')


    #to convert lists to dictionary
    res = dict(zip(Op, score))

    # Printing resultant dictionary
    #print ("Resultant dictionary is : " +  str(res))
    mx = max(res, key=res.get)
    #print(f'\nBest Operator in Your Upazila/Thana is : {mx}')

    max_in_dictionary = {'operator' : mx}            #'score' : max(score)}


  return render_template('prediction.html', prediction = max_in_dictionary)


if __name__ == "__main__":
  app.run(debug=True)
