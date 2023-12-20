from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


gb_pipe = pickle.load(open('gb_pipe.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/pred_salary', methods=['POST'])
def pred_salary():
    gender = request.form.get('gender')
    age = request.form.get('age')
    designation = request.form.get('des')
    unit = request.form.get('unit')
    leaves_used = request.form.get('lu')
    leaves_remaining = request.form.get('lr')
    ratings = request.form.get('rat')
    past_exp = request.form.get('pe')
    curr_exp = request.form.get('ce')

    input = pd.DataFrame([[gender, age, designation, unit, leaves_used, leaves_remaining, ratings, past_exp, curr_exp]],columns=['SEX', 'AGE', 'DESIGNATION', 'UNIT', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS', 'PAST EXP', 'CURR Exp'])
    prediction = gb_pipe.predict(input)[0]

    return str(prediction)



if __name__ == '__main__':
    app.run(debug=True)