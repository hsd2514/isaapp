import joblib
from flask import Flask, jsonify, request , render_template
import pandas as pd
import numpy as np
import json


data =    data = {
        'employee_id': 74430,
        'department': 'HR',
        'region': 'region_4',
        'education': 'Bachelors',
        'gender': 'f',
        'recruitment_channel': 'other',
        'no_of_trainings': 1,
        'age': 31,
        'previous_year_rating': 3.0,
        'length_of_service': 5,
        'KPIs_met_more_than_80': 0,
        'awards_won': 0
    }


#lets conect to our template
app = Flask(__name__)
app.json.sort_keys = False



@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    content = request.form
    request_data = json.loads(json.dumps(content))
    print(request_data)
    employee_id = content.get('employee_id')
    department = content.get('department')
    region = content.get('region')
    education = content.get('education')
    gender = content.get('gender')
    recruitment_channel = content.get('recruitment_channel')
    no_of_trainings = content.get('no_of_trainings')
    age = content.get('age')
    previous_year_rating = content.get('previous_year_rating')
    length_of_service = content.get('length_of_service')
    KPIs_met_more_than_80 = content.get('KPIs_met_more_than_80')
    awards_won = content.get('awards_won')

    
    # result = {
    #     'employee_id': int(employee_id),
    #     'department': str(department),
    #     'region': region,
    #     'education': education,
    #     'gender': gender,
    #     'recruitment_channel': recruitment_channel,
    #     'no_of_trainings': int(no_of_trainings),
    #     'age': int(age),
    #     'previous_year_rating': float(previous_year_rating),
    #     'length_of_service': int(length_of_service),
    #     'KPIs_met_more_than_80': int(KPIs_met_more_than_80),
    #     'awards_won': int(awards_won),
    # }
    
    result={
        "employee_id": int(employee_id),
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": int(no_of_trainings),
        "age": int(age),
        "previous_year_rating": float(previous_year_rating),
        "length_of_service": int(length_of_service),
        "KPIs_met_more_than_80": int(KPIs_met_more_than_80),
        "awards_won": int(awards_won)
        
        
    }
    
    df=pd.DataFrame([result])
    model=joblib.load('production_pipelineV3.pkl')
    prediction=model.predict(df)
    prediction = int(prediction[0])
    
    
    return render_template('result.html',employee_id=employee_id,department=department,region=region,education=education,gender=gender,recruitment_channel=recruitment_channel,no_of_trainings=no_of_trainings,age=age,previous_year_rating=previous_year_rating,length_of_service=length_of_service,KPIs_met_more_than_80=KPIs_met_more_than_80,awards_won=awards_won,prediction=prediction)

    


if __name__ == '__main__':
    app.run(debug=True)