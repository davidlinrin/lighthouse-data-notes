from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)

num_feats = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
cat_feats = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

class ToDenseTransformer():

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit(self, X, y=None, **fit_params):
        return self
    
def numFeat(data):
    return data[num_feats]

def catFeat(data):
    return data[cat_feats]
    
class Proba(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        res_prob = model.predict_proba(df)
        return res_prob.tolist()
    
class Status(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        res = model.predict(df)
        return res.tolist()

model = pickle.load(open( "project_IV.p", "rb"))
#model = pickle.load(open( "project_IV_2.p", "rb"))


api.add_resource(Proba, '/proba')
api.add_resource(Status, '/status')

if __name__ == '__main__':
    app.run(debug=True)