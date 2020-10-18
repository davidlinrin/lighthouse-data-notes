import pandas as pd
import numpy as np
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

class add(Resource):
    def get(self):
        parser.add_argument('num_1', type=float)
        parser.add_argument('num_2', type=float)
        args = parser.parse_args()
        num_1 = args['num_1']
        num_2 = args['num_2']
        if (num_2 !=None) and (num_1!=None):
            calculation = f'sum: {num_1+num_2}!'
        else:
            calculation = f'{num_1} and {num_2}'
        return (jsonify(calculation))
    
class subtract(Resource):
    def get(self):
        # create request parser
        parser = reqparse.RequestParser()
            # create argument 'num_1'
        parser.add_argument('num_1', type=int)
            # parse 'num_1'
        num_1 = parser.parse_args().get('num_1')
        parser2 = reqparse.RequestParser()
            # create argument 'num_2'
        parser2.add_argument('num_2', type=int)
            # parse 'name'
        num_2 = parser2.parse_args().get('num_2')   
        if num_2 & num_1:
            calculation = f'Subtraction of num_1-num_2: {num_1-num_2}!'
        else:
            calculation = 'you need to input two numbers for parameters num_1 and num_2!'
        # make json from greeting string 
        return jsonify(beepbeep_boop=calculation)  
    
class multiply(Resource):
    def get(self):
            # create request parser
        parser = reqparse.RequestParser()
            # create argument 'num_1'
        parser.add_argument('num_1', type=int)
            # parse 'num_1'
        num_1 = parser.parse_args().get('num_1')
        parser2 = reqparse.RequestParser()
            # create argument 'num_2'
        parser2.add_argument('num_2', type=int)
            # parse 'name'
        num_2 = parser2.parse_args().get('num_2')   
        if num_2 & num_1:
            calculation = f'Multiplication of numbers is: {num_1*num_2}!'
        else:
            calculation = 'you need to input two numbers for parameters num_1 and num_2!'
            # make json from greeting string 
        return jsonify(beepbeep_boop=calculation)
    
class divide(Resource):
    def get(self):
            # create request parser
        parser = reqparse.RequestParser()
            # create argument 'num_1'
        parser.add_argument('num_1', type=int)
            # parse 'num_1'
        num_1 = parser.parse_args().get('num_1')
        parser2 = reqparse.RequestParser()
            # create argument 'num_2'
        parser2.add_argument('num_2', type=int)
            # parse 'name'
        num_2 = parser2.parse_args().get('num_2')   
        if num_2 & num_1:
            calculation = f'Division of numbers is: {num_1/num_2}!'
        else:
            calculation = 'you need to input two numbers for parameters num_1 and num_2!'
            # make json from greeting string 
        return jsonify(beepbeep_boop=calculation)  
    
#assign endpoint
api.add_resource(add, '/add',)    
api.add_resource(subtract, '/subtract',)    
api.add_resource(multiply, '/multiply',)   
api.add_resource(divide, '/divide',) 

if __name__ == '__main__':
    app.run(debug=True)
