
import string
from datetime import date
import random
from uuid import uuid4
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import pandas as pd
import os

'''
variables and predictions
'''
class Predictions:

    def __init__(self, model_dict):
        self.model_dict = model_dict
        self.prediction_variable_name = model_dict['prediction_variable_name']
        self.numerical_variables = self.get_numerical_variables()
        self.categorical_variables = self.model_dict['response_coding_dict']

    def get_numerical_variables(self):
        return [f for f in self.model_dict['features'] if f not in self.model_dict['response_coding_dict'].keys()]

    def make_prediction_data(self, variable_inputs_request):
        features = self.model_dict['features']
        feature_inputs = [variable_inputs_request[f+'_input'] for f in features]

        # switch categorical features for response coding
        for c in self.categorical_variables.keys():
            response_coding = self.categorical_variables[c]
            feature_position = features.index(c)
            category = feature_inputs[feature_position]
            feature_inputs[feature_position] = response_coding[category]

        feature_inputs = np.array(feature_inputs).reshape(1, len(features))
        prediction_data = pd.DataFrame(feature_inputs, columns=features)

        return prediction_data

    def predict(self, prediction_data):
        return str(round(self.model_dict['model'].predict(prediction_data)[0], 2))


