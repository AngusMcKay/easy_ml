
import string
from datetime import date
import random
from uuid import uuid4
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import re


'''
data review
'''


class ProcessVariables:

    def __init__(self, request, all_columns):
        self.request = request.form
        self.all_columns = all_columns
        self.variables_dict = self.generate_variables_dict()

    def get_renamed_variable(self, original_name):
        if self.request[original_name+'_rename'] == "":
            return original_name
        else:
            return self.request[original_name+'_rename']

    def generate_variables_dict(self):
        variable_selection_dict = {
            'prediction_variable': self.request['prediction_variable'],
            'original_names': list(self.all_columns),
            'variable_names': [self.get_renamed_variable(c) for c in self.all_columns],
            'variable_types': [self.request[c+'_type'] for c in self.all_columns],
            'variable_include': [self.request.get(c + '_include', "off") == "on" for c in self.all_columns]
        }

        # make sure prediction variable is included
        prediction_variable = variable_selection_dict['prediction_variable']
        prediction_variable_col_number = variable_selection_dict['original_names'].index(prediction_variable)
        variable_selection_dict['variable_include'][prediction_variable_col_number] = True
        variable_selection_dict['prediction_variable_name'] = variable_selection_dict['variable_names'][
            prediction_variable_col_number]

        return variable_selection_dict

    def process_data(self, data):
        processed_data = data.copy()
        processed_data.columns = self.variables_dict['variable_names']
        # convert categorical variables to strings - need to do this for referencing later in response coding
        for i, c in enumerate(processed_data.columns):
            if self.variables_dict['variable_types'][i] == 'Categorical':
                processed_data[c] = processed_data[c].astype(str)

        processed_data = processed_data.loc[:, self.variables_dict['variable_include']]

        return processed_data

    def columns_names_contain_duplicates(self, data):
        data_columns = np.array(data.columns)
        duplicate = [sum(c == data_columns) > 1 for c in data_columns]
        duplicates_exist = sum(duplicate) > 0
        name_valid = [str(not d) for d in duplicate]
        return duplicates_exist, name_valid


class DataReview:

    # TO ADD:
    # option to remove outliers
    # data imputation

    def __init__(self, data, variables_dict):
        self.data = data
        self.variables_dict = variables_dict

    def is_number(self, n):
        try:
            if np.isnan(n):
                return False
        except:
            pass
        try:
            float(n)
            return True
        except ValueError:
            return False

    def find_bad_data_types(self):
        bad_categorical_data_types_dict = {}
        bad_numerical_data_types_dict = {}
        for c in self.data.columns:
            variable_number = self.variables_dict['variable_names'].index(c)
            variable_type = self.variables_dict['variable_types'][variable_number]
            if variable_type == 'Numerical':
                variable_values = self.data[c]
                nans = [not self.is_number(x) for x in variable_values]
                nan_row_numbers = [i for i in range(len(variable_values)) if nans[i]]
                bad_numerical_data_types_dict[c] = nan_row_numbers
            else:
                variable_values = self.data[c]
                nulls = variable_values.isnull()
                null_row_numbers = [i for i in range(len(variable_values)) if nulls[i]]
                bad_categorical_data_types_dict[c] = null_row_numbers

        return bad_numerical_data_types_dict, bad_categorical_data_types_dict

    def remove_and_replace_bad_values(self, data_review_request_form):
        bad_numerical_data_types_dict, bad_categorical_data_types_dict = self.find_bad_data_types()
        rows_to_remove = []
        entries_replaced = 0
        for v in data_review_request_form.keys():
            variable_number = self.variables_dict['variable_names'].index(v)
            variable_type = self.variables_dict['variable_types'][variable_number]
            variable_action = data_review_request_form[v]
            if variable_type == 'Numerical':
                if variable_action == 'Replace':
                    all_values = self.data[v]
                    bad_rows = bad_numerical_data_types_dict[v]
                    valid_values = [float(all_values[i]) for i in range(len(all_values)) if i not in bad_rows]
                    mean_valid_value = np.mean(valid_values)
                    row_mask = [i in bad_rows for i in range(len(all_values))]
                    self.data.loc[row_mask, v] = mean_valid_value
                    entries_replaced += sum(row_mask)
                else:
                    rows_to_remove += bad_numerical_data_types_dict[v]
            else:
                if variable_action == 'Replace':
                    row_mask = [i in bad_categorical_data_types_dict[v] for i in range(len(self.data))]
                    self.data.loc[row_mask, v] = '_unknown_'
                else:
                    rows_to_remove += bad_categorical_data_types_dict[v]

        rows_to_remove = list(set(rows_to_remove))
        rows_to_keep_mask = [i for i in range(len(self.data)) if i not in rows_to_remove]
        self.data = self.data.iloc[rows_to_keep_mask, :]

        rows_removed = len(rows_to_remove)

        return self.data, rows_removed, entries_replaced



'''
model training
'''


class ModelTraining:

    def __init__(self, data, variables_dict):
        self.data = data
        self.variables_dict = variables_dict
        self.prediction_variable = self.variables_dict['prediction_variable']
        self.prediction_variable_col_number = self.variables_dict['original_names'].index(self.prediction_variable)
        self.prediction_variable_name = self.variables_dict['prediction_variable_name']
        self.model_type = self.get_model_type()
        self.features = self.get_features()
        self.response_coding_dict = self.response_code_variables()

        # the following attributes are assigned by get_best_model()
        self.preds = None
        self.mse = None
        self.mae = None
        self.mpe = None

    def get_model_type(self):
        prediction_variable_type = self.variables_dict['variable_types'][self.prediction_variable_col_number]
        if prediction_variable_type == 'Categorical':
            return 'classification'
        else:
            return 'regression'

    def get_features(self):
        all_columns = self.data.columns
        features = [c for c in all_columns if c != self.prediction_variable_name]
        return features

    def response_code_variables(self):
        # applies response coding to data and also returns response coding dictionary to be used later in prediction
        features = self.features
        prediction_variable_name = self.variables_dict['prediction_variable_name']

        response_coding_dict = {}
        for i, feature in enumerate(features):

            feature_dict_pos = self.variables_dict['variable_names'].index(feature)
            feature_type = self.variables_dict['variable_types'][feature_dict_pos]

            if feature_type == 'Categorical':

                variable_coding_dict = {}
                unique_categories = list(self.data[feature].unique())

                for c in unique_categories:

                    # first add to dictionary, then replace category with response code value
                    variable_coding_dict[c] = np.mean(
                        self.data.loc[self.data[feature] == c, prediction_variable_name])
                    self.data.loc[self.data[feature] == c, feature] = variable_coding_dict[c]

                response_coding_dict[feature] = variable_coding_dict

        return response_coding_dict

    def train_and_get_best_model(self):
        if self.model_type == 'regression':
            lin_mod = LinearRegression()
            rf_mod = RandomForestRegressor(random_state=123)
        else:
            lin_mod = LogisticRegression(random_state=123, C=99999)
            rf_mod = RandomForestClassifier(random_state=123)

        train_X = self.data[self.features]
        train_y = self.data[self.prediction_variable_name]

        # fit linear model
        lin_mod.fit(train_X, train_y)

        # cv for random forest
        # note: eventually might want to be able to control parameters through the app somehow
        rf_parameters = {"n_estimators": [5, 10, 30, 50, 100],
                         "max_depth": [2, 3, 5, 7, 10],
                         "min_samples_leaf": [2, 5, 15, 50]
                         }
        rf_cv = GridSearchCV(rf_mod, rf_parameters)
        rf_cv.fit(train_X, train_y)
        rf_cv_best_params = rf_cv.best_params_
        rf_mod.set_params(**rf_cv_best_params)
        rf_mod.fit(train_X, train_y)

        best_model_dict = self.get_best_model(rf_mod, lin_mod, train_X, train_y)

        return best_model_dict

    def get_best_model(self, random_forest, linear_model, data_X, data_y):
        if self.model_type == 'regression':
            lin_mod_preds = linear_model.predict(data_X)
            lin_mod_mse = np.sqrt(sum((lin_mod_preds - data_y) ** 2) / len(data_y))
            rf_mod_preds = random_forest.predict(data_X)
            rf_mod_mse = np.sqrt(sum((rf_mod_preds - data_y) ** 2) / len(data_y))
            if rf_mod_mse <= lin_mod_mse:
                self.preds = rf_mod_preds
                self.mse = rf_mod_mse
                self.mae = sum(abs(rf_mod_preds - data_y)) / len(data_y)
                self.mpe = 100*sum(abs(rf_mod_preds / data_y - 1)) / len(data_y)
                return {'model_class': 'regression',
                        'model_type': 'random_forest',
                        'model': random_forest,
                        'prediction_variable_name': self.prediction_variable_name,
                        'features': self.features,
                        'response_coding_dict': self.response_coding_dict}
            else:
                return {'model_class': 'regression',
                        'model_type': 'linear_model',
                        'model': linear_model,
                        'prediction_variable_name': self.prediction_variable_name,
                        'features': self.featuresm,
                        'response_coding_dict': self.response_coding_dict}

        else:
            # need to think about how to deal with multi-class problems, maybe restrict to two classes?
            # and then use auc. And think about multi-class in another way.
            pass




