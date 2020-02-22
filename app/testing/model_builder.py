

import pandas as pd
import numpy as np
import pickle

# generate some random data to work with
sq_ft = np.random.normal(100, 10, 1000)
rooms = np.random.choice([2, 3, 4, 5, 6], 1000, replace=True)
previous_price = np.random.normal(200000, 10000, 1000)
annual_costs = previous_price/500 + sq_ft*1 + rooms*30 + np.random.normal(100, 15, 1000)
test_data = pd.DataFrame({'sq_ft': sq_ft,
                          'rooms': rooms,
                          'previous_price': previous_price,
                          'annual_costs': annual_costs})
test_data.to_csv('/home/angus/projects/property/app/testing/test_data/test_data.csv', index=False)


'''
upload csv
'''
input_filepath = '/home/angus/projects/property/app/testing/test_data/test_data.csv'
raw_data = pd.read_csv(input_filepath)


'''
variable selection
'''
all_columns = list(raw_data.columns)
input_variable_selection = {
    'input_prediction_variable': 'annual_costs',
    'input_variable_names': ['Square feet', 'Number of rooms', 'Previous price', 'Annual costs'],
    'input_variable_type': ['Numerical', 'Numerical', 'Numerical', 'Numerical'],
    'input_variable_include': [True, True, True, True]
}

# make sure prediction variable is included
prediction_variable = input_variable_selection['input_prediction_variable']
prediction_variable_col_number = all_columns.index(prediction_variable)
input_variable_selection['input_variable_include'][prediction_variable_col_number] = True

# get prediction variable details
prediction_variable_renamed = input_variable_selection['input_variable_names'][prediction_variable_col_number]
prediction_type = input_variable_selection['input_variable_type'][prediction_variable_col_number]

# rename and subset raw_data
processed_data = raw_data.copy()
processed_data.columns = input_variable_selection['input_variable_names']
processed_data = processed_data.loc[:, input_variable_selection['input_variable_include']]

# get features
feature_list = list(processed_data.columns)
feature_list = [f for f in feature_list if f!=prediction_variable_renamed]

# other validation is all names are different and contain letters and numbers
# at least one variable that is not the prediction variable


'''
data review and cleaning
'''
processed_data.head(3)
# include other things here like max and min and missing values that will be removed
# will have to check numerical values are numerical etc
# eventually might want to include imputation


'''
model training (note, uses all data to learn parameters via cv, testing is done separately by app user)
'''
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV

train_X = processed_data[feature_list]
train_y = processed_data[prediction_variable_renamed]

if prediction_type == 'Numerical':
    lin_mod = LinearRegression()
    rf_mod = RandomForestRegressor(random_state=123)
else:
    lin_mod = LogisticRegression(random_state=123, C=99999)
    rf_mod = RandomForestClassifier(random_state=123)

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

# select best model
def get_best_model(random_forest, linear_model, model_class, data_X, data_y):
    if model_class == 'Numerical':
        lin_mod_preds = linear_model.predict(data_X)
        lin_mod_mse = np.sqrt(sum((lin_mod_preds - data_y)**2) / len(data_y))
        rf_mod_preds = random_forest.predict(data_X)
        rf_mod_mse = np.sqrt(sum((rf_mod_preds - data_y) ** 2) / len(data_y))
        if rf_mod_mse <= lin_mod_mse:
            return {'model_class': 'regression',
                    'model_type': 'random_forest',
                    'model': random_forest}
        else:
            return {'model_class': 'regression',
                    'model_type': 'linear_model',
                    'model': linear_model}

    else:
        # need to think about how to deal with multi-class problems, maybe restrict to two classes?
        # and then use auc. And think about multi-class in another way.
        pass


best_model = get_best_model(rf_mod, lin_mod, prediction_type, train_X, train_y)
best_model['features'] = feature_list


'''
model validation
'''
if best_model['model_class'] == 'regression':
    train_preds = best_model['model'].predict(train_X)
    errors = train_preds - train_y
    absolute_errors = abs(errors)
    mae = sum(absolute_errors) / len(train_y)

    percentage_errors = absolute_errors / train_y
    mean_percentage_error = sum(percentage_errors) / len(train_y)

    # would be good to show range of errors in like a box plot or histogram

else:
    pass


'''
review model output
'''
filepath = '/home/angus/projects/property/app/models/Categorical model.pkl'
with open(filepath, "rb") as f:
    cat_model = pickle.load(f)

filepath = '/home/angus/projects/property/app/models/Numerical model.pkl'
with open(filepath, "rb") as f:
    num_model = pickle.load(f)



'''
Model approximation
'''
import shap
if best_model['model_type'] == 'random_forest':
    pass





'''
Get individual predictions
'''
input_feature1 = 1  # etc...


