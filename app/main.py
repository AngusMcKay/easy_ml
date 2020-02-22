"""
flask app for auto generating reports
including machine learning models as required
author: Angus
"""

from flask import Flask, render_template, redirect, url_for, request, flash, session
from app.helpers.general_helper_functions import allowed_file, random_filename, FormValidation
from app.helpers.model_builder_helper_functions import DataReview, ProcessVariables, ModelTraining
from app.helpers.get_predictions_helper_functions import Predictions
import os
import pandas as pd
from uuid import uuid4
import pickle

app = Flask(__name__)
app.secret_key = 'myseceretkey'

'''
HOME
'''
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


'''
INSTRUCTIONS
'''
@app.route('/instructions/main')
def instructions_main():
    return render_template('instructions/main.html')


@app.route('/instructions/gettingstarted/main')
def instructions_getting_started():
    return render_template('instructions/gettingstarted/main.html')


@app.route('/instructions/gettingstarted/overview')
def instructions_getting_started_overview():
    return render_template('instructions/gettingstarted/overview.html')


@app.route('/instructions/data/main')
def instructions_data():
    return render_template('instructions/data/main.html')


@app.route('/instructions/data/format')
def instructions_data_format():
    return render_template('instructions/data/format.html')


@app.route('/instructions/data/types')
def instructions_data_types():
    return render_template('instructions/data/types.html')


@app.route('/instructions/data/size')
def instructions_data_size():
    return render_template('instructions/data/size.html')


@app.route('/instructions/data/cleaning')
def instructions_data_cleaning():
    return render_template('instructions/data/cleaning.html')


@app.route('/instructions/aistudio/main')
def instructions_aistudio():
    return render_template('instructions/aistudio/main.html')


'''
AI STUDIO
'''
@app.route('/aistudio/main')
def ai_studio_main():
    return render_template('aistudio/main.html')


# MODEL BUILDER #

@app.route('/aistudio/modelbuilder/main')
def ai_studio_model_builder_main():
    return render_template('aistudio/modelbuilder/main.html')


@app.route('/aistudio/modelbuilder/uploaddata')
def ai_studio_model_builder_upload_data():
    return render_template('aistudio/modelbuilder/uploaddata.html')


@app.route('/aistudio/modelbuilder/variables', methods=['POST'])
def ai_studio_model_builder_variables():

    extensions_types = ['csv']  # should go in config
    upload_folder = 'data'

    # first check file exists
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('ai_studio_model_builder_upload_data'))

    file = request.files['file']

    # check filename exists
    if file.filename == '':
        flash('No file selected!')
        return redirect(url_for('ai_studio_model_builder_upload_data'))

    if file and allowed_file(file.filename, extensions_types):
        upload_filename = random_filename('.csv')
        filepath = os.path.join(app.root_path, upload_folder, upload_filename)
        file.save(filepath)

    else:
        flash('File must be csv format!')
        return redirect(url_for('ai_studio_model_builder_upload_data'))

    raw_data = pd.read_csv(filepath)

    all_columns = list(raw_data.columns)
    # add some items to session for handling validation of variable renaming
    session['raw_data_columns'] = all_columns
    rename_default_values = [""]*len(all_columns)
    session['raw_data_columns_renamed'] = rename_default_values
    name_valid_default = ["True"]*len(all_columns)
    session['name_valid'] = name_valid_default

    session['raw_data_filepath'] = filepath

    return render_template('aistudio/modelbuilder/variables.html', all_columns=all_columns,
                           rename_values=rename_default_values, name_valid=name_valid_default)


@app.route('/aistudio/modelbuilder/variablesinvalidname')
def ai_studio_model_builder_variables_invalid_name():

    all_columns = session['raw_data_columns']
    rename_values = session['raw_data_columns_renamed']
    name_valid = session['name_valid']

    return render_template('aistudio/modelbuilder/variables.html', all_columns=all_columns,
                           rename_values=rename_values, name_valid=name_valid)


@app.route('/aistudio/modelbuilder/datareview', methods=['POST'])
def ai_studio_model_builder_data_review():

    # check variable names
    all_columns = session['raw_data_columns']
    session['raw_data_columns_renamed'] = [request.form[c+'_rename'] for c in all_columns]
    form_validation = FormValidation(request)
    invalid_exists, variable_valid = form_validation.form_contains_invalid_values(allowed_regex='^[A-Za-z0-9-_\s]*$')
    if invalid_exists:
        session['name_valid'] = [variable_valid[c+'_rename'] for c in all_columns]
        flash('Variable names can only include letters, numbers, spaces, dashes and underscores')
        return redirect(url_for('ai_studio_model_builder_variables_invalid_name'))

    # pick up raw data
    raw_data_filepath = session['raw_data_filepath']
    raw_data = pd.read_csv(raw_data_filepath)
    all_columns = raw_data.columns

    # process variable inputs and check no duplicate column names
    variable_processor = ProcessVariables(request, all_columns)
    processed_data = variable_processor.process_data(raw_data)
    duplicates_exist, name_valid = variable_processor.columns_names_contain_duplicates(processed_data)
    if duplicates_exist:
        session['name_valid'] = name_valid
        flash('Included variables must have different names')
        return redirect(url_for('ai_studio_model_builder_variables_invalid_name'))

    # save data to file
    folder = 'data/processed'
    filename = random_filename('.pkl')
    filepath = os.path.join(app.root_path, folder, filename)
    with open(filepath, "wb") as f:
        pickle.dump(processed_data, f)

    # review data for bad data types
    data_reviewer = DataReview(processed_data, variable_processor.variables_dict)
    bad_numerical_data_types_dict, bad_categorical_data_types_dict = data_reviewer.find_bad_data_types()

    # save what needed to session
    session['cleaned_data_filepath'] = filepath
    session['variables_dict'] = variable_processor.variables_dict

    if len(sum(bad_numerical_data_types_dict.values(), []) + sum(bad_categorical_data_types_dict.values(), [])) == 0:
        message = "Data review complete, no amendments needed. Click continue to proceed to model training."
        return render_template('aistudio/modelbuilder/datareviewcomplete.html', message=message)

    else:
        return render_template('aistudio/modelbuilder/datareview.html',
                               bad_numerical_data_types_dict=bad_numerical_data_types_dict,
                               bad_categorical_data_types_dict=bad_categorical_data_types_dict)


@app.route('/aistudio/modelbuilder/datareviewcomplete', methods=['POST'])
def ai_studio_model_builder_data_review_complete():

    # pick up processed data and variables dict
    processed_data_filepath = session['cleaned_data_filepath']
    with open(processed_data_filepath, "rb") as f:
        processed_data = pickle.load(f)
    variables_dict = session['variables_dict']

    # clean up data
    data_reviewer = DataReview(processed_data, variables_dict)
    cleaned_data, rows_removed, entries_replaced = data_reviewer.remove_and_replace_bad_values(request.form)

    # resave
    with open(processed_data_filepath, "wb") as f:
        pickle.dump(cleaned_data, f)

    # create message for user
    rows_removed_message = ""
    entries_replaced_message = ""
    if rows_removed > 0:
        rows_removed_message = ", " + str(rows_removed) + " rows removed"
    if entries_replaced > 0:
        entries_replaced_message = ", " + str(entries_replaced) + " entries replaced"
    message = "Data cleaning completed" + rows_removed_message + entries_replaced_message + "."

    return render_template('aistudio/modelbuilder/datareviewcomplete.html', message=message)



@app.route('/aistudio/modelbuilder/modeltraininginprogressredirecttest')
def ai_studio_model_builder_model_training_in_progress_redirect_test():
    return render_template('aistudio/modelbuilder/modeltraininginprogressredirecttest.html')


@app.route('/aistudio/modelbuilder/modeltraininginprogress')
def ai_studio_model_builder_model_training_in_progress():
    return render_template('aistudio/modelbuilder/modeltraininginprogress.html')


@app.route('/aistudio/modelbuilder/modeltrainingcomplete')
def ai_studio_model_builder_model_training_complete():

    # pick up cleaned data and variables info
    cleaned_data_filepath = session['cleaned_data_filepath']
    with open(cleaned_data_filepath, 'rb') as f:
        cleaned_data = pickle.load(f)
    variables_dict = session['variables_dict']

    # get model
    model_training = ModelTraining(cleaned_data, variables_dict)
    best_model_dict = model_training.train_and_get_best_model()

    # save model
    folder = 'models/tmp'
    filename = random_filename('.pkl')
    filepath = os.path.join(app.root_path, folder, filename)
    with open(filepath, "wb") as f:
        pickle.dump(best_model_dict, f)

    # save filepath to session
    session['tmp_model_filepath'] = filepath
    session['model_type'] = model_training.model_type
    session['mse'] = model_training.mse
    session['mae'] = model_training.mae
    session['mpe'] = model_training.mpe

    return render_template('aistudio/modelbuilder/modeltrainingcomplete.html')


@app.route('/aistudio/modelbuilder/trainingoutput')
def ai_studio_model_builder_training_output():

    prediction_variable_name = session['variables_dict']['prediction_variable_name']

    if session['model_type'] == 'regression':
        mae = round(session['mae'], 1)
        mpe = round(session['mpe'], 1)
        return render_template('aistudio/modelbuilder/trainingoutputregression.html',
                               mae=mae, mpe=mpe, prediction_variable_name=prediction_variable_name)
    else:
        return render_template('aistudio/modelbuilder/trainingoutputclassification.html')


@app.route('/aistudio/modelbuilder/savemodel')
def ai_studio_model_builder_save_model():
    return render_template('aistudio/modelbuilder/savemodel.html', model_name="")


@app.route('/aistudio/modelbuilder/savemodelinvalidname')
def ai_studio_model_builder_save_model_invalid_name():
    invalid_model_name = session['invalid_model_name']
    return render_template('aistudio/modelbuilder/savemodel.html', model_name=invalid_model_name)


@app.route('/aistudio/modelbuilder/modelsaved', methods=['POST'])
def ai_studio_model_builder_model_saved():

    # first check model name ok
    form_validation = FormValidation(request)
    invalid_exists, variable_valid = form_validation.form_contains_invalid_values(allowed_regex='^[A-Za-z0-9-_\s]+$')
    if invalid_exists:
        session['invalid_model_name'] = request.form['model_name']
        flash('Model name can only include letters, numbers, spaces, dashes and underscores')
        return redirect(url_for('ai_studio_model_builder_save_model_invalid_name'))

    tmp_model_filepath = session['tmp_model_filepath']

    with open(tmp_model_filepath, 'rb') as f:
        model_dict = pickle.load(f)

    model_name = request.form['model_name']

    # save model
    folder = 'models'
    filename = model_name + '.pkl'
    filepath = os.path.join(app.root_path, folder, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model_dict, f)

    return render_template('aistudio/modelbuilder/modelsaved.html')


# VALIDATION #

@app.route('/aistudio/validation/main')
def ai_studio_validation_main():
    return render_template('aistudio/validation/main.html')


# GET PREDICTIONS #
@app.route('/aistudio/getpredictions/main')
def ai_studio_get_predictions_main():
    return render_template('aistudio/getpredictions/main.html')


@app.route('/aistudio/getpredictions/selectmodel')
def ai_studio_get_predictions_select_model():
    # get list of available models
    folder = 'models'
    model_directory = os.path.join(app.root_path, folder)
    models_list = os.listdir(model_directory)
    models_list = [m.replace('.pkl', '') for m in models_list if m[-4:] == '.pkl']

    return render_template('aistudio/getpredictions/selectmodel.html', models_list=models_list)


@app.route('/aistudio/getpredictions/entervariables', methods=['POST'])
def ai_studio_get_predictions_enter_variables():
    # get model dictionary
    model_name = request.form['selected_model']
    session['selected_model'] = model_name
    model_filename = model_name + '.pkl'
    folder = 'models'
    model_filepath = os.path.join(app.root_path, folder, model_filename)
    with open(model_filepath, 'rb') as f:
        model_dict = pickle.load(f)

    # get variables
    prediction_variables = Predictions(model_dict)
    prediction_variable_name = prediction_variables.prediction_variable_name
    numerical_variables = prediction_variables.numerical_variables
    categorical_variables = prediction_variables.categorical_variables
    # sort categorical variable lists so they appear ordered in list
    for v in categorical_variables.keys():
        categorical_variables[v] = sorted(categorical_variables[v])

    # default variable values (to allow being able to set these after validation)
    default_variable_selection = {}
    for v in list(numerical_variables) + list(categorical_variables.keys()):
        default_variable_selection[v+'_input'] = ""
    default_variable_valid = {}
    for v in list(numerical_variables) + list(categorical_variables.keys()):
        default_variable_valid[v+'_input'] = "True"

    # store values in session to pick up in validation
    session['prediction_variable_name'] = prediction_variable_name
    session['numerical_variables'] = numerical_variables
    session['categorical_variables'] = categorical_variables
    session['variable_selection'] = default_variable_selection
    session['variable_valid'] = default_variable_valid
    session['enter_variables_or_prediction'] = 'variables'

    return render_template('aistudio/getpredictions/entervariables.html',
                           prediction_variable_name=prediction_variable_name,
                           numerical_variables=numerical_variables,
                           categorical_variables=categorical_variables,
                           variable_selection=default_variable_selection,
                           variable_valid=default_variable_valid)


@app.route('/aistudio/getpredictions/entervariablesinvalidnumber')
def ai_studio_get_predictions_enter_variables_invalid_number():

    prediction_variable_name = session['prediction_variable_name']
    numerical_variables = session['numerical_variables']
    categorical_variables = session['categorical_variables']
    variable_selection = session['variable_selection']
    variable_valid = session['variable_valid']

    return render_template('aistudio/getpredictions/entervariables.html',
                           prediction_variable_name=prediction_variable_name,
                           numerical_variables=numerical_variables,
                           categorical_variables=categorical_variables,
                           variable_selection=variable_selection,
                           variable_valid=variable_valid)


@app.route('/aistudio/getpredictions/prediction', methods=['POST'])
def ai_studio_get_predictions_prediction():

    # form validation - NEED TO FIX THIS TO PICK UP FORM CORRECTLY
    numerical_variable_inputs = [v+'_input' for v in session['numerical_variables']]
    form_validation = FormValidation(request)
    invalid_exists, variable_valid = form_validation.form_contains_invalid_values(
        allowed_regex="^-?[0-9]+\.?[0-9]*$", keys_to_check=numerical_variable_inputs)
    if invalid_exists:
        session['variable_selection'] = request.form
        session['variable_valid'] = variable_valid
        if session['enter_variables_or_prediction'] == 'prediction':
            flash('Numerical variables must be a valid number!')
            return redirect(url_for('ai_studio_get_predictions_prediction_invalid_number'))
        else:
            flash('Numerical variables must be a valid number!')
            return redirect(url_for('ai_studio_get_predictions_enter_variables_invalid_number'))

    # get model dictionary
    model_name = session['selected_model']
    model_filename = model_name + '.pkl'
    folder = 'models'
    model_filepath = os.path.join(app.root_path, folder, model_filename)
    with open(model_filepath, 'rb') as f:
        model_dict = pickle.load(f)

    # get variables
    predictions = Predictions(model_dict)
    prediction_variable_name = predictions.prediction_variable_name
    numerical_variables = predictions.numerical_variables
    categorical_variables = predictions.categorical_variables.copy()
    # sort categorical variable lists so they appear ordered in list
    for v in categorical_variables.keys():
        categorical_variables[v] = sorted(categorical_variables[v])

    # get prediction
    variable_selection = request.form
    prediction_data = predictions.make_prediction_data(variable_selection)
    prediction = predictions.predict(prediction_data)

    # store prediction in session to pick up in validation
    session['prediction'] = prediction
    session['enter_variables_or_prediction'] = 'prediction'

    return render_template('aistudio/getpredictions/prediction.html',
                           prediction=prediction,
                           prediction_variable_name=prediction_variable_name,
                           numerical_variables=numerical_variables,
                           categorical_variables=categorical_variables,
                           variable_selection=variable_selection,
                           variable_valid=variable_valid)


@app.route('/aistudio/getpredictions/predictioninvalidnumber')
def ai_studio_get_predictions_prediction_invalid_number():

    prediction_variable_name = session['prediction_variable_name']
    numerical_variables = session['numerical_variables']
    categorical_variables = session['categorical_variables']
    variable_selection = session['variable_selection']
    variable_valid = session['variable_valid']
    prediction = session['prediction']

    return render_template('aistudio/getpredictions/prediction.html',
                           prediction=prediction,
                           prediction_variable_name=prediction_variable_name,
                           numerical_variables=numerical_variables,
                           categorical_variables=categorical_variables,
                           variable_selection=variable_selection,
                           variable_valid=variable_valid)


# MODEL BREAKDOWN #
@app.route('/aistudio/modelbreakdown/main')
def ai_studio_model_breakdown_main():
    return render_template('aistudio/modelbreakdown/main.html')



'''
REPORTS
'''
@app.route('/reports/main')
def reports_main():
    return render_template('reports/main.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # host 0.0.0.0 makes available externally, port defaults to 5000
