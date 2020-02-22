
import string
from datetime import date
import random
from uuid import uuid4
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
import re


def allowed_file(filename, allowed_extensions):
    return filename.rsplit('.', 1)[1].lower() in allowed_extensions


def random_filename(extension='.csv', date_prefix=False, letters_or_uuid='uuid'):
    """Generate a random string """
    if date_prefix:
        date_prefix = date.today().strftime("%Y%m%d")
    else:
        date_prefix = ""

    if letters_or_uuid == 'uuid':
        random_string = str(uuid4())
    else:
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for i in range(10))

    return date_prefix + random_string + extension


class FormValidation:

    def __init__(self, request):
        self.request = request

    def form_contains_invalid_values(self, allowed_regex, keys_to_check='all'):
        form = self.request.form
        variable_valid = {k: 'True' for k in form.keys()}
        invalid_exists = False
        if keys_to_check == 'all':
            keys_to_check = list(form.keys())
        for k in form.keys():
            do_check = (k in keys_to_check)
            valid = bool(re.match(allowed_regex, form[k]))
            if do_check and not valid:
                variable_valid[k] = 'False'
                invalid_exists = True
        return invalid_exists, variable_valid

    def form_contains_duplicates(self):
        form = self.request.form
        values_list = list(form.values())
        number_values = len(values_list)
        number_unique_values = len(set(values_list))
        return number_values != number_unique_values
