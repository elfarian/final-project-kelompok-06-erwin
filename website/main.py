from flask import Flask, jsonify, make_response, render_template, request
from flask_expects_json import expects_json
from jsonschema import ValidationError
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

schema = {
    'type': 'object',
    'properties': {
        'store': {'type': 'number'},
        'unemployment': {'type': 'number'},
        'cpi': {'type': 'number'},
        'temperature': {'type': 'number'},
        'holiday': {'type': 'number'},
        'day': {'type': 'number'},
        'month': {'type': 'number'},
    },
    'required': ['store', 'unemployment', 'cpi', 'temperature', 'holiday', 'day', 'month']
}

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
@expects_json(schema)
def predict():
    input_data = request.get_json(force=True)
    input_data = np.array(input_data).tolist()

    store = input_data['store']
    unemployment = input_data['unemployment']
    cpi = input_data['cpi']
    temperature = input_data['temperature']
    day = input_data['day']
    month = input_data['month']
    holiday = input_data['holiday']

    new_data_for_detect = [
        holiday, temperature, cpi, unemployment, day, month
    ]
    for x in range(1, 46):
        status_store = 0
        if x == store:
            status_store = 1
        new_data_for_detect.append(status_store)

    filename_reg = 'model/regresi_linier_model.sav'
    filename_random_forest = 'model/random_forest_model.sav'

    loaded_model_reg = pickle.load(open(filename_reg, 'rb'))
    loaded_model_random_forest = pickle.load(open(filename_random_forest, 'rb'))

    X = [new_data_for_detect]

    pred_reg = loaded_model_reg.predict(X)
    pred_random_forest = loaded_model_random_forest.predict(X)
    
    output = [{'message': "Model 1 = $%.5f" % pred_reg[0]}, 
    {'message': "Model 2 = $%.5f" % pred_random_forest[0] }]

    return jsonify(output)


    
@app.errorhandler(400)
def bad_request(error):
    if isinstance(error.description, ValidationError):
        original_error = error.description
        return make_response(jsonify({'error': original_error.message}), 400)
    # handle other "Bad Request"-errors
    return error
