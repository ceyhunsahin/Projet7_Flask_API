import flask
import pandas as pd
import json
import numpy as np
import xgboost as xgb
import os
from flask import send_from_directory


path = 'https://raw.githubusercontent.com/ceyhunsahin/Projet_7_Dashboard/master/Projet_File/test_sample_data_home_risk.csv'
path2 = 'https://raw.githubusercontent.com/ceyhunsahin/Projet_7_Dashboard/master/Projet_File/test_sample_data_home_risk_normalise.csv'




# Load our model with load_model
model = xgb.XGBClassifier ()
model.load_model("pipeline_housing.json")
#model = pickle.load(open(path2+"pipeline_housing.pkl", "rb"))
print (model)
# Load data test sample
df_test = pd.read_csv(path)

df_test = df_test.loc[:, ~df_test.columns.str.match ('Unnamed')]
df_test_normalize = pd.read_csv(path2, index_col=0)

# df_test_normalize['SK_ID_CURR'] = df_test['SK_ID_CURR']
df_test = df_test.sort_index ()

# Define the threshold of for application
threshold = 0.9

# defining flask pages
app = flask.Flask (__name__)
app.config["DEBUG"] = True


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='assets/favicon.ico')
# defining home page
@app.route ('/', methods=['GET'])
def home():
    return "<h1>My first Flask API</h1><p>This site is a prototype API \
    for home risk project 7 of OpenClassRooms DataScientist training.</p>"


# defining page for the results of a prediction via index
@app.route ('/scores', methods=['GET'])
def predict():
    # get the index from a request
    print (type (flask.request.args.get ('index')))
    if type (flask.request.args.get ('index')) is None:
        data_index = '100030'
    else:
        data_index = flask.request.args.get ('index')
    print ('data_index_api', data_index)

    print ('df_test_normalize', df_test_normalize)
    # get inputs features from the data with index
    df_client = df_test_normalize[df_test_normalize.index == int (data_index)]
    data = df_client.to_json ()

    score = model.predict_proba (df_client)[:, 1]

    df_normalize = df_test_normalize.copy ()
    df_proba = pd.DataFrame (model.predict_proba (df_normalize)[:, 1], columns=['proba'],
                             index=df_normalize.index.tolist ())

    df_proba['Predict'] = np.where (df_proba['proba'] < threshold, 0, 1)

    df_normalize['Proba_Score'] = df_proba['proba']
    df_normalize['Predict'] = df_proba['Predict']

    df_test_new_normalize = df_normalize.to_json ()

    #  JSON format!

    dict_result = { 'Credit_score': score[0], "json_data": data, 'Total_score': df_test_new_normalize }

    class NumpyFloatValuesEncoder (json.JSONEncoder):
        def default(self, obj):
            if isinstance (obj, np.float32):
                return float (obj)
            return json.JSONEncoder.default (self, obj)

    dict_result = json.dumps (dict_result, cls=NumpyFloatValuesEncoder)

    df_normalize.drop (['Proba_Score', 'Predict'], axis=1)

    return dict_result

    # define endpoint for Flask


app.add_url_rule ('/scores', 'scores', predict)

app.run (port=5002, debug=False)
