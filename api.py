import flask
import pandas as pd
import json
import numpy as np
import xgboost as xgb



# Load our model with load_model
model = xgb.XGBClassifier ()
model.load_model("pipeline_housing.json")

# Load data test sample
path = 'Projet_File/test_sample_data_home_risk.csv'
path2 = 'Projet_File/test_sample_data_home_risk_normalise.csv'
df_test = pd.read_csv(path)
df_test = df_test.sample(1200, random_state=42)
df_test = df_test.loc[:, ~df_test.columns.str.match ('Unnamed')]
# Load data test sample normalisée
df_test_normalize = pd.read_csv(path2, index_col=0)
df_test_normalize = df_test_normalize.sample(1200, random_state=42)
df_test = df_test.sort_index ()

# Define the threshold of for application
threshold = 0.9

# defining flask pages
app = flask.Flask (__name__)
app.config["DEBUG"] = True


# defining home page
@app.route ('/', methods=['GET'])
def home():
    return "<h1>My first Flask API</h1><p>This site is a prototype API \
    for home risk project 7 of OpenClassRooms DataScientist training.</p>"

# defining page for the results of a prediction via index
@app.route ('/scores', methods=['GET'])
def predict():
    # get the index from a request, defined a data_index parameter as default

    if type (flask.request.args.get ('index')) is None:
        data_index = '100111'
    else:
        data_index = flask.request.args.get ('index')

    # get inputs features from the data with index
    df_client = df_test_normalize[df_test_normalize.index == int (data_index)]
    data = df_client.to_json ()

    # predict_proba returns a list as [0,1], 0 -> for payments accepted, 1 -> for payments refused
    # we have chosen second parameter for refused value
    score = model.predict_proba (df_client)[:, 1]

    df_normalize = df_test_normalize.copy ()
    # for add probabilities, used normalized dataset
    df_proba = pd.DataFrame (model.predict_proba (df_normalize)[:, 1], columns=['proba'],
                             index=df_normalize.index.tolist ())
    # for add prediction, used threshold value
    df_proba['Predict'] = np.where (df_proba['proba'] < threshold, 0, 1)

    df_normalize['Proba_Score'] = df_proba['proba']
    df_normalize['Predict'] = df_proba['Predict']

    #  JSON format!
    df_test_new_normalize = df_normalize.to_json ()
    dict_result = { 'Credit_score': score[0], "json_data": data, 'Total_score': df_test_new_normalize }

    # for json format some values are categorical, however it is difficult to handle these values as float,
    # these values types are changed by using JSON encoder
    class NumpyFloatValuesEncoder (json.JSONEncoder):
        def default(self, obj):
            if isinstance (obj, np.float32):
                return float (obj)
            return json.JSONEncoder.default (self, obj)

    # JSON format dumps method for send the data to Dashboard
    dict_result = json.dumps (dict_result, cls=NumpyFloatValuesEncoder)

    # Each request of dashboard, df_normalize dataframe adding ['Proba_Score', 'Predict'] columns,
    # so It needs to drop these columns at the end of the API
    df_normalize.drop (['Proba_Score', 'Predict'], axis=1)

    return dict_result

    # define endpoint for Flask


app.add_url_rule ('/scores', 'scores', predict)
if __name__ == '__main__':
    app.run (host='localhost',port=5002, debug=True)

