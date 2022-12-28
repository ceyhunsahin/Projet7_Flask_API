# Projet7_Flask_API

<h3> Description de projet </h3>
Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite développer un modèle de scoring de la probabilité de défaut de paiement du client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Elle décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

This repostory consist of making a scoring model of default payment for a loan company and deploy it on the web on an API.

The data used for this project can be find on kaggle here: https://www.kaggle.com/c/home-credit-default-risk/data

Here is a summary of folder available on it:

It contains all the developpment work for the model API (code + csv file + json files (for xgboost model) + requirements.txt)

First of all, it needs to open this [Flask API](https://oc-p7-home-risk-flaskapi.herokuapp.com/) link, then connect 
[Dashboard](https://oc-dashboard-home-risk.herokuapp.com/). Flask API gets all needed parameters and returns a json format for using in Dashboard.
So, without opening the Flask API, Dashboard won't handle the data for visualising.

The API has been deployed on Heroku and was available at these adresses. But beacuse of shutting down the free dynos and services 
by heroku, dashboard and flask api was deployed to the docker hub. 

For Docker, it can be type your shell:

`docker container run -p 8000:8000 csahin2086/dash_app`

and with another shell:

`docker container run -p 5000:5000 csahin2086/flask_app`

Please note that they also have had specific github repository for deployment on Heroku:

Python librairies needed for the API are the following: python = 3.10 pandas=1.4.3 numpy=1.22.4 scikit-learn=1.1.2 
plotly=5.10.0 shap=0.41.0 xgboost=1.6.1 pillow=9.2.0
