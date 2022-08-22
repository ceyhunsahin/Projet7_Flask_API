# Projet7_Flask_API

This repository contains all the work for the project 7 of the OpenClassRooms DataScientist training. 

This repostory consist of making a scoring model of default payment for a loan company and deploy it on the web on an API.

The data used for this project can be find on kaggle here: https://www.kaggle.com/c/home-credit-default-risk/data

Here is a summary of folder available on it:

It contains all the developpment work for the model API (code + csv file + json files (for xgboost model) + requirements.txt)

First of all, it needs to open this [Flask API](https://oc-p7-home-risk-flaskapi.herokuapp.com/) link, then connect 
[Dashboard](https://oc-dashboard-home-risk.herokuapp.com/). Flask API gets all needed parameters and returns a json format for using Dashboard.
So, without opening the Flask API, Dashboard won't handle the data for visualising.

The API has been deployed on Heroku and are available at these adresses:

Please note that they also have specific github repository for deployment on Heroku:

Python librairies needed for the API are the following: python = 3.10 pandas=1.4.3 numpy=1.22.4 scikit-learn=1.1.2 
plotly=5.10.0 shap=0.41.0 xgboost=1.6.1 pillow=9.2.0
