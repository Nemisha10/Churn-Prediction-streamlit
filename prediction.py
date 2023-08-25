#!/usr/bin/env python
# coding: utf-8



import sys

import joblib

sys.modules['sklearn.externals.joblib'] = joblib

joblib_file = "joblib_knn_model.pkl" 
joblib_knn_model = joblib.load(joblib_file)
joblib.dump(joblib_knn_model, joblib_file)

x_sample = [[80005, 56, 7, 71.54, 246, False, True],
       [80006, 40, 4, 33.71, 336, False, True]]

def forecast(data=x_sample):
 joblib_knn_model = joblib.load(joblib_file)
 return joblib_knn_model.predict(data)

