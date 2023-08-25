#!/usr/bin/env python
# coding: utf-8



import sys

import joblib

sys.modules['sklearn.externals.joblib'] = joblib

joblib_file = "joblib_knn_model.pkl"  
joblib.dump(knn_model, joblib_file)

def forecast(data):
 joblib_knn_model = joblib.load(joblib_file)
 return joblib_knn_model.forecast(data)

