import numpy as np
import pandas as pd
import yaml

# loading the config file
with open("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

from utils.montoring_ml_model_evidently import *
current_data = pd.read_csv(cfg['location']['yesterday_data_predictions'])
refernce_data =  pd.read_csv(cfg['location']['trained_model_predictions'])
print('current data', current_data.head())
print('refernce data', refernce_data.head())
target = 'quality'
prediction = 'prediction'
numerical_columns = refernce_data.select_dtypes(include=np.number).columns.tolist()
drift = drift_model_monitoring(current_data,refernce_data,target,prediction,numerical_columns,None)
drift.numerical_target_data_drift()
total_features,drifted_features,drift_percentage,target_drift = drift.num_target_data_drift_detection()
target_drift_happened = target_drift <0.05
print(f'data drift information total_features : {total_features}, drifted_features_count : {drifted_features} , percentage_of_features_drifted: {drift_percentage},target_drift_happened : {target_drift_happened}')
drift.regression_model_monitor()
mean_error_deviation, mean_abs_error_deviation, mean_abs_perc_error_deviation = drift.regression_model_drift_detection()
print(f'mean error deviation : {mean_error_deviation} , mean_abs_error_deviation: {mean_abs_error_deviation}, mean_absolute_percentage_error_deviation : {mean_abs_perc_error_deviation}')
if mean_error_deviation < 10 or mean_abs_error_deviation < 10 or mean_abs_perc_error_deviation < 10 or target_drift_happened or drift_percentage < 10:
 print('model is drifted')
