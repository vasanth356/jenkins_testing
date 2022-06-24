import yaml
from utils.evaluation import *
import pandas as pd
import joblib
from utils.mlflow_class import *
# metrics 
min_metrics = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Log Error',
                            'Mean Absolute Percentage Error', 'Mean Squared Log Error', 'Root Mean Squared Error'
                            ,'Median Absolute Error', 'Max Error', 'Hamming Loss', 'Log Loss', 'Zero One Loss'
                            ,'Davies Bouldin Score']

# metrics which need to be maximum for best model
max_metrics = ['Explained Variance Score', 'R2 Score', 'Gini Score', 'Accuracy',
                            'Precision','Recall', 'F1 Score','F-Beta Score', 'AUC Score'
                            ,'Matthews CorrCoef', 'Cohen Kappa Score', 'Silhouette Score', 'Silhouette Sample'
                            ,'Mutual Info Score', 'Normalized Mutual Info Score', 'Adjusted Mutual Info Score',
                            'Adjusted Rand Score', 'Fowlkes Mallows Score', 'Homogeneity Score', 'Completeness Score'
                            , 'V Measure Score', 'Calinski Harabasz Score']


# loading the config file
with open("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])

df = pd.read_csv(cfg['location']['today_data_with_target'])
print(df.columns)
target = cfg['model_parameters']['target']
features=df.columns
features=features.drop(target)
test_x = df[features]
test_y = df[target]

# getting the different versions of mlflow model
# loading the model it to a file
versions = []

for mv in client.search_model_versions(exper_name):
 versions.append(dict(mv))
print('all the versions', versions)
latest_version = versions[-1]['version']
old_version =versions[-2]['version']
print('latest_version is ', latest_version)
# loading the old model model registry
model_fetch = GettingModel(cfg['experiment_details']['name'],old_version)
model_mlflow = model_fetch.model()
print('model of old version', model_mlflow)
last_model_predictions = model_mlflow.predict(test_x)
output = EvalMetric(problem_type=  cfg['model_parameters']['problem_type'], y_test=test_y, y_pred=last_model_predictions, idealFlag=0, metricName=None, sample=None,beta=None,
                           pred_prob=None, average='macro')
# creating the output dictionary
output_dictionary_old_model = output.to_dict('dict')
metric_names = output_dictionary_old_model['Metrics']
metric_values = output_dictionary_old_model[ 'Score']
print('old model metric values', output_dictionary_old_model)


# loading the latest trained model from model registry
model_fetch = GettingModel(cfg['experiment_details']['name'], latest_version)
model_mlflow = model_fetch.model()
print('model of new version', model_mlflow)
latest_model_predictions = model_mlflow.predict(test_x)
output = EvalMetric(problem_type= cfg['model_parameters']['problem_type'], y_test=test_y, y_pred=latest_model_predictions, idealFlag=0, metricName=None, sample=None,beta=None,
                           pred_prob=None, average='macro')
 # creating the output dictionary
output_dictionary_new_model = output.to_dict('dict')
metric_names = output_dictionary_new_model['Metrics']
metric_values = output_dictionary_new_model[ 'Score']
print('new model metric values', output_dictionary_new_model)

def comparsion(first_dictonary, second_dictonary):
 first_score = 0
 metrics = []
 metric_names = first_dictonary['Metrics']
 metrics_values_first = first_dictonary[ 'Score']
 metrics_values_second = second_dictonary[ 'Score']
 for count in range(len(metric_names.keys())):
  if metric_names[count] in min_metrics:
   if metrics_values_first[count] < metrics_values_second[count]:
    first_score = first_score+1
    print('it is minimum metric')
    print('values are ', metrics_values_first[count],  metrics_values_second[count])
    print('name of metric', metric_names[count])
  if metric_names[count] in max_metrics:
   if metrics_values_first[count] > metrics_values_second[count]:
    first_score = first_score+1
    print('it is maximum metric')
    print('name of metric', metric_names[count])
    print('values are ', metrics_values_first[count],  metrics_values_second[count])
 return first_score
old_model_score = comparsion(output_dictionary_old_model,output_dictionary_new_model)
new_model_score = comparsion(output_dictionary_new_model,output_dictionary_old_model )

if old_model_score >= new_model_score:
 print('old is better')
else :
 print('new is better')
