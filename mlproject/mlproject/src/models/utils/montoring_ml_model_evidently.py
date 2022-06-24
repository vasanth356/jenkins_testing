import pandas as pd
import json
import yaml
# loading the config file
with open("/home/vasanth/jenkins/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, NumTargetDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection, NumTargetDriftProfileSection
from evidently import ColumnMapping
from evidently.dashboard.tabs import RegressionPerformanceTab
from evidently.model_profile.sections import RegressionPerformanceProfileSection
from evidently.dashboard.tabs import ClassificationPerformanceTab
from evidently.model_profile.sections import ClassificationPerformanceProfileSection


class drift_model_monitoring:
    def __init__(self,current,reference,target,prediction,numerical_features,categorical_features,reference_probs=None,current_probs=None):
        self.reference=reference
        self.current=current
        self.target=target
        self.numerical_features=numerical_features
        self.categorical_features=categorical_features
        self.datetime_features=list(self.reference.select_dtypes(include=['datetime64[ns]']).columns) # change to refrence if error
        self.prediction=prediction
        self.reference_probs=reference_probs
        self.current_probs=current_probs
        if self.reference_probs is not None and self.current_probs is not None:
            self.merged_reference_data = pd.concat([self.reference, self.reference_probs], axis=1)
            self.merged_current_data = pd.concat([self.current, self.current_probs], axis=1)
        #merged referance and current data should be outside probablity model monitor 
        self.column_mapping = ColumnMapping(target=self.target,
                               prediction=self.prediction,
                               numerical_features=self.numerical_features,
                               datetime_features=self.datetime_features,
                               categorical_features=self.categorical_features)   
    def categorical_target_data_drift(self):
        data_drift= Dashboard(tabs=[DataDriftTab(),CatTargetDriftTab()])
        data_drift.calculate(self.reference, self.current, column_mapping=self.column_mapping)
        data_drift.save('reports/10.html')
        
        ''' The report returns the share of drifting features when the target column is categorical and to compare between reference dataset and current dataset 
        data drift can analysed using this dashboard'''
        
    def numerical_target_data_drift(self):
        data_drift= Dashboard(tabs=[NumTargetDriftTab(),DataDriftTab()])
        data_drift.calculate(self.reference, self.current, column_mapping=self.column_mapping)
        data_drift.save(cfg['location']['data_drift'])
        
        ''' The report returns the share of drifting features when the target column is Numerical and to compare between reference dataset and current dataset 
        data drift can analysed using this dashboard'''

        '''In absence of ground truth labels, you can monitor for changes in the input data. Use it e.g. to decide when to retrain the model, 
        apply business logic on top of the model output, or whether to act on predictions. You can combine it with monitoring model outputs 
        using the Numerical or Categorical Target Drift report from the two functions mentioned above.'''
        
    def cat_target_data_drift_detection(self):
        data_drift_profile = Profile(sections=[DataDriftProfileSection(),CatTargetDriftProfileSection()])
        data_drift_profile.calculate(self.reference,self.current, column_mapping = self.column_mapping)
        data_drift_report_dict = json.loads(data_drift_profile.json())
        total_features = data_drift_report_dict['data_drift']['data']['metrics']['n_features']
        drifted_features = data_drift_report_dict['data_drift']['data']['metrics']['n_drifted_features']
        target_drift=data_drift_report_dict['cat_target_drift']['data']['metrics']['target_drift']
        return total_features,drifted_features,(drifted_features/total_features)*100,target_drift
    
    '''The profiling calculate the same metrics as visual reports. You can think about 
        profiles as "JSON versions" of the Evidently dashboards.
        You will need two datasets. The reference dataset serves as a benchmark. We analyze the change by comparing the current production data to the reference data.
        You can potentially choose any two datasets for comparison. But keep in mind that only the reference dataset will be used as a basis for comparison.
        The requirements for the data inputs and column_mapping are the same for Profiles and Dashboards.'''
    
    def num_target_data_drift_detection(self):
        data_drift_profile = Profile(sections=[DataDriftProfileSection(),NumTargetDriftProfileSection()])
        data_drift_profile.calculate(self.reference,self.current, column_mapping = self.column_mapping)
        data_drift_report_dict = json.loads(data_drift_profile.json())
        total_features = data_drift_report_dict['data_drift']['data']['metrics']['n_features']
        drifted_features = data_drift_report_dict['data_drift']['data']['metrics']['n_drifted_features']
        target_drift=data_drift_report_dict['num_target_drift']['data']['metrics']['target_drift']
        return total_features,drifted_features,(drifted_features/total_features)*100,target_drift
    
    '''JSON profiles help integrate Evidently in your prediction pipelines. For example, you can log and 
        store JSON profiles for further analysis, or build a conditional workflow based on the result of the check 
        (e.g. to trigger alert, retraining, or generate a visual report).'''
        
    def regression_model_monitor(self):
        dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
        dashboard.calculate(self.reference,self.current, column_mapping=self.column_mapping)
        dashboard.save(cfg['location']['model_drift'])
        
        '''The Regression Performance report evaluates the quality of a regression model.
        It can also compare it to the past performance of the same model, or the performance of an alternative model.'''
        
    def classification_model_monitor(self):
        model_performance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
        model_performance_dashboard.calculate(self.reference,self.current, column_mapping=self.column_mapping)
        model_performance_dashboard.save('reports/test_2.html')
        
        '''Classification Performance report evaluates the quality of a classification model. 
            It works both for binary and multi-class classification.This report can be generated for a single model, or as a comparison. 
            You can contrast your current production model performance against the past or an alternative model.'''     
    
    def probability_model_monitor(self):
        self.reference.reset_index(inplace=True, drop=True)
        self.current.reset_index(inplace=True, drop=True)
        self.column_mapping.prediction = self.reference[self.target].unique().tolist()
        prob_classification_dashboard = Dashboard(tabs=[ProbClassificationPerformanceTab()])
        prob_classification_dashboard.calculate(self.merged_reference_data,self.merged_current_data,column_mapping =self.column_mapping)
        prob_classification_dashboard.save('reports/test_3.html')
            
        
    def classification_probability_drift_detection(self):
        model_prob_classification_profile = Profile(sections=[ProbClassificationPerformanceProfileSection()])
        model_prob_classification_profile.calculate(self.merged_reference_data,self.merged_current_data, 
                                           column_mapping =self.column_mapping)
        
        model_drift_report_dict = json.loads(model_prob_classification_profile.json())
        reference_accuracy=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['reference']['accuracy']
        current_accuracy=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['current']['accuracy']
        accuracy_deviation=(reference_accuracy-current_accuracy)*100/reference_accuracy
        reference_precision=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['reference']['precision']
        current_precision=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['current']['precision']
        precision_deviation=(reference_precision-current_precision)*100/reference_precision
        reference_recall=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['reference']['recall']
        current_recall=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['current']['recall']
        recall_deviation=(reference_recall-current_recall)*100/reference_recall
        reference_f1=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['reference']['f1']
        current_f1=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['current']['f1']
        f1_deviation=(reference_f1-current_f1)*100/reference_f1
        reference_roc_auc=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['reference']['roc_auc']
        current_refrence_roc_auc=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['current']['roc_auc']
        roc_auc_deviation=(reference_roc_auc-current_refrence_roc_auc)*100/reference_roc_auc
        reference_log_loss=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['reference']['log_loss']
        current_log_loss=model_drift_report_dict['probabilistic_classification_performance']['data']['metrics']['current']['log_loss']
        log_loss_deviation=(reference_log_loss-current_log_loss)*100/reference_log_loss
        return accuracy_deviation,precision_deviation,recall_deviation,f1_deviation,roc_auc_deviation,log_loss_deviation
    
    '''Probabilistic Classification Performance report evaluates the quality of a probabilistic classification model. 
        It works both for binary and multi-class classification.This report can be generated for a single model, or as 
        a comparison. You can contrast your current production model performance against the past or an alternative model.
        from the json file this function will fetch accuracy_deviation,precision_deviation,recall_deviation,f1_deviation,roc_auc_deviation,
        log_loss_deviation of reference and current data and we can compare it if any deviaton is present'''
        
    def classification_model_drift_detection(self):
        classification_performance_profile = Profile(sections=[ClassificationPerformanceProfileSection()])
        classification_performance_profile.calculate(self.reference,self.current, column_mapping =self.column_mapping)
        classification_drift_report_dict = json.loads(classification_performance_profile.json())
        reference_accuracy=classification_drift_report_dict['classification_performance']['data']['metrics']['reference']['accuracy']
        current_accuracy=classification_drift_report_dict['classification_performance']['data']['metrics']['current']['accuracy']
        accuracy_deviation=(reference_accuracy-current_accuracy)*100/reference_accuracy
        reference_precision=classification_drift_report_dict['classification_performance']['data']['metrics']['reference']['precision']
        current_precision=classification_drift_report_dict['classification_performance']['data']['metrics']['current']['precision']
        precision_deviation=(reference_precision-current_precision)*100/reference_precision
        reference_recall=classification_drift_report_dict['classification_performance']['data']['metrics']['reference']['recall']
        current_recall=classification_drift_report_dict['classification_performance']['data']['metrics']['current']['recall']
        recall_deviation=(reference_recall-current_recall)*100/reference_recall
        reference_f1=classification_drift_report_dict['classification_performance']['data']['metrics']['reference']['f1']
        current_f1=classification_drift_report_dict['classification_performance']['data']['metrics']['current']['f1']
        f1_deviation=(reference_f1-current_f1)*100/reference_f1
        return accuracy_deviation,precision_deviation,recall_deviation,f1_deviation
    
    '''From the reference model and current model this function will fetch accuracy_deviation,precision_deviation,
        recall_deviation,f1_deviation and we can compare between reference and current classification model'''
            
    def regression_model_drift_detection (self):
        regression_performance_profile = Profile(sections=[RegressionPerformanceProfileSection()])
        regression_performance_profile.calculate(self.reference, self.current, column_mapping=self.column_mapping)
        regression_report_dict = json.loads(regression_performance_profile.json())
        reference_mean_error=regression_report_dict['regression_performance']['data']['metrics']['reference']['mean_error']
        current_mean_error=regression_report_dict['regression_performance']['data']['metrics']['current']['mean_error']
        mean_error_deviation=(reference_mean_error-current_mean_error)*100/reference_mean_error
        
        reference_mean_abs_error=regression_report_dict['regression_performance']['data']['metrics']['reference']['mean_abs_error']
        current_mean_abs_error=regression_report_dict['regression_performance']['data']['metrics']['current']['mean_abs_error']
        mean_abs_error_deviation=(reference_mean_abs_error-current_mean_abs_error)*100/reference_mean_abs_error
        reference_mean_abs_perc_error=regression_report_dict['regression_performance']['data']['metrics']['reference'][ 'mean_abs_perc_error']
        current_mean_abs_perc_error=regression_report_dict['regression_performance']['data']['metrics']['current'][ 'mean_abs_perc_error']
        mean_abs_perc_error_deviation=(reference_mean_abs_perc_error-current_mean_abs_perc_error)*100/reference_mean_abs_perc_error
        return mean_error_deviation,mean_abs_error_deviation,mean_abs_perc_error_deviation
    
    '''From the reference model and current model this function will fetch mean_error_deviation,mean_abs_error_deviation,mean_abs_perc_error_deviation 
        and we can compare between reference and current regression model'''
        
        
def monitoring(problem_type=None,reference_data=None,current_data=None,target=None,prediction=None,numerical_features=None,
              categorical_features=None,ref_feature_probablity=None,current_feature_probablity=None):
              
    if problem_type=='regression':
        reg_drift=drift_model_monitoring(reference_data,current_data,target,prediction,numerical_features,categorical_features)
        mean_error_deviation,mean_abs_error_deviation,mean_abs_perc_error_deviation=reg_drift.regression_model_drift_detection()
        reg_drift.regression_model_monitor()
    elif problem_type=='classification':
        clas_drift=drift_model_monitoring(reference_data,current_data,target,prediction,numerical_features,categorical_features)
        accuracy_deviation,precision_deviation,recall_deviation,f1_deviation=clas_drift.classification_model_drift_detection()
        print(accuracy_deviation,precision_deviation,recall_deviation,f1_deviation)
        clas_drift.classification_model_monitor()
    elif problem_type=='classification_prob_score':
        clas_drift=drift_model_monitoring(reference_data,current_data,target,prediction,numerical_features,categorical_features,ref_feature_probablity,current_feature_probablity)
        accuracy_deviation,precision_deviation,recall_deviation,f1_deviation,roc_auc_deviation,log_loss_deviation=clas_drift.classification_probability_drift_detection()
        print(accuracy_deviation,precision_deviation,recall_deviation,f1_deviation,roc_auc_deviation,log_loss_deviation)
        clas_drift.probability_model_monitor()
        
        
#classification probability score 
# monitoring('classification_prob_score',train_data,test_data,'target',iris.target_names.tolist(),numerical_features,categorical_features,train_probas,test_probas)
# regression model monitoring 
# monitoring('regression',train_data,test_data,'target',numerical_features,categorical_features)
# classification model monitor 
# monitoring('classification_prob_score',train_data,test_data,'target',numerical_features,categorical_features)

