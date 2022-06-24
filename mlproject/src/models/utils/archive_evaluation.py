from sklearn import metrics
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
MetricList = []
flag = 0


# classfication metrics
class ClassificationMetrics:
    def __init__(self, package, average, metric_name=None, beta=None, pred_prob=None, ):
        self.y_true = package[0]
        self.y_pred = package[1]
        self.beta = beta
        self.average = average
        global flag
        flag = 1
        if metric_name is None:
            self._run_all()
        elif metric_name == 'Accuracy':
            self._accuracy(self.y_true, self.y_pred)
        elif metric_name == 'Precision':
            self._precision(self.y_true, self.y_pred)
        elif metric_name == 'Recall':
            self._recall(self.y_true, self.y_pred)
        elif metric_name == 'F1 Score':
            self._f1_score(self.y_true, self.y_pred)
        elif metric_name == 'F-Beta Score':
            if beta is not None:
                self._fbeta_score(self.y_true, self.y_pred, beta)
            else:
                print('\033[1m' + '\nF-Beta Score:' + '\033[1m' + '\nBeta Value Required for F-Beta score')
                f_beta = ['F-Beta Score', 'Beta Value Required']
                MetricList.append(f_beta)
        elif metric_name == 'AUC Score':
            self._auc(self.y_true, self.y_pred)
        elif metric_name == 'Matthews CorrCoef':
            self._matthews_corrcoef(self.y_true, self.y_pred)
        elif metric_name == 'Hamming Loss':
            self._hamming_loss(self.y_true, self.y_pred)
        elif metric_name == 'Log Loss':
            self._logloss(self.y_true, self.y_pred)
        elif metric_name == 'Zero One Loss':
            self._zero_one_loss(self.y_true, self.y_pred)
        elif metric_name == 'Cohen Kappa Score':
            self._cohen_kappa_score(self.y_true, self.y_pred)
        elif metric_name == 'Gini Score':
            if pred_prob is not None:
                self._gini_score(pred_prob)
            else:
                display('Gini Score Required prediction probability')
        else:
            display('Invalid Metric Name')
            flag = 0

    def _run_all(self):
        self._accuracy(self.y_true, self.y_pred)
        self._precision(self.y_true, self.y_pred, self.average)
        self._recall(self.y_true, self.y_pred, self.average)
        self._f1_score(self.y_true, self.y_pred, self.average)
        self._auc(self.y_true, self.y_pred)
        self._matthews_corrcoef(self.y_true, self.y_pred)
        self._hamming_loss(self.y_true, self.y_pred)
        self._logloss(self.y_true, self.y_pred)
        self._zero_one_loss(self.y_true, self.y_pred)
        self._cohen_kappa_score(self.y_true, self.y_pred)

    @staticmethod
    def _gini_score(pred_prob):
        # Logic For Gini Score
        sum_pp = sum(pred_prob)
        val = sum_pp / sum(sum_pp)
        gini = ['Gini Score', 1 - sum(val * val), '0']
        MetricList.append(gini)

    @staticmethod
    def _accuracy(y_true, y_pred):

        accuracy = ['Accuracy', metrics.accuracy_score(y_true, y_pred), '1']
        MetricList.append(accuracy)

    @staticmethod
    def _precision(y_true, y_pred, average):

        precision = ['Precision', metrics.precision_score(y_true, y_pred, average=average, zero_division=0), '1']
        MetricList.append(precision)

    @staticmethod
    def _recall(y_true, y_pred, average):

        recall = ['Recall', metrics.recall_score(y_true, y_pred, average=average), '1']
        MetricList.append(recall)

    @staticmethod
    def _f1_score(y_true, y_pred, average):

        f1_score = ['F1 Score', metrics.f1_score(y_true, y_pred, average=average), '1']
        MetricList.append(f1_score)

    @staticmethod
    def _fbeta_score(y_true, y_pred, beta):

        f_beta = ['F-Beta Score', metrics.fbeta_score(y_true, y_pred, beta=beta), '1']
        MetricList.append(f_beta)

    @staticmethod
    def _auc(y_true, y_pred):

        auc = ['AUC Score', metrics.roc_auc_score(y_true, y_pred, multi_class='ovr'), '1']
        MetricList.append(auc)

    @staticmethod
    def _classification_report(y_true, y_pred):

        class_report = ['Classification Report', metrics.classification_report(y_true, y_pred)]
        MetricList.append(class_report)

    @staticmethod
    def _matthews_corrcoef(y_true, y_pred):

        matthews = ['Matthews CorrCoef', metrics.matthews_corrcoef(y_true, y_pred), '1']
        MetricList.append(matthews)

    @staticmethod
    def _hamming_loss(y_true, y_pred):

        hamming_loss = ['Hamming Loss', metrics.hamming_loss(y_true, y_pred), '0']
        MetricList.append(hamming_loss)

    @staticmethod
    def _logloss(y_true, y_pred):

        log_loss = ['Log Loss', metrics.log_loss(y_true, y_pred), '0']
        MetricList.append(log_loss)

    @staticmethod
    def _zero_one_loss(y_true, y_pred):

        zero_one_loss = ['Zero One Loss', metrics.zero_one_loss(y_true, y_pred), '0']
        MetricList.append(zero_one_loss)

    @staticmethod
    def _cohen_kappa_score(y_true, y_pred):
        cohen = ['Cohen Kappa Score', metrics.cohen_kappa_score(y_true, y_pred), '1']
        MetricList.append(cohen)


# class for regression Metrics
class RegressionMetrics:
    def __init__(self, package, metric_name=None):
        self.y_true = package[0]
        self.y_pred = package[1]
        global flag
        flag = 1
        if metric_name is None:
            self._run_all()
        elif metric_name == 'Explained Variance Score':
            self._explained_variance_score(self.y_true, self.y_pred)
        elif metric_name == 'R2 Score':
            self._r2_score(self.y_true, self.y_pred)
        elif metric_name == 'Mean Absolute Error':
            self._mean_absolute_error(self.y_true, self.y_pred)
        elif metric_name == 'Mean Absolute Percentage Error':
            self._mean_absolute_percentage_error(self.y_true, self.y_pred)
        elif metric_name == 'Median Absolute Error':
            self._median_absolute_error(self.y_true, self.y_pred)
        elif metric_name == 'Mean Squared Error':
            self._mean_squared_error(self.y_true, self.y_pred)
        elif metric_name == 'Root Mean Squared Error':
            self._root_mean_squared_error(self.y_true, self.y_pred)
        elif metric_name == 'Mean Squared Log Error':
            self._mean_squared_log_error(self.y_true, self.y_pred)
        elif metric_name == 'Root Mean Squared Log Error':
            self._root_mean_squared_log_error(self.y_true, self.y_pred)
        elif metric_name == 'Max Error':
            self._max_error(self.y_true, self.y_pred)
        else:
            display('Invalid Metric Name')
            flag = 0

    def _run_all(self):
        self._explained_variance_score(self.y_true, self.y_pred)
        self._r2_score(self.y_true, self.y_pred)
        self._mean_absolute_error(self.y_true, self.y_pred)
        self._mean_absolute_percentage_error(self.y_true, self.y_pred)
        self._median_absolute_error(self.y_true, self.y_pred)
        self._mean_squared_error(self.y_true, self.y_pred)
        self._root_mean_squared_error(self.y_true, self.y_pred)
        self._mean_squared_log_error(self.y_true, self.y_pred)
        self._root_mean_squared_log_error(self.y_true, self.y_pred)
        self._max_error(self.y_true, self.y_pred)

    @staticmethod
    def _explained_variance_score(y_true, y_pred):
        op = ['Explained Variance Score', metrics.explained_variance_score(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _r2_score(y_true, y_pred):
        op = ['R2 Score', metrics.r2_score(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _mean_absolute_error(y_true, y_pred):
        op = ['Mean Absolute Error', metrics.mean_absolute_error(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _mean_absolute_percentage_error(y_true, y_pred):
        x = np.abs(y_true - y_pred) / np.abs(y_true)
        x = (100 / len(y_true)) * x
        x = np.mean(x)
        x = str(x)
        op = ['Mean Absolute Percentage Error', x, '0']
        MetricList.append(op)

    @staticmethod
    def _median_absolute_error(y_true, y_pred):
        op = ['Median Absolute Error', metrics.median_absolute_error(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _mean_squared_error(y_true, y_pred):
        op = ['Mean Squared Error', metrics.mean_squared_error(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _root_mean_squared_error(y_true, y_pred):
        op = ['Root Mean Squared Error', np.sqrt(metrics.mean_squared_error(y_true, y_pred)), '0']
        MetricList.append(op)

    @staticmethod
    def _mean_squared_log_error(y_true, y_pred):
        y_true = np.sqrt(y_true * y_true)
        y_pred = np.sqrt(y_pred * y_pred)
        op = ['Mean Squared Log Error', metrics.explained_variance_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _root_mean_squared_log_error(y_true, y_pred):
        y_true = np.sqrt(y_true * y_true)
        y_pred = np.sqrt(y_pred * y_pred)
        op = ['Root Mean Squared Log Error', np.sqrt(metrics.mean_squared_log_error(y_true, y_pred)), '0']
        MetricList.append(op)

    @staticmethod
    def _max_error(y_true, y_pred):
        op = ['Max Error', metrics.max_error(y_true, y_pred), '0']
        MetricList.append(op)


# class for the clustering metrics
class ClusteringMetrics:
    def __init__(self, package, samples, metric_name=None):
        self.y_true = package[0]
        self.y_pred = package[1]
        self.samples = samples
        global flag
        flag = 1

        if metric_name is None:
            self._run_all()
        elif metric_name == 'Mutual Info Score':
            self._mutual_info_score(self.y_true, self.y_pred)
        elif metric_name == 'Normalized Mutual Info Score':
            self._normalized_mutual_info_score(self.y_true, self.y_pred)
        elif metric_name == 'Adjusted Mutual Info Score':
            self._adjusted_mutual_info_score(self.y_true, self.y_pred)
        elif metric_name == 'Adjusted Rand Score':
            self._adjusted_rand_score(self.y_true, self.y_pred)
        elif metric_name == 'Fowlkes Mallows Score':
            self._fowlkes_mallows_score(self.y_true, self.y_pred)
        elif metric_name == 'Homogeneity Score':
            self._homogeneity_score(self.y_true, self.y_pred)
        elif metric_name == 'Completeness Score':
            self._completeness_score(self.y_true, self.y_pred)
        elif metric_name == 'V Measure Score':
            self._v_measure_score(self.y_true, self.y_pred)
        elif metric_name == 'Silhouette Score':
            self._silhouette_score(self.samples, self.y_pred)
        elif metric_name == 'Silhouette Sample':
            self._silhouette_samples(self.samples, self.y_pred)
        elif metric_name == 'Davies Bouldin Score':
            self._davies_bouldin_score(self.samples, self.y_pred)
        elif metric_name == 'Calinski Harabasz Score':
            self._calinski_harabasz_score(self.samples, self.y_pred)
        else:
            display('Invalid Metric Name')
            flag = 0

    def _run_all(self):
        self._mutual_info_score(self.y_true, self.y_pred)
        self._normalized_mutual_info_score(self.y_true, self.y_pred)
        self._adjusted_mutual_info_score(self.y_true, self.y_pred)
        self._adjusted_rand_score(self.y_true, self.y_pred)
        self._fowlkes_mallows_score(self.y_true, self.y_pred)
        self._homogeneity_score(self.y_true, self.y_pred)
        self._completeness_score(self.y_true, self.y_pred)
        self._v_measure_score(self.y_true, self.y_pred)
        self._silhouette_score(self.samples, self.y_pred)
        self._davies_bouldin_score(self.samples, self.y_pred)
        self._calinski_harabasz_score(self.samples, self.y_pred)

    @staticmethod
    def _silhouette_score(y_true, y_pred):
        op = ['Silhouette Score', metrics.silhouette_score(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _silhouette_samples(y_true, y_pred):
        op = ['Silhouette Sample', metrics.silhouette_samples(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _mutual_info_score(y_true, y_pred):
        op = ['Mutual Info Score', metrics.mutual_info_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _normalized_mutual_info_score(y_true, y_pred):
        op = ['Normalized Mutual Info Score', metrics.normalized_mutual_info_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _adjusted_mutual_info_score(y_true, y_pred):
        op = ['Adjusted Mutual Info Score', metrics.adjusted_mutual_info_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _adjusted_rand_score(y_true, y_pred):
        op = ['Adjusted Rand Score', metrics.adjusted_rand_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _fowlkes_mallows_score(y_true, y_pred):
        op = ['Fowlkes Mallows Score', metrics.fowlkes_mallows_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _homogeneity_score(y_true, y_pred):
        op = ['Homogeneity Score', metrics.homogeneity_score(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _completeness_score(y_true, y_pred):
        op = ['Completeness Score', metrics.completeness_score(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _v_measure_score(y_true, y_pred):
        op = ['V Measure Score', metrics.v_measure_score(y_true, y_pred), '1']
        MetricList.append(op)

    @staticmethod
    def _davies_bouldin_score(y_true, y_pred):
        op = ['Davies Bouldin Score', metrics.davies_bouldin_score(y_true, y_pred), '0']
        MetricList.append(op)

    @staticmethod
    def _calinski_harabasz_score(y_true, y_pred):
        op = ['Calinski Harabasz Score', metrics.calinski_harabasz_score(y_true, y_pred), 'Higher the better']
        MetricList.append(op)


# function to activate the classes for the specific Machine Learning type

def EvalMetric(problem_type=None, y_test=None, y_pred=None, idealFlag=0, metricName=None, sample=None, beta=None,
               pred_prob=None, average='macro'):
    MetricList.clear()
    if problem_type is None or y_test is None or y_pred is None:
        display('problem_type,y_test and y_pred is Mandatory Paramater' + 'Some Parameter Missing')

    if y_test.size == y_pred.size:
        package = [y_test, y_pred]

        if problem_type == 'classification':
            ClassificationMetrics(package, average, metricName, beta, pred_prob)
        elif problem_type == 'regression':
            RegressionMetrics(package, metric_name=metricName)

        elif problem_type == 'clustering':
            ClusteringMetrics(package, sample, metric_name=metricName)

        if flag == 1:
            df = pd.DataFrame(MetricList, columns=['Metrics', 'Score', 'Ideal Values'])
            if idealFlag == 1:
                df = df.drop(['Ideal Values'], axis=1)
            return (df)

    else:
        print('y_pred and y_test should be of same size')
