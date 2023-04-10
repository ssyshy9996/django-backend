import pandas as pd
import json
import os
from rest_framework.views import APIView
from ...models import CustomUser, Scenario, ScenarioSolution, Score
from ...views import BASE_DIR
from rest_framework.response import Response
from ..public import save_files_return_paths
from sklearn import metrics
import numpy as np
import tensorflow as tf
from apis.apiback.ui.dashboard import unsupervised_FinalScore, finalScore_supervised
from ..public import save_scores, get_score, save_metric, save_property, get_metric, get_property
from algorithms.supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised
from algorithms.supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised
from algorithms.supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised
from algorithms.supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised
from algorithms.unsupervised.Functions.Accountability.Accountability import analyse as get_accountability_score_unsupervised
from algorithms.unsupervised.Functions.Explainability.Explainability import analyse as get_explainability_score_unsupervised
from algorithms.unsupervised.Functions.Fairness.Fairness import analyse as get_fairness_score_unsupervised
from algorithms.unsupervised.Functions.Robustness.Robustness import analyse as get_robustness_score_unsupervised


def get_performance_metrics_supervised(model, test_data, target_column):
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    import sklearn.metrics as metrics
    model = pd.read_pickle(model)
    print("MODEL workign")
    print("TEST DATA working")
    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
    else:
        X_test = test_data.iloc[:, :DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_data.reset_index(
            drop=True).iloc[:, DEFAULT_TARGET_COLUMN_INDEX:]

    y_true = y_test.values.flatten()
    if (isinstance(model, tf.keras.Sequential)):
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test).flatten()
    labels = np.unique(np.array([y_pred, y_true]).flatten())

    dict_performance_metrics = {"accuracy": round(metrics.accuracy_score(y_true, y_pred), 2),
                                "global recall": round(metrics.recall_score(y_true, y_pred, labels=labels, average="micro"), 2),
                                "class weighted recall": round(metrics.recall_score(y_true, y_pred, average="weighted"), 2),
                                "global precision": round(metrics.precision_score(y_true, y_pred, labels=labels, average="micro"), 2),
                                "class weighted precision": round(metrics.precision_score(y_true, y_pred, average="weighted"), 2),
                                "global f1 score": round(metrics.f1_score(y_true, y_pred, average="micro"), 2),
                                "class weighted f1 score": round(metrics.f1_score(y_true, y_pred, average="weighted"), 2)}

    return dict_performance_metrics


def get_properties_section(train_data, test_data, factsheet):
    if "properties" in factsheet:
        factsheet = factsheet["properties"]

        properties = pd.DataFrame({
            "Model Type": [factsheet["explainability"]["algorithm_class"]["clf_name"][1]],
            "Train Test Split": [factsheet["methodology"]["train_test_split"]["train_test_split"][1]],
            "Train / Test Data Size": str(train_data.shape[0]) + " samples / " + str(test_data.shape[0]) + " samples",
            "Regularization Technique": [factsheet["methodology"]["regularization"]["regularization_technique"][1]],
            "Normalization Technique": [factsheet["methodology"]["normalization"]["normalization"][1]],
            "Number of Features": [factsheet["explainability"]["model_size"]["n_features"][1]],
        })
        properties = properties.transpose()
        properties = properties.reset_index()
        properties['index'] = properties['index'].str.title()
        properties.rename(columns={"index": "key", 0: "value"}, inplace=True)
        return properties
    return None


def get_properties_section_unsupervised(train_data, test_data, factsheet):
    if "properties" in factsheet:
        factsheet = factsheet["properties"]

        properties = pd.DataFrame({
            # "Model Type": [factsheet["explainability"]["algorithm_class"]["clf_name"][1]],
            "Train Test Split": [factsheet["methodology"]["train_test_split"]["train_test_split"][1]],
            "Train / Test Data Size": str(train_data.shape[0]) + " samples / " + str(test_data.shape[0]) + " samples",
            "Regularization Technique": [factsheet["methodology"]["regularization"]["regularization_technique"][1]],
            "Normalization Technique": [factsheet["methodology"]["normalization"]["normalization"][1]],
            "Number of Features": [factsheet["explainability"]["model_size"]["n_features"][1]],
        })
        properties = properties.transpose()
        properties = properties.reset_index()
        properties['index'] = properties['index'].str.title()
        properties.rename(columns={"index": "key", 0: "value"}, inplace=True)
        return properties
    return None


DEFAULT_TARGET_COLUMN_INDEX = -1


def get_performance_metrics(uploaddic, model, test_data, target_column):

    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
    else:
        X_test = test_data.iloc[:, :DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_data.reset_index(
            drop=True).iloc[:, DEFAULT_TARGET_COLUMN_INDEX:]
        y_true = y_test.values.flatten()
    if (isinstance(model, tf.keras.Sequential)):
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test).flatten()
        labels = np.unique(np.array([y_pred, y_true]).flatten())


def get_factsheet_completeness_score(factsheet):
    import pandas as pd
    print("FACTSHEET: ", factsheet)
    modelname = factsheet['properties']['methodology']['factsheet_completeness']['model_name'][1]
    purposedesc = factsheet['properties']['methodology']['factsheet_completeness']['purpose_description'][1]
    trainingdatadesc = factsheet['properties']['methodology']['factsheet_completeness']['training_data_description'][1]
    modelinfo = factsheet['properties']['methodology']['factsheet_completeness']['model_information'][1]
    authors = factsheet['properties']['methodology']['factsheet_completeness']['authors'][1]
    contactinfo = factsheet['properties']['methodology']['factsheet_completeness']['contact_information'][1]

    dict_factsheet = {'modelname': modelname, 'purposedesc': purposedesc, 'trainingdatadesc': trainingdatadesc,
                      'modelinfo': modelinfo, 'authors': authors, 'contactinfo': contactinfo}
    return dict_factsheet


def save_score(solution):
    uploaddic = {}

    def get_factsheet_completeness_score(factsheet):
        import collections
        info = collections.namedtuple('info', 'description value')
        result = collections.namedtuple(
            'result', 'score properties')

        factsheet = save_files_return_paths(factsheet)[0]
        with open(factsheet, 'r') as g:
            factsheet = json.loads(g.read())

        score = 0
        properties = {"dep": info('Depends on', 'Factsheet')}
        GENERAL_INPUTS = ["model_name", "purpose_description", "domain_description",
                          "training_data_description", "model_information", "authors", "contact_information"]

        n = len(GENERAL_INPUTS)
        ctr = 0
        for e in GENERAL_INPUTS:
            if "general" in factsheet and e in factsheet["general"]:
                ctr += 1
                properties[e] = info("Factsheet Property {}".format(
                    e.replace("_", " ")), "present")
            else:
                properties[e] = info("Factsheet Property {}".format(
                    e.replace("_", " ")), "missing")
                score = round(ctr/n*5)

        return result(score=score, properties=properties)

    path_factsheet = f"{solution.factsheet_file}"

    completeness_prop = get_factsheet_completeness_score(
        path_factsheet)

    uploaddic['modelname'] = completeness_prop[1]['model_name'][1]
    uploaddic['purposedesc'] = completeness_prop[1]['purpose_description'][1]
    uploaddic['trainingdatadesc'] = completeness_prop[1]['training_data_description'][1]
    uploaddic['modelinfo'] = completeness_prop[1]['model_information'][1]
    uploaddic['authors'] = completeness_prop[1]['authors'][1]
    uploaddic['contactinfo'] = completeness_prop[1]['contact_information'][1]

    path_testdata = solution.test_file
    path_module = solution.model_file
    path_traindata = solution.training_file
    path_factsheet = solution.factsheet_file
    path_outliersdata = solution.outlier_data_file
    target_column = solution.target_column
    soulutionType = solution.solution_type
    weights_metrics = solution.weights_metrics
    weights_pillars = solution.weights_pillars
    mappings_config = solution.metrics_mappings_file

    path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata, weights_metrics, weights_pillars, mappings_config, target_column = save_files_return_paths(
        f"{path_module}", f"{path_traindata}", f"{path_testdata}", f"{path_factsheet}", f"{path_outliersdata}", f"{weights_metrics}", f"{weights_pillars}", f"{mappings_config}", f"{target_column}")

    # -- get properties for pillar
    explain = {}
    fairness = {}
    robust = {}
    methodology = {}
    if (solution.solution_type == 'supervised'):
        explain = get_explainability_score_supervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        robust = get_robustness_score_supervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        methodology = get_accountability_score_supervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        fairness = get_fairness_score_supervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        pass
    else:
        fairness = get_fairness_score_unsupervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        explain = get_explainability_score_unsupervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        methodology = get_accountability_score_unsupervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)
        robust = get_robustness_score_unsupervised(
            path_module, path_traindata, path_testdata, path_factsheet, mappings_config, target_column, path_outliersdata)

    # for fairness
    try:
        uploaddic['underfitting_property'] = fairness[1]['underfitting']
    except:
        uploaddic['underfitting_property'] = None
    try:
        uploaddic['overfitting_property'] = fairness[1]['overfitting']
    except:
        uploaddic['overfitting_property'] = None
    try:
        uploaddic['statistical_parity_difference_property'] = fairness[1]['statistical_parity_difference']
    except:
        uploaddic['statistical_parity_difference_property'] = None
    try:
        uploaddic['equal_opportunity_difference_property'] = fairness[1]['equal_opportunity_difference']
    except:
        uploaddic['equal_opportunity_difference_property'] = None
    try:
        uploaddic['average_odds_difference_property'] = fairness[1]['average_odds_difference']
    except:
        uploaddic['average_odds_difference_property'] = None
    try:
        uploaddic['disparate_impact_property'] = fairness[1]['disparate_impact']
    except:
        uploaddic['disparate_impact_property'] = None
    try:
        uploaddic['class_balance_property'] = fairness[1]['class_balance']
    except:
        uploaddic['class_balance_property'] = None
    # for explainability
    try:
        uploaddic['algorithm_class_property'] = explain[1]['algorithm_class']
    except:
        uploaddic['algorithm_class_property'] = None
    try:
        uploaddic['correlated_features_property'] = explain[1]['correlated_features']
    except:
        uploaddic['correlated_features_property'] = None
    try:
        uploaddic['model_size_property'] = explain[1]['model_size']
    except:
        uploaddic['model_size_property'] = None
    try:
        uploaddic['feature_relevance_property'] = explain[1]['feature_relevance']
    except:
        uploaddic['feature_relevance_property'] = None
    try:
        uploaddic['permutation_feature_importance_property'] = explain[1]['permutation_feature_importance']
    except:
        uploaddic['permutation_feature_importance_property'] = None
    # for methodology
    try:
        uploaddic['normalization_property'] = methodology[1]['normalization']
    except:
        uploaddic['normalization_property'] = None
    try:
        uploaddic['missing_data_property'] = methodology[1]['missing_data']
    except:
        uploaddic['missing_data_property'] = None
    try:
        uploaddic['regularization_property'] = methodology[1]['regularization']
    except:
        uploaddic['regularization_property'] = None
    try:
        uploaddic['train_test_split_property'] = methodology[1]['train_test_split']
    except:
        uploaddic['train_test_split_property'] = None
    try:
        uploaddic['factsheet_completeness_property'] = methodology[1]['factsheet_completeness']
    except:
        uploaddic['factsheet_completeness_property'] = None

    # for robustness
    try:
        uploaddic['confidence_score_property'] = robust[1]['confidence_score']
    except:
        uploaddic['confidence_score_property'] = None
    try:
        uploaddic['clique_method_property'] = robust[1]['clique_method']
    except:
        uploaddic['clique_method_property'] = None
    try:
        uploaddic['clever_score_property'] = robust[1]['clever_score']
    except:
        uploaddic['clever_score_property'] = None
    try:
        uploaddic['er_fast_gradient_attack_property'] = robust[1]['er_fast_gradient_attack']
    except:
        uploaddic['er_fast_gradient_attack_property'] = None
    try:
        uploaddic['er_carlini_wagner_attack_property'] = robust[1]['er_carlini_wagner_attack']
    except:
        uploaddic['er_carlini_wagner_attack_property'] = None
    try:
        uploaddic['er_deepfool_attack_property'] = robust[1]['er_deepfool_attack']
    except:
        uploaddic['er_deepfool_attack_property'] = None
    try:
        uploaddic['loss_sensitivity_property'] = robust[1]['loss_sensitivity']
    except:
        uploaddic['loss_sensitivity_property'] = None

    # -- end
    test_data = pd.read_csv(path_testdata)
    try:
        uploaddic['accuracy'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["accuracy"]
        print("Accuracy: ", uploaddic['accuracy'])
    except:
        uploaddic['accuracy'] = None
        print("Accuracy ERROR")

    try:
        uploaddic['globalrecall'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["global recall"]
        print("Gloabl Recall is: ", uploaddic['globalrecall'])
    except:
        uploaddic['globalrecall'] = None
        print("Recall ERROR")
    try:
        uploaddic['classweightedrecall'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["class weighted recall"]
        print("Classweighted Recall is: ",
              uploaddic['classweightedrecall'])
    except:
        uploaddic['classweightedrecall'] = None
        print("Classweighted Recall ERROR")
    try:
        uploaddic['globalprecision'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["global precision"]
        print("Global Precision is: ", uploaddic['globalprecision'])
    except:
        uploaddic['globalprecision'] = None
        print("Global Precision ERROR")
    try:
        uploaddic['classweightedprecision'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["class weighted precision"]
        print("Classweighted Precision is:",
              uploaddic['classweightedprecision'])
    except:
        uploaddic['classweightedprecision'] = None
        print("Classweighted Precision ERROR")
    try:
        uploaddic['globalf1score'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["global f1 score"]
        print("Global F1 Score is:", uploaddic['globalf1score'])
    except:
        uploaddic['globalf1score'] = None
        print("Global F1 Score ERROR")
    try:
        uploaddic['classweightedf1score'] = get_performance_metrics_supervised(
            model=path_module, test_data=test_data, target_column="Target")["class weighted f1 score"]
        print("Classweightedf1score is:",
              uploaddic['classweightedf1score'])

    except:
        uploaddic['classweightedf1score'] = None
        print("Classweightedf1score- ERROR")

    try:
        if (soulutionType == 'unsupervised'):
            result = unsupervised_FinalScore(
                path_module, path_traindata, path_testdata, path_outliersdata, path_factsheet, mappings_config, weights_metrics, weights_pillars)
            uploaddic['disparate_impact'] = result['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
            uploaddic['underfitting'] = result['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
            uploaddic['overfitting'] = result['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
            uploaddic['statistical_parity_difference'] = result['Metricscores'][
                'Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
            uploaddic['fairness_score'] = result['Pillarscores']['Fairnessscore']
            uploaddic['normalization'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Normalizationscore']
            uploaddic['missing_data'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
            uploaddic['regularization'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Regularizationscore']
            uploaddic['train_test_split'] = result['Metricscores']['Metricscores']['Accountabilityscore']['Traintestsplitscore']
            uploaddic['factsheet_completeness'] = result['Metricscores'][
                'Metricscores']['Accountabilityscore']['Factsheecompletnessscore']
            uploaddic['methodology_score'] = result['Pillarscores']['Accountabilityscore']
            uploaddic['correlated_features'] = result['Metricscores']['Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
            uploaddic['permutation_feature_importance'] = result['Metricscores'][
                'Metricscores']['Explainabilityscore']['Permutationfeatureimportancescore']
            uploaddic['clever_score'] = result['Metricscores']['Metricscores']['Robustnessscore']['Cleverscore']
            uploaddic['model_size'] = result['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
            uploaddic['explainability_score'] = result['Pillarscores']['Explainabilityscore']
            uploaddic['robustness_score'] = result['Pillarscores']['Robustnessscore']
            uploaddic['trust_score'] = result['Trustscore']

            uploaddic['equal_opportunity_difference'] = None
            uploaddic['average_odds_difference'] = None
            uploaddic['class_balance'] = None
            uploaddic['feature_relevance'] = None
            uploaddic['algorithm_class'] = None
            uploaddic['confidence_score'] = None
            uploaddic['clique_method'] = None
            uploaddic['confidence_score'] = None
            uploaddic['er_fast_gradient_attack'] = None
            uploaddic['er_carlini_wagner_attack'] = None
            uploaddic['er_deepfool_attack'] = None
            uploaddic['loss_sensitivity'] = None

        else:
            resultSuper = finalScore_supervised(
                path_module, path_traindata, path_testdata, path_factsheet, mappings_config, weights_metrics, weights_pillars)

            uploaddic['fairness_score'] = resultSuper['Pillarscores']['Fairnessscore']
            try:
                uploaddic['methodology_score'] = resultSuper['Pillarscores']['Accountabilityscore']
            except:
                uploaddic['accountability_score'] = resultSuper['Pillarscores']['Accountabilityscore']

            uploaddic['trust_score'] = resultSuper['Trustscore']

            print('result super:', resultSuper['Pillarscores'])
            uploaddic['explainability_score'] = resultSuper['Pillarscores']['Explainabilityscore']
            uploaddic['robustness_score'] = resultSuper['Pillarscores']['Robustnessscore']
            uploaddic['underfitting'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
            uploaddic['overfitting'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
            uploaddic['statistical_parity_difference'] = resultSuper['Metricscores'][
                'Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
            uploaddic['equal_opportunity_difference'] = resultSuper['Metricscores'][
                'Metricscores']['Fairnessscore']['Equalopportunityscore']
            uploaddic['average_odds_difference'] = resultSuper['Metricscores'][
                'Metricscores']['Fairnessscore']['Averageoddsdifferencescore']
            uploaddic['disparate_impact'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
            uploaddic['class_balance'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Classbalancescore']
            uploaddic['algorithm_class'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Algorithmclassscore']
            uploaddic['correlated_features'] = resultSuper['Metricscores'][
                'Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
            uploaddic['model_size'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
            uploaddic['feature_relevance'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Featurerevelancescore']
            uploaddic['confidence_score'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Confidencescore']
            uploaddic['clique_method'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Cliquemethodscore']
            uploaddic['loss_sensitivity'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Losssensitivityscore']
            uploaddic['clever_score'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Cleverscore']
            uploaddic['er_fast_gradient_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Erfastgradientattack']
            uploaddic['er_carlini_wagner_attack'] = resultSuper['Metricscores'][
                'Metricscores']['Robustnessscore']['Ercarliniwagnerscore']
            uploaddic['er_deepfool_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Erdeepfoolattackscore']
            uploaddic['normalization'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Normalizationscore']
            uploaddic['missing_data'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
            uploaddic['regularization'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Regularizationscore']
            uploaddic['train_test_split'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Traintestsplitscore']
            uploaddic['factsheet_completeness'] = resultSuper['Metricscores'][
                'Metricscores']['Accountabilityscore']['Factsheecompletnessscore']
            uploaddic['permutation_feature_importance'] = None

        save_scores(solution.id, uploaddic)
        save_metric(solution.id, uploaddic)
        save_property(solution.id, uploaddic)

    except Exception as e:
        print('save score error:', e)


class analyze(APIView):
    def get(self, request, id):

        print("User not exist.... Created new")
        # return Response(uploaddic)

    def post(self, request):
        uploaddic = {}

        if request.data is not None:
            userexist = CustomUser.objects.get(
                email=request.data['emailid'])
            solution = ScenarioSolution.objects.get(
                user_id=userexist.id,
                solution_name=request.data['SelectSolution'])

            weights_metrics = solution.weights_metrics
            weights_pillars = solution.weights_pillars
            model_data = solution.model_file
            test_data = solution.test_file
            train_data = solution.training_file
            factsheet = solution.factsheet_file
            mappings = solution.metrics_mappings_file
            target_column = solution.target_column
            outliers_data = solution.outlier_data_file
            test_data, train_data, factsheet, model_data, weights_pillars, weights_metrics, mappings, target_column, outliers_data = save_files_return_paths(
                f"{test_data}", f"{train_data}", f"{factsheet}", f"{model_data}", f"{weights_pillars}", f"{weights_metrics}", f"{mappings}", f"{target_column}", f"{outliers_data}")

            test_data = pd.read_csv(test_data)
            train_data = pd.read_csv(train_data)
            factsheet = pd.read_json(factsheet)
            weights_metrics = pd.read_json(weights_metrics)
            weights_pillars = pd.read_json(weights_pillars)
            metricData = {
                'fairness': {},
                'explainability': {},
                'robustness': {},
                'methodology': {},
            }

            if (solution.solution_type == 'supervised'):
                metricData['fairness']['main'] = weights_pillars['pillars']['fairness']
                metricData['fairness']['underfitting'] = weights_metrics['fairness']['underfitting']
                metricData['fairness']['overfitting'] = weights_metrics['fairness']['overfitting']
                metricData['fairness']['statistical_parity_difference'] = weights_metrics['fairness']['statistical_parity_difference']
                metricData['fairness']['equal_opportunity_difference'] = weights_metrics['fairness']['equal_opportunity_difference']
                metricData['fairness']['average_odds_difference'] = weights_metrics['fairness']['average_odds_difference']
                metricData['fairness']['disparate_impact'] = weights_metrics['fairness']['disparate_impact']
                metricData['fairness']['class_balance'] = weights_metrics['fairness']['class_balance']

                metricData['explainability']['main'] = weights_pillars['pillars']['explainability']
                metricData['explainability']['algorithm_class'] = weights_metrics['explainability']['algorithm_class']
                metricData['explainability']['correlated_features'] = weights_metrics['explainability']['correlated_features']
                metricData['explainability']['model_size'] = weights_metrics['explainability']['model_size']
                metricData['explainability']['feature_relevance'] = weights_metrics['explainability']['feature_relevance']

                metricData['robustness']['main'] = weights_pillars['pillars']['robustness']
                metricData['robustness']['confidence_score'] = weights_metrics['robustness']['confidence_score']
                metricData['robustness']['clique_method'] = weights_metrics['robustness']['clique_method']
                metricData['robustness']['loss_sensitivity'] = weights_metrics['robustness']['loss_sensitivity']
                metricData['robustness']['clever_score'] = weights_metrics['robustness']['clever_score']
                metricData['robustness']['er_fast_gradient_attack'] = weights_metrics['robustness']['er_fast_gradient_attack']
                metricData['robustness']['er_carlini_wagner_attack'] = weights_metrics['robustness']['er_carlini_wagner_attack']
                metricData['robustness']['er_deepfool_attack'] = weights_metrics['robustness']['er_deepfool_attack']

                metricData['methodology']['main'] = weights_pillars['pillars']['accountability']
                try:
                    metricData['methodology']['normalization'] = weights_metrics['methodology']['normalization']
                except:
                    metricData['methodology']['normalization'] = weights_metrics['accountability']['normalization']
                try:
                    metricData['methodology']['missing_data'] = weights_metrics['methodology']['missing_data']
                except:
                    metricData['methodology']['missing_data'] = weights_metrics['accountability']['missing_data']
                try:
                    metricData['methodology']['regularization'] = weights_metrics['methodology']['regularization']
                except:
                    metricData['methodology']['regularization'] = weights_metrics['accountability']['regularization']
                try:
                    metricData['methodology']['train_test_split'] = weights_metrics['methodology']['train_test_split']
                except:
                    metricData['methodology']['train_test_split'] = weights_metrics['accountability']['train_test_split']
                try:
                    metricData['methodology']['factsheet_completeness'] = weights_metrics['methodology']['factsheet_completeness']
                except:
                    metricData['methodology']['factsheet_completeness'] = weights_metrics['accountability']['factsheet_completeness']
            else:
                metricData['fairness']['main'] = weights_pillars['pillars']['fairness']
                metricData['fairness']['underfitting'] = weights_metrics['fairness']['underfitting']
                metricData['fairness']['overfitting'] = weights_metrics['fairness']['overfitting']
                metricData['fairness']['statistical_parity_difference'] = weights_metrics['fairness']['statistical_parity_difference']
                metricData['fairness']['disparate_impact'] = weights_metrics['fairness']['disparate_impact']

                metricData['explainability']['main'] = weights_pillars['pillars']['explainability']
                metricData['explainability']['correlated_features'] = weights_metrics['explainability']['correlated_features']
                metricData['explainability']['model_size'] = weights_metrics['explainability']['model_size']
                metricData['explainability']['permutation_feature_importance'] = weights_metrics['explainability']['permutation_feature_importance']

                metricData['robustness']['main'] = weights_pillars['pillars']['robustness']
                metricData['robustness']['clever_score'] = weights_metrics['robustness']['clever_score']

                metricData['methodology']['main'] = weights_pillars['pillars']['accountability']
                metricData['methodology']['normalization'] = weights_metrics['accountability']['normalization']
                metricData['methodology']['missing_data'] = weights_metrics['accountability']['missing_data']
                metricData['methodology']['train_test_split'] = weights_metrics['accountability']['train_test_split']
                metricData['methodology']['factsheet_completeness'] = weights_metrics['accountability']['factsheet_completeness']

            uploaddic['weight'] = metricData

            uploaddic['ScenarioName'] = solution.solution_name
            uploaddic['Description'] = solution.description

            if (solution.solution_type == 'supervised'):
                data = get_properties_section(
                    train_data, test_data, factsheet)
                uploaddic['ModelType'] = data[data.columns[1]][0]
                uploaddic['TrainTestSplit'] = data[data.columns[1]][1]
                uploaddic['DataSize'] = data[data.columns[1]][2]
                uploaddic['RegularizationTechnique'] = data[data.columns[1]][3]
                uploaddic['NormalizationTechnique'] = data[data.columns[1]][4]
                uploaddic['NumberofFeatures'] = data[data.columns[1]][5]
            else:
                data = get_properties_section_unsupervised(
                    train_data, test_data, factsheet)
                uploaddic['ModelType'] = data[data.columns[1]][0]
                uploaddic['TrainTestSplit'] = data[data.columns[1]][1]
                uploaddic['DataSize'] = data[data.columns[1]][2]
                uploaddic['RegularizationTechnique'] = data[data.columns[1]][3]
                uploaddic['NormalizationTechnique'] = data[data.columns[1]][4]
                # uploaddic['NumberofFeatures'] = data[data.columns[1]][5]

            try:
                score_data = get_score(solution.id)
                property_data = get_property(solution.id)
                metric_data = get_metric(solution.id)
                uploaddic.update(score_data)
                uploaddic.update(property_data)
                uploaddic.update(metric_data)
                print('dat:', uploaddic)
                return Response(uploaddic, status=200)

            except Exception as e:
                print('which error?', e)
                save_score(solution)
                return Response(uploaddic)
