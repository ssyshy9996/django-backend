import django.core.files.storage
from algorithms.unsupervised.Functions.Accountability.NormalizationScore import normalization_score as get_normalization_score_unsupervised
from algorithms.supervised.Functions.Robustness.CleverScore_supervised import get_clever_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clever_score_unsupervised
from algorithms.TrustScore.TrustScore import get_trust_score as get_trust_score_supervised
from algorithms.unsupervised.Functions.Fairness.Fairness import analyse as get_fairness_score_unsupervised
from algorithms.supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised
from algorithms.unsupervised.Functions.Explainability.Explainability import analyse as get_explainability_score_unsupervised
from algorithms.supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised
from algorithms.unsupervised.Functions.Robustness.Robustness import analyse as get_robustness_unsupervised
from algorithms.unsupervised.Functions.Accountability.Accountability import analyse as get_accountability_score_unsupervised
from algorithms.supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised
from algorithms.supervised.Functions.Fairness.AverageOddsDifferenceScore_supervised import get_average_odds_difference_score_supervised
from algorithms.supervised.Functions.Fairness.EqualOpportunityDifferenceScore_supervised import get_equal_opportunity_difference_score_supervised
from algorithms.unsupervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_unsupervised
from algorithms.supervised.Functions.Fairness.StatisticalParityDifferenceScore import get_statistical_parity_difference_score_supervised
from algorithms.unsupervised.Functions.Fairness.UnderfittingScore import underfitting_score as get_underfitting_score_unsupervised
from algorithms.supervised.Functions.Fairness.UnderfittingScore_supervised import get_underfitting_score_supervised
from algorithms.unsupervised.Functions.Fairness.OverfittingScore import overfitting_score as get_overfitting_score_unsupervised
from algorithms.supervised.Functions.Fairness.OverfittingScore_supervised import get_overfitting_score_supervised
from algorithms.supervised.Functions.Fairness.ClassBalanceScore_supervised import get_class_balance_score_supervised
from algorithms.unsupervised.Functions.Fairness.DisparateImpactScore import disparate_impact_score as get_disparate_impact_score_unsupervised
from algorithms.supervised.Functions.Fairness.DisparateImpactScore_supervised import get_disparate_impact_score_supervised
from algorithms.unsupervised.Functions.Explainability.PermutationFeatureScore import permutation_feature_importance_score as get_permutation_feature_importance_score_unsupervised
from algorithms.supervised.Functions.Explainability.FeatureRelevanceScore_supervised import get_feature_relevance_score_supervised
from algorithms.supervised.Functions.Explainability.AlgorithmClassScore_supervised import get_algorithm_class_score_supervised
from algorithms.unsupervised.Functions.Explainability.CorrelatedFeaturesScore import correlated_features_score as get_correlated_features_score_unsupervised
from algorithms.supervised.Functions.Explainability.CorrelatedFeaturesScore_supervised import get_correlated_features_score_supervised
from algorithms.unsupervised.Functions.Explainability.ModelSizeScore import model_size_score as get_modelsize_score_unsupervised
from algorithms.supervised.Functions.Explainability.ModelSizeScore_supervised import get_model_size_score_supervised as get_modelsize_score_supervised
from algorithms.supervised.Functions.Robustness.LossSensitivityScore_supervised import get_loss_sensitivity_score_supervised
from algorithms.supervised.Functions.Robustness.ERFastGradientAttackScore_supervised import get_er_fast_gradient_attack_score_supervised as get_fast_gradient_attack_score_supervised
from algorithms.supervised.Functions.Robustness.ERDeepFoolAttackScore_supervised import get_deepfool_attack_score_supervised as get_deepfoolattack_score_supervised
from algorithms.supervised.Functions.Robustness.ERCarliniWagnerScore_supervised import get_er_carlini_wagner_score_supervised as get_carliwagnerwttack_score_supervised
from algorithms.supervised.Functions.Robustness.ConfidenceScore_supervised import get_confidence_score_supervised
from algorithms.supervised.Functions.Robustness.CliqueMethodScore_supervised import get_clique_method_supervised as get_clique_method_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clique_method_score_unsupervised
from algorithms.unsupervised.Functions.Accountability.TrainTestSplitScore import train_test_split_score as get_train_test_split_score_unsupervised
from algorithms.unsupervised.Functions.Accountability.RegularizationScore import regularization_score as get_regularization_score_unsupervised
from algorithms.supervised.Functions.Accountability.MissingDataScore_supervised import get_missing_data_score_supervised
from algorithms.unsupervised.Functions.Accountability.MissingDataScore import missing_data_score as get_missing_data_score_unsupervised
from rest_framework.parsers import MultiPartParser, FormParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg.inspectors.view import SwaggerAutoSchema
from rest_framework.decorators import authentication_classes, api_view, parser_classes
from algorithms.supervised.Functions.Accountability.NormalizationScore_supervised import get_normalization_score_supervised
from algorithms.supervised.Functions.Accountability.RegularizationScore_supervised import get_regularization_score_supervised
from algorithms.supervised.Functions.Accountability.TrainTestSplitScore_supervised import get_train_test_split_score_supervised
from algorithms.supervised.Functions.Accountability.FactSheetCompletnessScore_supervised import get_factsheet_completeness_score_supervised
from algorithms.unsupervised.Functions.Accountability.FactSheetCompletnessScore import get_factsheet_completeness_score_unsupervised
from algorithms.supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised
from algorithms.unsupervised.Functions.Robustness.Robustness import analyse as get_robustness_score_unsupervised
from algorithms.TrustScore.TrustScore import trusting_AI_scores_unsupervised
from algorithms.TrustScore.TrustScore import trusting_AI_scores_supervised
from .models import Scenario
from operator import index
from statistics import mode
from requests.api import request
import yfinance as yf
from datetime import datetime
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import serializers, status
from django.shortcuts import render
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date, timedelta
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
import math
from django.http import JsonResponse
import json
from .models import CustomUser, Scenario, ScenarioSolution
from .serilizers import UserSerializer, SolutionSerializer
# import stripe
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from rest_framework import status
from rest_framework.decorators import api_view
from pathlib import Path
import os
import collections
import json

from .algorithms.unsupervised.fairness import analyse as analyse_fairness_unsupervised
from .algorithms.unsupervised.explainability import analyse as analyse_explainability_unsupervised
from .algorithms.unsupervised.robustness import analyse as analyse_robustness_unsupervised
from .algorithms.unsupervised.methodology import analyse as analyse_methodology_unsupervised
from .authentication import CustomUserAuthentication

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
print("Base dir is:", BASE_DIR)


def scenario_detail(request, scenarioName, user_id):
    # Query the database for the relevant scenario based on scenarioName and user_id
    scenario = Scenario.objects.get(name=scenarioName, user_id=user_id)
    return render(request, 'scenario_detail.html', {'scenario': scenario})


# factsheet=pd.read_json(os.path.join(BASE_DIR,r'apis/TestValues/factsheet.json'))
# training_dataset=pd.read_csv(os.path.join(BASE_DIR,r'apis/TestValues/train.csv'))
# test_dataset=pd.read_csv(os.path.join(BASE_DIR,r'apis/TestValues/test.csv'))
# mappings1=pd.read_json(os.path.join(BASE_DIR,r'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json'))
# print("Mappings:",mappings1)
# factsheet=pd.read_json(os.path.join('factsheet.json'))
# training_dataset=pd.read_csv('train.csv')
# test_dataset=pd.read_csv('test.csv')
# mappings1=pd.read_json('default.json')

# def normalization_score(model, train_data, test_data, factsheet, mappings):
#     import numpy as np
#     import collections
#     import pandas as pd
#     import json
#     from pathlib import Path
#     train_data=pd.read_csv(train_data)
#     test_data=pd.read_csv(test_data)
#     with open(mappings, 'r') as f:
#         mappings = json.loads(f.read())
#     with open(factsheet, 'r') as g:
#         factsheet = json.loads(g.read())
#     mappings=mappings["score_normalization"]["mappings"]["value"]


#     info = collections.namedtuple('info', 'description value')
#     result = collections.namedtuple('result', 'score properties')

#     X_train = train_data.iloc[:, :-1]
#     X_test = test_data.iloc[:, :-1]
#     train_mean = np.mean(np.mean(X_train))
#     train_std = np.mean(np.std(X_train))
#     test_mean = np.mean(np.mean(X_test))
#     test_std = np.mean(np.std(X_test))
#     from cmath import isclose

#     properties = {"dep" :info('Depends on','Training and Testing Data'),
#     "Training_mean": info("Mean of the training data", "{:.2f}".format(train_mean)),
#     "Training_std": info("Standard deviation of the training data", "{:.2f}".format(train_std)),
#     "Test_mean": info("Mean of the test data", "{:.2f}".format(test_mean)),
#     "Test_std": info("Standard deviation of the test data", "{:.2f}".format(test_std))
#     }
#     if not (any(X_train < 0) or any(X_train > 1)) and not (any(X_test < 0) or any(X_test > 1)):
#         score = mappings["training_and_test_normal"]
#         properties["normalization"] = info("Normalization", "Training and Testing data are normalized")
#     elif isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (not isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and not isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
#         score = mappings["training_standardized"]
#         properties["normalization"] = info("Normalization", "Training data are standardized")

#     elif isclose(train_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(train_std, 1, rel_tol=1e-3, abs_tol=1e-6) and (isclose(test_mean, 0, rel_tol=1e-3, abs_tol=1e-6) and isclose(test_std, 1, rel_tol=1e-3, abs_tol=1e-6)):
#         score = mappings["training_and_test_standardize"]
#         properties["normalization"] = info("Normalization", "Training and Testing data are standardized")
#     elif any(X_train < 0) or any(X_train > 1):
#         score = mappings["None"]
#         properties["normalization"] = info("Normalization", "None")
#     elif not (any(X_train < 0) or any(X_train > 1)) and (any(X_test < 0) or any(X_test > 1)):
#         score = mappings["training_normal"]
#         properties["normalization"] = info("Normalization", "Training data are normalized")
#     return result(score=score, properties=properties)


# path_testdata=os.path.join(BASE_DIR,'apis/TestValues/test.csv')
# path_traindata=os.path.join(BASE_DIR,'apis/TestValues/train.csv')
# path_module=os.path.join(BASE_DIR,'apis/TestValues/model.pkl')
# path_factsheet=os.path.join(BASE_DIR,'apis/TestValues/factsheet.json')
# path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
# print(normalization_score(path_module,path_traindata,path_testdata,path_factsheet,path_mapping_accountabiltiy))

#####################################################################################


################################ Robustness  ######################################
def analyse_robustness(model, train_data, test_data, config, factsheet):
    import numpy as np
    import collections
    import random
    import sklearn.metrics as metrics
    from art.attacks.evasion import FastGradientMethod, CarliniL2Method, DeepFool
    from art.estimators.classification import SklearnClassifier
    from sklearn.preprocessing import OneHotEncoder
    from art.metrics import clever_u, RobustnessVerificationTreeModelsCliqueMethod
    from art.estimators.classification import KerasClassifier
    from art.metrics import loss_sensitivity
    import tensorflow as tf
    import numpy.linalg as la
    import json

    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')
    from .FourPillars.Robustness.ConfidenceScore.ConfidenceScore import confidence_score
    from .FourPillars.Robustness.CleverScore.CleverScore import clever_score
    from .FourPillars.Robustness.CliqueMethod.CliqueMethodScore import clique_method
    from .FourPillars.Robustness.LossSensitivity.LossSensitivityScore import loss_sensitivity_score
    from .FourPillars.Robustness.ERFastGradientMethod.FastGradientAttackScore import fast_gradient_attack_score
    from .FourPillars.Robustness.ERCWAttack.CarliWagnerAttackScore import carlini_wagner_attack_score
    from .FourPillars.Robustness.ERDeepFool.DeepFoolAttackScore import deepfool_attack_score
    """Reads the thresholds from the config file.
    Calls all robustness metric functions with correct arguments.
    Organizes all robustness metrics in a dict. Then returns the scores and the properties.
        Args:
            model: ML-model.
            training_dataset: pd.DataFrame containing the used training data.
            test_dataset: pd.DataFrame containing the used test data.
            config: Config file containing the threshold values for the metrics.
            factsheet: json document containing all information about the particular solution.

        Returns:
            Returns a result object containing all metric scores
            and matching properties for every metric
    """
    model = pd.read_pickle(model)
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    config = pd.read_json(config)

    factsheet = pd.read_json(factsheet)

    clique_method_thresholds = config["score_clique_method"]["thresholds"]["value"]
    print("clique_method_thresholds:", clique_method_thresholds)
    clever_score_thresholds = config["score_clever_score"]["thresholds"]["value"]
    loss_sensitivity_thresholds = config["score_loss_sensitivity"]["thresholds"]["value"]
    confidence_score_thresholds = config["score_confidence_score"]["thresholds"]["value"]
    fsg_attack_thresholds = config["score_fast_gradient_attack"]["thresholds"]["value"]
    cw_attack_thresholds = config["score_carlini_wagner_attack"]["thresholds"]["value"]
    deepfool_thresholds = config["score_carlini_wagner_attack"]["thresholds"]["value"]

    output = dict(
        confidence_score=confidence_score(
            model, train_data, test_data, confidence_score_thresholds),
        clique_method=clique_method(
            model, train_data, test_data, clique_method_thresholds, factsheet),
        loss_sensitivity=loss_sensitivity_score(
            model, train_data, test_data, loss_sensitivity_thresholds),
        clever_score=clever_score(
            model, train_data, test_data, clever_score_thresholds),
        er_fast_gradient_attack=fast_gradient_attack_score(
            model, train_data, test_data, fsg_attack_thresholds),
        er_carlini_wagner_attack=carlini_wagner_attack_score(
            model, train_data, test_data, cw_attack_thresholds),
        er_deepfool_attack=deepfool_attack_score(
            model, train_data, test_data, deepfool_thresholds)
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
# path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
path_mapping_fairness = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')
print("Robustness reslt:", analyse_robustness(path_module,
      path_traindata, path_testdata, path_mapping_fairness, path_factsheet))
print("############################################################################")
###########################################################################################

########################### Fairness ###############################################


def analyse_fairness(model, training_dataset, test_dataset, factsheet, config):
    import numpy as np
    np.random.seed(0)
    from scipy.stats import chisquare
    import operator
    import funcy

    model = pd.read_pickle(model)
    training_dataset = pd.read_csv(training_dataset)
    test_dataset = pd.read_csv(test_dataset)

    with open(config, 'r') as f:
        config = json.loads(f.read())
    with open(factsheet, 'r') as g:
        factsheet = json.loads(g.read())

    from .FourPillars.Fairness.Always.Overfitting.OverfittingScore import overfitting_score
    from .FourPillars.Fairness.Always.Underfitting.UnderfittingScore import underfitting_score

    from .FourPillars.Fairness.Conditional.StatisticalParityDifference.StatisticalParityDifferenceScore import statistical_parity_difference_score
    from .FourPillars.Fairness.Conditional.EqualOpportunityDifference.EqualOpportunityDifferenceScore import equal_opportunity_difference_score
    from .FourPillars.Fairness.Conditional.AverageOddsDifference.AverageOddsDifferenceScore import average_odds_difference_score
    from .FourPillars.Fairness.Conditional.DisparateImpact.DisparateImpactScore import disparate_impact_score
    from .FourPillars.Fairness.Always.ClassBalance.ClassBalanceScore import class_balance_score

    import collections
    info = collections.namedtuple('info', 'description value')
    result = collections.namedtuple('result', 'score properties')

    target_column = "Income"
    """Triggers the fairness analysis and in a first step all fairness metrics get computed.
    In a second step, the scores for the fairness metrics are then created from
    mapping every metric value to a respective score.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.
        config: Config file containing the threshold values for the metrics.

    Returns:
        Returns a result object containing all metric scores
        and matching properties for every metric

    """

    statistical_parity_difference_thresholds = config[
        "score_statistical_parity_difference"]["thresholds"]["value"]
    overfitting_thresholds = config["score_overfitting"]["thresholds"]["value"]
    underfitting_thresholds = config["score_underfitting"]["thresholds"]["value"]
    equal_opportunity_difference_thresholds = config[
        "score_equal_opportunity_difference"]["thresholds"]["value"]
    average_odds_difference_thresholds = config["score_average_odds_difference"]["thresholds"]["value"]
    disparate_impact_thresholds = config["score_disparate_impact"]["thresholds"]["value"]

    output = dict(
        underfitting=underfitting_score(
            model, training_dataset, test_dataset, factsheet, underfitting_thresholds),
        overfitting=overfitting_score(
            model, training_dataset, test_dataset, factsheet, overfitting_thresholds),
        statistical_parity_difference=statistical_parity_difference_score(
            model, training_dataset, factsheet, statistical_parity_difference_thresholds),
        equal_opportunity_difference=equal_opportunity_difference_score(
            model, test_dataset, factsheet, equal_opportunity_difference_thresholds),
        average_odds_difference=average_odds_difference_score(
            model, test_dataset, factsheet, average_odds_difference_thresholds),
        disparate_impact=disparate_impact_score(
            model, test_dataset, factsheet, disparate_impact_thresholds),
        class_balance=class_balance_score(training_dataset, factsheet)
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
# path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
path_mapping_fairness = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/fairness/default.json')
print("Fairness reslt:", analyse_fairness(path_module, path_traindata,
      path_testdata, path_factsheet, path_mapping_fairness))
print("############################################################################")

###############################################################################


def analyse_explainability(clf, train_data, test_data, config, factsheet):
    import numpy as np
    import pandas as pd
    import json
    import collections
    from .FourPillars.Explainability.AlgorithmClass.AlgorithmClassScore import algorithm_class_score

    from .FourPillars.Explainability.CorrelatedFeatures.CorrelatedFeaturesScore import correlated_features_score
    from .FourPillars.Explainability.FeatureRelevance.FeatureRelevanceScore import feature_relevance_score
    from .FourPillars.Explainability.ModelSize.ModelSizeScore import model_size_score

    result = collections.namedtuple('result', 'score properties')
    info = collections.namedtuple('info', 'description value')

    # convert path data to values
    clf = pd.read_pickle(clf)
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    config = pd.read_json(config)

    factsheet = pd.read_json(factsheet)

    # with open(config, 'r') as f:
    #     config = json.loads(f.read())
    # with open(factsheet, 'r') as g:
    #     factsheet = json.loads(g.read())
    # function parameters
    target_column = factsheet["general"].get("target_column")
    clf_type_score = config["score_algorithm_class"]["clf_type_score"]["value"]
    ms_thresholds = config["score_model_size"]["thresholds"]["value"]
    cf_thresholds = config["score_correlated_features"]["thresholds"]["value"]
    high_cor = config["score_correlated_features"]["high_cor"]["value"]
    fr_thresholds = config["score_feature_relevance"]["thresholds"]["value"]
    threshold_outlier = config["score_feature_relevance"]["threshold_outlier"]["value"]
    penalty_outlier = config["score_feature_relevance"]["penalty_outlier"]["value"]

    output = dict(
        algorithm_class=algorithm_class_score(clf, clf_type_score),
        correlated_features=correlated_features_score(
            train_data, test_data, thresholds=cf_thresholds, target_column=target_column, high_cor=high_cor),
        model_size=model_size_score(train_data, ms_thresholds),
        feature_relevance=feature_relevance_score(clf, train_data, target_column=target_column, thresholds=fr_thresholds,
                                                  threshold_outlier=threshold_outlier, penalty_outlier=penalty_outlier)
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
# path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
path_mapping_fairness = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
print("Explainability reslt:", analyse_explainability(path_module,
      path_traindata, path_testdata, path_mapping_fairness, path_factsheet))
print("############################################################################")


def analyse_methodology(model, training_dataset, test_dataset, factsheet, methodology_config):
    import collections
    info = collections.namedtuple('info', 'description value')
    import json
    import os
    import numpy as np
    import collections
    # from .FourPillars.helperfunctions import list_of_metrics
    from .FourPillars.Accountability.FactSheetCompletness.FactSheetCompletnessScore import get_factsheet_completeness_score
    from .FourPillars.Accountability.MissingData.MissingDataScore import missing_data_score
    from .FourPillars.Accountability.Normalization.NormalizationScore import normalization_score
    from .FourPillars.Accountability.Regularization.RegularizationScore import regularization_score
    from .FourPillars.Accountability.TrainTestSplit.TrainTestSplitScore import train_test_split_score
    import tensorflow as tf
    from math import isclose
    import re
    result = collections.namedtuple('result', 'score properties')

    model = pd.read_pickle(model)
    training_dataset = pd.read_csv(training_dataset)
    test_dataset = pd.read_csv(test_dataset)

    with open(methodology_config, 'r') as f:
        methodology_config = json.loads(f.read())
    with open(factsheet, 'r') as g:
        factsheet = json.loads(g.read())
    normalization_mapping = methodology_config["score_normalization"]["mappings"]["value"]
    missing_data_mapping = methodology_config["score_missing_data"]["mappings"]["value"]['no_null_values']
    train_test_split_mapping = methodology_config["score_train_test_split"]["mappings"]["value"]['50-60 95-97']

    # metrics = list_of_metrics("methodology")
    output = dict(
        # output[metric] = exec("%s_score(model, training_dataset, test_dataset, factsheet, methodology_config)" % metric)
        normalization=normalization_score(
            model, training_dataset, test_dataset, factsheet, normalization_mapping),
        missing_data=missing_data_score(
            model, training_dataset, test_dataset, factsheet, missing_data_mapping),
        regularization=regularization_score(
            model, training_dataset, test_dataset, factsheet, methodology_config),
        train_test_split=train_test_split_score(
            model, training_dataset, test_dataset, factsheet, train_test_split_mapping),
        # test_accuracy = test_accuracy_score(model, training_dataset, test_dataset, factsheet, accuracy_thresholds),
        # f1_score = f1_score(model, training_dataset, test_dataset, factsheet, f1_score_thresholds),
        factsheet_completeness=get_factsheet_completeness_score(
            model, training_dataset, test_dataset, factsheet, methodology_config)
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)


path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
path_mapping_accountabiltiy = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
print("Accountability reslt:", analyse_methodology(path_module,
      path_traindata, path_testdata, path_factsheet, path_mapping_accountabiltiy))
print("############################################################################")


def get_final_score(model, train_data, test_data, config_weights, mappings_config, factsheet, recalc=False):
    mappingConfig1 = mappings_config

    with open(mappings_config, 'r') as f:
        mappings_config = json.loads(f.read())
    # mappings_config=pd.read_json(mappings_config)

    config_fairness = mappings_config["fairness"]
    config_explainability = mappings_config["explainability"]
    config_robustness = mappings_config["robustness"]
    config_methodology = mappings_config["methodology"]

    methodology_config = os.path.join(
        BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
    config_explainability = os.path.join(
        BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
    config_fairness = os.path.join(
        BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/fairness/default.json')
    config_robustness = os.path.join(
        BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')

    def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config):
        # if "scores" in factsheet.keys() and "properties" in factsheet.keys():
        #     scores = factsheet["scores"]
        #     properties = factsheet["properties"]
        # else:
        output = dict(
            fairness=analyse_fairness(
                model, train_data, test_data, factsheet, config_fairness),
            explainability=analyse_explainability(
                model, train_data, test_data, config_explainability, factsheet),
            robustness=analyse_robustness(
                model, train_data, test_data, config_robustness, factsheet),
            methodology=analyse_methodology(
                model, train_data, test_data, factsheet, methodology_config)
        )
        scores = dict((k, v.score) for k, v in output.items())
        properties = dict((k, v.properties) for k, v in output.items())
        # factsheet["scores"] = scores
        # factsheet["properties"] = properties
        # write_into_factsheet(factsheet, solution_set_path)

        return result(score=scores, properties=properties)

    with open(mappingConfig1, 'r') as f:
        default_map = json.loads(f.read())
    # default_map=pd.read_json(mappingConfig1)

    with open(factsheet, 'r') as g:
        factsheet = json.loads(g.read())
    # factsheet=pd.read_json(factsheet)

    # print("mapping is default:")
    # print(default_map == mappings_config)
    if default_map == mappings_config:
        if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
            scores = factsheet["scores"]
            properties = factsheet["properties"]
    else:
        result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness,
                                    config_explainability, config_robustness, config_methodology)
        scores = result.score
        factsheet["scores"] = scores
        properties = result.properties
        factsheet["properties"] = properties
    # try:
    #     write_into_factsheet(factsheet, solution_set_path)
    # except Exception as e:
    #     print("ERROR in write_into_factsheet: {}".format(e))
    # else:
    #     result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology, solution_set_path)
    #     scores = result.score
    #     properties = result.properties

    final_scores = dict()
    # scores = tuple(scores)
    print("Scores is:", scores)
    with open(config_weights, 'r') as n:
        config_weights = json.loads(n.read())
    # config_weights=pd.read_json(config_weights)
    # configdict = {}
    print("Config weight:", config_weights)

    fairness_score = 0
    explainability_score = 0
    robustness_score = 0
    methodology_score = 0
    for pillar in scores.items():
        # print("Pillars:", pillar)
        # pillar = {'pillar':pillar}

        if pillar[0] == 'fairness':
            # print("Pillar fairness is:", pillar[1])
            fairness_score = int(
                pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
            + int(pillar[1]['statistical_parity_difference'])*0.15 + \
                int(pillar[1]['equal_opportunity_difference'])*0.2
            + int(pillar[1]['average_odds_difference']) * \
                0.1 + int(pillar[1]['disparate_impact'])*0.1
            + int(pillar[1]['class_balance'])*0.1

            print("Fairness Score is:", fairness_score)

        if pillar[0] == 'explainability':
            # print("Pillar explainability is:", pillar[1]['algorithm_class'])
            algorithm_class = 0
            correlated_features = 0
            model_size = 0
            feature_relevance = 0

            if str(pillar[1]['algorithm_class']) != 'nan':
                algorithm_class = int(pillar[1]['algorithm_class'])*0.55

            if str(pillar[1]['correlated_features']) != 'nan':
                correlated_features = int(
                    pillar[1]['correlated_features'])*0.15

            if str(pillar[1]['model_size']) != 'nan':
                model_size = int(pillar[1]['model_size'])*5

            if str(pillar[1]['feature_relevance']) != 'nan':
                feature_relevance = int(pillar[1]['feature_relevance'])*0.15

            print("algorithm_class Score is:", algorithm_class)
            print("correlated_features Score is:", correlated_features)
            print("model_size Score is:", model_size)
            print("feature_relevance Score is:", feature_relevance)

            explainability_score = algorithm_class + \
                correlated_features + model_size + feature_relevance

            print("explainability Score is:", explainability_score)

        if pillar[0] == 'robustness':
            # print("Pillar robustness is:", pillar[1])

            confidence_score = 0
            clique_method = 0
            loss_sensitivity = 0
            clever_score = 0
            er_fast_gradient_attack = 0
            er_carlini_wagner_attack = 0
            er_deepfool_attack = 0

            if str(pillar[1]['confidence_score']) != 'nan':
                confidence_score = int(pillar[1]['confidence_score'])*0.2

            if str(pillar[1]['clique_method']) != 'nan':
                clique_method = int(pillar[1]['clique_method'])*0.2

            if str(pillar[1]['loss_sensitivity']) != 'nan':
                loss_sensitivity = int(pillar[1]['loss_sensitivity'])*0.2

            if str(pillar[1]['clever_score']) != 'nan':
                clever_score = int(pillar[1]['clever_score'])*0.2

            if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
                er_fast_gradient_attack = int(
                    pillar[1]['er_fast_gradient_attack'])*0.2

            if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
                er_carlini_wagner_attack = int(
                    pillar[1]['er_carlini_wagner_attack'])*0.2

            if str(pillar[1]['er_deepfool_attack']) != 'nan':
                er_deepfool_attack = int(pillar[1]['er_deepfool_attack'])*0.2

            robustness_score = confidence_score + clique_method + loss_sensitivity + \
                clever_score + er_fast_gradient_attack + \
                er_carlini_wagner_attack + er_deepfool_attack

            print("robustness Score is:", robustness_score)

        if pillar[0] == 'methodology':
            # print("Pillar methodology is:", pillar[1])
            normalization = 0
            missing_data = 0
            regularization = 0
            train_test_split = 0
            factsheet_completeness = 0

            if str(pillar[1]['normalization']) != 'nan':
                normalization = int(pillar[1]['normalization'])*0.2

            if str(pillar[1]['missing_data']) != 'nan':
                missing_data = int(pillar[1]['missing_data'])*0.2

            if str(pillar[1]['regularization']) != 'nan':
                regularization = int(pillar[1]['regularization'])*0.2

            if str(pillar[1]['train_test_split']) != 'nan':
                train_test_split = int(pillar[1]['train_test_split'])*0.2

            if str(pillar[1]['factsheet_completeness']) != 'nan':
                factsheet_completeness = int(
                    pillar[1]['factsheet_completeness'])*0.2

            methodology_score = normalization + missing_data + \
                regularization + train_test_split + factsheet_completeness

            print("methodology Score is:", methodology_score)

    trust_score = fairness_score*0.25 + explainability_score * \
        0.25 + robustness_score*0.25 + methodology_score*0.25
    print("Trust Score is:", trust_score)
    #     config = config_weights[pillar]
    #     weighted_scores = list(map(lambda x: scores[pillar][x] * config[x], scores[pillar].keys()))
    #     sum_weights = np.nansum(np.array(list(config.values()))[~np.isnan(weighted_scores)])
    # if sum_weights == 0:
    #     result = 0
    # else:
    #     result = round(np.nansum(weighted_scores)/sum_weights,1)
    #     final_scores[pillar] = result

    # return scores, properties


path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
config_weights = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
mappings_config = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
factsheet = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
# solution_set_path,
# path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
# path_mapping_fairness=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
print("Final Score result:", get_final_score(path_module, path_traindata,
      path_testdata, config_weights, mappings_config, path_factsheet))


result = collections.namedtuple('result', 'score properties')
FACTSHEET_NAME = "Newfact"


def write_into_factsheet(new_factsheet, solution_set_path):
    factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
    with open(factsheet_path, 'w') as outfile:
        json.dump(new_factsheet, outfile, indent=4)
    return


def get_final_score_unsupervised(model, train_data, test_data, outliers_data, config_weights, mappings_config, factsheet, solution_set_path, recalc=False):
    mappingConfig1 = mappings_config

    with open(mappings_config, 'r') as f:
        mappings_config = json.loads(f.read())

    config_fairness = mappings_config["fairness"]
    config_explainability = mappings_config["explainability"]
    config_robustness = mappings_config["robustness"]
    config_methodology = mappings_config["methodology"]

    # with open('configs/unsupervised/mappings/default.json', 'r') as f:
    #     default_map = json.loads(f.read())
    with open(mappingConfig1, 'r') as f:
        default_map = json.loads(f.read())

    # print("mapping is default:")
    # print(default_map == mappings_config)

    with open(factsheet, 'r') as g:
        factsheet = json.loads(g.read())

    if default_map == mappings_config:
        if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
            scores = factsheet["scores"]
            properties = factsheet["properties"]
        else:
            print("======================================================== no scores ======================================")
            result = trusting_AI_scores_unsupervised(model, train_data, test_data, outliers_data, factsheet, config_fairness, config_explainability,
                                                     config_robustness, config_methodology, solution_set_path)
            scores = result.score
            factsheet["scores"] = scores
            properties = result.properties
            factsheet["properties"] = properties
            try:
                print("======================================================== write into factsheet ======================================")

                write_into_factsheet(factsheet, solution_set_path)
            except Exception as e:
                print("ERROR in write_into_factsheet: {}".format(e))
    else:
        result = trusting_AI_scores_unsupervised(model, train_data, test_data, outliers_data, factsheet, config_fairness, config_explainability,
                                                 config_robustness, config_methodology, solution_set_path)
        scores = result.score
        properties = result.properties

    final_scores = dict()

    print("Scores is:", scores)
    with open(config_weights, 'r') as n:
        config_weights = json.loads(n.read())
    # config_weights=pd.read_json(config_weights)
    # configdict = {}
    print("Config weight:", config_weights)

    fairness_score = 0
    explainability_score = 0
    robustness_score = 0
    methodology_score = 0
    for pillar in scores.items():
        print("Pillars is:", pillar)
        # pillar = {'pillar':pillar}

        if pillar[0] == 'fairness':
            # print("Pillar fairness is:", pillar[1])
            fairness_score = int(
                pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
            + int(pillar[1]['statistical_parity_difference']) * \
                0.15 + int(pillar[1]['disparate_impact'])*0.1

            print("Fairness Score is:", fairness_score)

        if pillar[0] == 'explainability':
            # print("Pillar explainability is:", pillar[1]['algorithm_class'])
            algorithm_class = 0
            correlated_features = 0
            model_size = 0
            feature_relevance = 0

            if str(pillar[1]['correlated_features']) != 'nan':
                correlated_features = int(
                    pillar[1]['correlated_features'])*0.15

            if str(pillar[1]['model_size']) != 'nan':
                model_size = int(pillar[1]['model_size'])*5

            if str(pillar[1]['permutation_feature_importance']) != 'nan':
                feature_relevance = int(
                    pillar[1]['permutation_feature_importance'])*0.15

            print("algorithm_class Score is:", algorithm_class)
            print("correlated_features Score is:", correlated_features)
            print("model_size Score is:", model_size)
            print("feature_relevance Score is:", feature_relevance)

            explainability_score = correlated_features + model_size + feature_relevance

            print("explainability Score is:", explainability_score)

        if pillar[0] == 'robustness':
            # print("Pillar robustness is:", pillar[1])

            confidence_score = 0
            clique_method = 0
            loss_sensitivity = 0
            clever_score = 0
            er_fast_gradient_attack = 0
            er_carlini_wagner_attack = 0
            er_deepfool_attack = 0

            # if str(pillar[1]['confidence_score']) != 'nan':
            #     confidence_score= int(pillar[1]['confidence_score'])*0.2

            # if str(pillar[1]['clique_method']) != 'nan':
            #     clique_method= int(pillar[1]['clique_method'])*0.2

            # if str(pillar[1]['loss_sensitivity']) != 'nan':
            #     loss_sensitivity= int(pillar[1]['loss_sensitivity'])*0.2

            if str(pillar[1]['clever_score']) != 'nan':
                clever_score = int(pillar[1]['clever_score'])*0.2

            # if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
            #     er_fast_gradient_attack= int(pillar[1]['er_fast_gradient_attack'])*0.2

            # if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
            #     er_carlini_wagner_attack= int(pillar[1]['er_carlini_wagner_attack'])*0.2

            # if str(pillar[1]['er_deepfool_attack']) != 'nan':
            #     er_deepfool_attack= int(pillar[1]['er_deepfool_attack'])*0.2

            robustness_score = clever_score

            print("robustness Score is:", robustness_score)

        if pillar[0] == 'methodology':
            # print("Pillar methodology is:", pillar[1])
            normalization = 0
            missing_data = 0
            regularization = 0
            train_test_split = 0
            factsheet_completeness = 0

            if str(pillar[1]['normalization']) != 'nan':
                normalization = int(pillar[1]['normalization'])*0.2

            if str(pillar[1]['missing_data']) != 'nan':
                missing_data = int(pillar[1]['missing_data'])*0.2

            if str(pillar[1]['regularization']) != 'nan':
                regularization = int(pillar[1]['regularization'])*0.2

            if str(pillar[1]['train_test_split']) != 'nan':
                train_test_split = int(pillar[1]['train_test_split'])*0.2

            if str(pillar[1]['factsheet_completeness']) != 'nan':
                factsheet_completeness = int(
                    pillar[1]['factsheet_completeness'])*0.2

            methodology_score = normalization + missing_data + \
                regularization + train_test_split + factsheet_completeness

            print("methodology Score is:", methodology_score)

    trust_score = fairness_score*0.25 + explainability_score * \
        0.25 + robustness_score*0.25 + methodology_score*0.25
    # print("Trust Score is:", trust_score)
    return trust_score
    ############################################
    # for pillar, item in scores.items():
    #     config = config_weights[pillar]
    #     weighted_scores = list(map(lambda x: scores[pillar][x] * config[x], scores[pillar].keys()))
    #     sum_weights = np.nansum(np.array(list(config.values()))[~np.isnan(weighted_scores)])
    #     if sum_weights == 0:
    #         result = 0
    #     else:
    #         result = round(np.nansum(weighted_scores) / sum_weights, 1)
    #     final_scores[pillar] = result

    # return final_scores, scores, properties
    ###############################################


path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/cblof/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/cblof/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/cblof/model.joblib')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/cblof/factsheet.json')
outliers_data = os.path.join(BASE_DIR, 'apis/TestValues/cblof/outliers.csv')
config_weights = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
mappings_config = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
factsheet = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
solution_set_path = os.path.join(BASE_DIR, 'apis/TestValues/New_path/')
# solution_set_path,
# path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
# path_mapping_fairness=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
print("Final Score unsupervised:", get_final_score_unsupervised(path_module, path_traindata,
      path_testdata, outliers_data, config_weights, mappings_config, path_factsheet, solution_set_path))


@csrf_exempt
def create_checkout_session(request):
    if request.method == "POST":
        try:
            checkout_session = stripe.checkout.Session.create(
                line_items=[
                    {
                        # Provide the exact Price ID (for example, pr_1234) of the product you want to sell
                        'price': 'price_1K6T4QAycbtKpV299EI4d61w',
                        'quantity': 3,
                    },
                ],
                mode='payment',
                success_url=YOUR_DOMAIN + '?success=true',
                cancel_url=YOUR_DOMAIN + '?canceled=true',
            )

        except Exception as e:
            print("eeeeee", e)
            return JsonResponse(e)
        return redirect(checkout_session.url, code=303)

# request.tutorial.title


class user(APIView):
    def get(self, request, email):
        uploaddic = {}
        print("email:", email)
        ScenarioName = []
        ModelLinks = []
        LinktoDataset = []
        Description = []

        userexist = CustomUser.objects.filter(email=email)
        if userexist:
            # serializer=UserSerializer(data=request.data)
            # print("Serializer:",serializer)
            # if serializer.is_valid():
            #     serializer.save(response_data=uploaddic)
            #     # serializer.save(response_data={"PreviousInv": perc_Invs, "PercentageAR": perc_returns, "StockNames": stockNames, "PercentARlistT": PercentAR_t, "SP500listt": SP500_t, "W5000listt": W5000_t, "corrlist": corrlist, "dailyreturnportf": list(daily_returns_portf), "fullloc1": list(full.iloc[1]), "fullloc0": list(full.iloc[0]), "full1loc1": list(full1.iloc[1]), "full1loc0": list(full1.iloc[0])})
            #     print("Nice to Add Second!")
            #     # return Response(stockInfoDic)
            #     return Response(uploaddic)

            # else:
            #     print('errors:  ',serializer.errors)

            userobj = CustomUser.objects.get(email=email)
            scenarioobj = Scenario.objects.filter(user_id=userobj.id).values()

            # print("Scenarioobj:",scenarioobj['response_data'][''])
            if scenarioobj:
                for i in scenarioobj:
                    print('i:', i)
                    ScenarioName.append(i['scenario_name'])
                    # str(i['scenario_name']).split("['")[1].split("']")[0])
                    # ModelLinks.append(
                    # str(a['ModelLinks']).split("['")[1].split("']")[0])
                    # LinktoDataset.append(
                    # str(a['LinktoDataset']).split("['")[1].split("']")[0])
                    Description.append(i['description'])
                    # str(a['description']).split("['")[1].split("']")[0])

            print("ScenarioName", ScenarioName)
            uploaddic['ScenarioName'] = ScenarioName
            # uploaddic['ModelLinks'] = ModelLinks
            # uploaddic['LinktoDataset'] = LinktoDataset
            uploaddic['Description'] = Description
            # print("User exist",scenarioobj)
            print("uploaddic", uploaddic)
            return Response(uploaddic)
            # return Response("Successfully Get!")
        else:
            print("User not exist.... Created new")
        return Response(uploaddic)

    def post(self, request):
        print('data:', request.data)
        userexist = CustomUser.objects.filter(email=request.data['emailid'])
        if userexist:
            uploaddic = {}

            ScenarioName = ''
            ModelLinks = ''
            LinktoDataset = ''
            Description = ''

            if request.data is not None:
                ScenarioName = request.data['ScenarioName'],
                ModelLinks = request.data['ModelLinks'],
                LinktoDataset = request.data['LinktoDataset'],
                Description = request.data['Description'],

            uploaddic['ScenarioName'] = ScenarioName
            uploaddic['ModelLinks'] = ModelLinks
            uploaddic['LinktoDataset'] = LinktoDataset
            uploaddic['Description'] = Description

            # user = ScenarioUser.objects.get(user_id=request.data['Userid'])
            # request.data['user'] = user.id
            # # serializer=FirstStockSerializer(data=request.data)
            # serializer = UserSerializer(data=request.data)
            # print("Serializer:", serializer)
            # if serializer.is_valid():
            #     serializer.save(response_data=uploaddic)
            #     # serializer.save(response_data={"PreviousInv": perc_Invs, "PercentageAR": perc_returns, "StockNames": stockNames, "PercentARlistT": PercentAR_t, "SP500listt": SP500_t, "W5000listt": W5000_t, "corrlist": corrlist, "dailyreturnportf": list(daily_returns_portf), "fullloc1": list(full.iloc[1]), "fullloc0": list(full.iloc[0]), "full1loc1": list(full1.iloc[1]), "full1loc0": list(full1.iloc[0])})
            #     print("Nice to Add Second!")
            #     # return Response(stockInfoDic)
            #     return Response(uploaddic)

            # else:
            #     print('errors:  ', serializer.errors)
            #     # return Response(serializer.errors)

            userobj = CustomUser.objects.get(email=request.data['emailid'])
            scenarioobj = Scenario.objects.create(
                user=userobj,
                scenario_name=request.data['ScenarioName'],
                # ModelLinks=request.data['ModelLinks'],
                # LinktoDataset=request.data['LinktoDataset'],
                description=request.data['Description'],
            )
            return Response('successfully created scenario', status=200)
        else:
            # try:
            #     user, created = CustomUser.objects.get_or_create(
            #         user_id=request.data['Userid'],
            #         email=request.data['emailid'],
            #     )
            #     return Response("Successfully add!")
            # except:
            #     message = {'detail': 'User with this email already exists'}
            #     return Response("Unable to create User")
            # return Response(message, status=status.HTTP_400_BAD_REQUEST)
            # CustomUser.objec
            print("User ID:", request.data['Userid'])
            print("Email ID:", request.data['emailid'])
            createuser = CustomUser.objects.create(
                email=request.data['emailid'], is_admin=0)
            createuser.save()

            uploaddic = {}

            ScenarioName = ''
            ModelLinks = ''
            LinktoDataset = ''
            Description = ''

            if request.data is not None:
                ScenarioName = request.data['ScenarioName'],
                ModelLinks = request.data['ModelLinks'],
                LinktoDataset = request.data['LinktoDataset'],
                Description = request.data['Description'],

            uploaddic['ScenarioName'] = ScenarioName
            uploaddic['ModelLinks'] = ModelLinks
            uploaddic['LinktoDataset'] = LinktoDataset
            uploaddic['Description'] = Description

            # user=ScenarioUser.objects.get(user_id=request.data['Userid'])
            request.data['user'] = createuser.id
            # serializer=FirstStockSerializer(data=request.data)
            serializer = UserSerializer(data=request.data)
            print("Serializer:", serializer)
            if serializer.is_valid():
                serializer.save(response_data=uploaddic)
                # serializer.save(response_data={"PreviousInv": perc_Invs, "PercentageAR": perc_returns, "StockNames": stockNames, "PercentARlistT": PercentAR_t, "SP500listt": SP500_t, "W5000listt": W5000_t, "corrlist": corrlist, "dailyreturnportf": list(daily_returns_portf), "fullloc1": list(full.iloc[1]), "fullloc0": list(full.iloc[0]), "full1loc1": list(full1.iloc[1]), "full1loc0": list(full1.iloc[0])})
                print("Nice to Add Second!")
                # return Response(stockInfoDic)
                return Response(uploaddic)

            else:
                print('errors:  ', serializer.errors)

            print("User not exist.... Created new")
            return Response("Successfully add!")

        print("Received Post request:", request.data['Userid'])
        return Response("Successfully add!")


class userset(APIView):
    def get(self, request, email):

        user = CustomUser.objects.get(email=email)
        return Response({
            'email': user.email,
            'password': user.password,
        }, status=200)

    def post(self, request):
        user = CustomUser.objects.get(email=request.data['email'])
        user.password = request.data['password']
        user.save()

        return Response('success', status=200)


class dashboard(APIView):
    def get(self, request, email):
        print('self:', email)
        uploaddic = {}
        userexist = CustomUser.objects.get(email=email)
        scenarioobj = ScenarioSolution.objects.filter(
            user_id=userexist.id).values().order_by('id')

        def get_final_score(model, train_data, test_data, config_weights, mappings_config, factsheet, recalc=False):
            mappingConfig1 = mappings_config

            with open(mappings_config, 'r') as f:
                mappings_config = json.loads(f.read())

            config_fairness = mappings_config["fairness"]
            config_explainability = mappings_config["explainability"]
            config_robustness = mappings_config["robustness"]
            config_methodology = mappings_config["methodology"]

            methodology_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
            config_explainability = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
            config_fairness = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/fairness/default.json')
            config_robustness = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')

            def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config):
                from algorithms.supervised.Functions.Fairness.FarinessScore_supervised import get_fairness_score_supervised
                from algorithms.supervised.Functions.Explainability.ExplainabilityScore_supervised import get_explainability_score_supervised
                from algorithms.supervised.Functions.Robustness.Robustness_supervised import get_robustness_score_supervised
                from algorithms.supervised.Functions.Accountability.AccountabilityScore_supervised import get_accountability_score_supervised
                output = dict(
                    fairness=get_fairness_score_supervised(
                        model, train_data, test_data, factsheet, config_fairness),
                    explainability=get_explainability_score_supervised(
                        model, train_data, test_data, config_explainability, factsheet),
                    robustness=get_robustness_score_supervised(
                        model, train_data, test_data, config_robustness, factsheet),
                    methodology=get_accountability_score_supervised(
                        model, train_data, test_data, factsheet, methodology_config)
                )
                scores = dict((k, v.score) for k, v in output.items())
                properties = dict((k, v.properties) for k, v in output.items())
                # factsheet["scores"] = scores
                # factsheet["properties"] = properties
                # write_into_factsheet(factsheet, solution_set_path)

                return result(score=scores, properties=properties)

            with open(mappingConfig1, 'r') as f:
                default_map = json.loads(f.read())
            # default_map=pd.read_json(mappingConfig1)

            factsheet = os.path.join(BASE_DIR, factsheet)
            print('url:', factsheet)
            with open(factsheet, 'r') as g:
                factsheet = json.loads(g.read())
            # factsheet=pd.read_json(factsheet)

            scores = []
            # print("mapping is default:")
            # print(default_map == mappings_config)
            print('asdf:', factsheet.keys())
            if default_map == mappings_config:
                if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
                    scores = factsheet["scores"]
                    properties = factsheet["properties"]
            else:
                result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness,
                                            config_explainability, config_robustness, config_methodology)
                scores = result.score
                factsheet["scores"] = scores
                properties = result.properties
                factsheet["properties"] = properties

            final_scores = dict()
            # scores = tuple(scores)
            print("Scores is:", scores)
            with open(config_weights, 'r') as n:
                config_weights = json.loads(n.read())
            # config_weights=pd.read_json(config_weights)
            # configdict = {}
            print("Config weight:", config_weights)

            fairness_score = 0
            explainability_score = 0
            robustness_score = 0
            methodology_score = 0
            for pillar in scores.items():
                # print("Pillars:", pillar)
                # pillar = {'pillar':pillar}

                if pillar[0] == 'fairness':
                    # print("Pillar fairness is:", pillar[1])
                    uploaddic['underfitting'] = int(pillar[1]['underfitting'])
                    uploaddic['overfitting'] = int(pillar[1]['overfitting'])
                    uploaddic['statistical_parity_difference'] = int(
                        pillar[1]['statistical_parity_difference'])
                    uploaddic['equal_opportunity_difference'] = int(
                        pillar[1]['equal_opportunity_difference'])
                    uploaddic['average_odds_difference'] = int(
                        pillar[1]['average_odds_difference'])
                    uploaddic['disparate_impact'] = int(
                        pillar[1]['disparate_impact'])
                    uploaddic['class_balance'] = int(
                        pillar[1]['class_balance'])

                    fairness_score = int(
                        pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
                    + int(pillar[1]['statistical_parity_difference'])*0.15 + \
                        int(pillar[1]['equal_opportunity_difference'])*0.2
                    + int(pillar[1]['average_odds_difference']) * \
                        0.1 + int(pillar[1]['disparate_impact'])*0.1
                    + int(pillar[1]['class_balance'])*0.1

                    uploaddic['fairness_score'] = fairness_score
                    print("Fairness Score is:", fairness_score)

                if pillar[0] == 'explainability':
                    # print("Pillar explainability is:", pillar[1]['algorithm_class'])
                    algorithm_class = 0
                    correlated_features = 0
                    model_size = 0
                    feature_relevance = 0

                    if str(pillar[1]['algorithm_class']) != 'nan':
                        algorithm_class = int(
                            pillar[1]['algorithm_class'])*0.55

                    if str(pillar[1]['correlated_features']) != 'nan':
                        correlated_features = int(
                            pillar[1]['correlated_features'])*0.15

                    if str(pillar[1]['model_size']) != 'nan':
                        model_size = int(pillar[1]['model_size'])*5

                    if str(pillar[1]['feature_relevance']) != 'nan':
                        feature_relevance = int(
                            pillar[1]['feature_relevance'])*0.15

                    explainability_score = algorithm_class + \
                        correlated_features + model_size + feature_relevance

                    uploaddic['algorithm_class'] = algorithm_class
                    uploaddic['correlated_features'] = correlated_features
                    uploaddic['model_size'] = model_size
                    uploaddic['feature_relevance'] = feature_relevance
                    uploaddic['explainability_score'] = explainability_score
                    print("explainability Score is:", explainability_score)

                if pillar[0] == 'robustness':
                    # print("Pillar robustness is:", pillar[1])

                    confidence_score = 0
                    clique_method = 0
                    loss_sensitivity = 0
                    clever_score = 0
                    er_fast_gradient_attack = 0
                    er_carlini_wagner_attack = 0
                    er_deepfool_attack = 0

                    if str(pillar[1]['confidence_score']) != 'nan':
                        confidence_score = int(
                            pillar[1]['confidence_score'])*0.2

                    if str(pillar[1]['clique_method']) != 'nan':
                        clique_method = int(pillar[1]['clique_method'])*0.2

                    if str(pillar[1]['loss_sensitivity']) != 'nan':
                        loss_sensitivity = int(
                            pillar[1]['loss_sensitivity'])*0.2

                    if str(pillar[1]['clever_score']) != 'nan':
                        clever_score = int(pillar[1]['clever_score'])*0.2

                    if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
                        er_fast_gradient_attack = int(
                            pillar[1]['er_fast_gradient_attack'])*0.2

                    if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
                        er_carlini_wagner_attack = int(
                            pillar[1]['er_carlini_wagner_attack'])*0.2

                    if str(pillar[1]['er_deepfool_attack']) != 'nan':
                        er_deepfool_attack = int(
                            pillar[1]['er_deepfool_attack'])*0.2

                    robustness_score = confidence_score + clique_method + loss_sensitivity + \
                        clever_score + er_fast_gradient_attack + \
                        er_carlini_wagner_attack + er_deepfool_attack

                    uploaddic['confidence_score'] = confidence_score
                    uploaddic['clique_method'] = clique_method
                    uploaddic['loss_sensitivity'] = loss_sensitivity
                    uploaddic['clever_score'] = clever_score
                    uploaddic['er_fast_gradient_attack'] = er_fast_gradient_attack
                    uploaddic['er_carlini_wagner_attack'] = er_carlini_wagner_attack
                    uploaddic['er_deepfool_attack'] = er_deepfool_attack
                    uploaddic['robustness_score'] = robustness_score
                    print("robustness Score is:", robustness_score)

                if pillar[0] == 'methodology':
                    # print("Pillar methodology is:", pillar[1])
                    normalization = 0
                    missing_data = 0
                    regularization = 0
                    train_test_split = 0
                    factsheet_completeness = 0

                    if str(pillar[1]['normalization']) != 'nan':
                        normalization = int(pillar[1]['normalization'])*0.2

                    if str(pillar[1]['missing_data']) != 'nan':
                        missing_data = int(pillar[1]['missing_data'])*0.2

                    if str(pillar[1]['regularization']) != 'nan':
                        regularization = int(pillar[1]['regularization'])*0.2

                    if str(pillar[1]['train_test_split']) != 'nan':
                        train_test_split = int(
                            pillar[1]['train_test_split'])*0.2

                    if str(pillar[1]['factsheet_completeness']) != 'nan':
                        factsheet_completeness = int(
                            pillar[1]['factsheet_completeness'])*0.2

                    methodology_score = normalization + missing_data + \
                        regularization + train_test_split + factsheet_completeness

                    uploaddic['normalization'] = normalization
                    uploaddic['missing_data'] = missing_data
                    uploaddic['regularization'] = regularization
                    uploaddic['train_test_split'] = train_test_split
                    uploaddic['factsheet_completeness'] = factsheet_completeness
                    uploaddic['methodology_score'] = (
                        "%.2f" % methodology_score)
                    print("methodology Score is:", methodology_score)

            trust_score = fairness_score*0.25 + explainability_score * \
                0.25 + robustness_score*0.25 + methodology_score*0.25
            uploaddic['trust_score'] = trust_score
            print("Trust Score is:", trust_score)
            #     config = config_weights[pillar]
            #     weighted_scores = list(map(lambda x: scores[pillar][x] * config[x], scores[pillar].keys()))
            #     sum_weights = np.nansum(np.array(list(config.values()))[~np.isnan(weighted_scores)])
            # if sum_weights == 0:
            #     result = 0
            # else:
            #     result = round(np.nansum(weighted_scores)/sum_weights,1)
            #     final_scores[pillar] = result

            # return scores, properties

        path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
        path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
        path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
        path_factsheet = os.path.join(
            BASE_DIR, 'apis/TestValues/factsheet.json')
        config_weights = os.path.join(
            BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
        mappings_config = os.path.join(
            BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
        factsheet = os.path.join(
            BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

        if scenarioobj:
            for i in scenarioobj:
                path_testdata = i["test_file"]
                path_module = i["model_file"]
                path_traindata = i["training_file"]
                path_factsheet = i["factsheet_file"]

        print("Final Score result:", get_final_score(path_module, path_traindata,
              path_testdata, config_weights, mappings_config, path_factsheet))

        result = collections.namedtuple('result', 'score properties')
        FACTSHEET_NAME = "Newfact"

        def write_into_factsheet(new_factsheet, solution_set_path):
            factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
            with open(factsheet_path, 'w') as outfile:
                json.dump(new_factsheet, outfile, indent=4)
            return

        def trusting_AI_scores_unsupervised(model, train_data, test_data, outliers_data, factsheet, config_fairness, config_explainability,
                                            config_robustness, methodology_config, solution_set_path):
            output = dict(
                fairness=analyse_fairness_unsupervised(
                    model, train_data, test_data, outliers_data, factsheet, config_fairness),
                explainability=analyse_explainability_unsupervised(
                    model, train_data, test_data, outliers_data, config_explainability, factsheet),
                robustness=analyse_robustness_unsupervised(
                    model, train_data, test_data, outliers_data, config_robustness, factsheet),
                methodology=analyse_accountability_unsupervised(
                    model, train_data, test_data, outliers_data, factsheet, methodology_config)
            )
            scores = dict((k, v.score) for k, v in output.items())
            properties = dict((k, v.properties) for k, v in output.items())

            return result(score=scores, properties=properties)

        def get_final_score_unsupervised(model, train_data, test_data, outliers_data, config_weights, mappings_config, factsheet, solution_set_path, recalc=False):
            mappingConfig1 = mappings_config

            with open(mappings_config, 'r') as f:
                mappings_config = json.loads(f.read())

            config_fairness = mappings_config["fairness"]
            config_explainability = mappings_config["explainability"]
            config_robustness = mappings_config["robustness"]
            config_methodology = mappings_config["methodology"]

            with open(mappingConfig1, 'r') as f:
                default_map = json.loads(f.read())

            with open(factsheet, 'r') as g:
                factsheet = json.loads(g.read())

            if default_map == mappings_config:
                if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
                    scores = factsheet["scores"]
                    properties = factsheet["properties"]
                else:
                    print(
                        "======================================================== no scores ======================================")
                    result = trusting_AI_scores_unsupervised(model, train_data, test_data, outliers_data, factsheet, config_fairness, config_explainability,
                                                             config_robustness, config_methodology, solution_set_path)
                    scores = result.score
                    factsheet["scores"] = scores
                    properties = result.properties
                    factsheet["properties"] = properties
                    try:
                        print(
                            "======================================================== write into factsheet ======================================")

                        write_into_factsheet(factsheet, solution_set_path)
                    except Exception as e:
                        print("ERROR in write_into_factsheet: {}".format(e))
            else:
                result = trusting_AI_scores_unsupervised(model, train_data, test_data, outliers_data, factsheet, config_fairness, config_explainability,
                                                         config_robustness, config_methodology, solution_set_path)
                scores = result.score
                properties = result.properties

            final_scores = dict()

            print("Scores is:", scores)
            with open(config_weights, 'r') as n:
                config_weights = json.loads(n.read())
            # config_weights=pd.read_json(config_weights)
            # configdict = {}
            print("Config weight:", config_weights)

            fairness_score = 0
            explainability_score = 0
            robustness_score = 0
            methodology_score = 0
            for pillar in scores.items():
                print("Pillars is:", pillar)
                # pillar = {'pillar':pillar}

                if pillar[0] == 'fairness':
                    # print("Pillar fairness is:", pillar[1])
                    fairness_score = int(
                        pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
                    + int(pillar[1]['statistical_parity_difference']) * \
                        0.15 + int(pillar[1]['disparate_impact'])*0.1

                    print("Fairness Score is:", fairness_score)

                    uploaddic['unsupervised_underfitting'] = int(
                        pillar[1]['underfitting'])
                    uploaddic['unsupervised_overfitting'] = int(
                        pillar[1]['overfitting'])
                    uploaddic['unsupervised_statistical_parity_difference'] = int(
                        pillar[1]['statistical_parity_difference'])
                    uploaddic['unsupervised_disparate_impact'] = int(
                        pillar[1]['disparate_impact'])
                    uploaddic['unsupervised_fairness_score'] = fairness_score

                if pillar[0] == 'explainability':
                    # print("Pillar explainability is:", pillar[1]['algorithm_class'])
                    algorithm_class = 0
                    correlated_features = 0
                    model_size = 0
                    feature_relevance = 0

                    if str(pillar[1]['correlated_features']) != 'nan':
                        correlated_features = int(
                            pillar[1]['correlated_features'])*0.15

                    if str(pillar[1]['model_size']) != 'nan':
                        model_size = int(pillar[1]['model_size'])*5

                    if str(pillar[1]['permutation_feature_importance']) != 'nan':
                        feature_relevance = int(
                            pillar[1]['permutation_feature_importance'])*0.15

                    print("algorithm_class Score is:", algorithm_class)
                    print("correlated_features Score is:", correlated_features)
                    print("model_size Score is:", model_size)
                    print("feature_relevance Score is:", feature_relevance)

                    explainability_score = correlated_features + model_size + feature_relevance

                    print("explainability Score is:", explainability_score)

                    uploaddic['unsupervised_correlated_features'] = correlated_features
                    uploaddic['unsupervised_model_size'] = model_size
                    uploaddic['unsupervised_feature_relevance'] = feature_relevance
                    uploaddic['unsupervised_explainability_score'] = explainability_score

                if pillar[0] == 'robustness':
                    # print("Pillar robustness is:", pillar[1])

                    confidence_score = 0
                    clique_method = 0
                    loss_sensitivity = 0
                    clever_score = 0
                    er_fast_gradient_attack = 0
                    er_carlini_wagner_attack = 0
                    er_deepfool_attack = 0

                    if str(pillar[1]['clever_score']) != 'nan':
                        clever_score = int(pillar[1]['clever_score'])*0.2

                    robustness_score = clever_score

                    print("robustness Score is:", robustness_score)

                    uploaddic['unsupervised_clever_score'] = clever_score
                    uploaddic['unsupervised_robustness_score'] = robustness_score

                if pillar[0] == 'methodology':
                    # print("Pillar methodology is:", pillar[1])
                    normalization = 0
                    missing_data = 0
                    regularization = 0
                    train_test_split = 0
                    factsheet_completeness = 0

                    if str(pillar[1]['normalization']) != 'nan':
                        normalization = int(pillar[1]['normalization'])*0.2

                    if str(pillar[1]['missing_data']) != 'nan':
                        missing_data = int(pillar[1]['missing_data'])*0.2

                    if str(pillar[1]['regularization']) != 'nan':
                        regularization = int(pillar[1]['regularization'])*0.2

                    if str(pillar[1]['train_test_split']) != 'nan':
                        train_test_split = int(
                            pillar[1]['train_test_split'])*0.2

                    if str(pillar[1]['factsheet_completeness']) != 'nan':
                        factsheet_completeness = int(
                            pillar[1]['factsheet_completeness'])*0.2

                    methodology_score = normalization + missing_data + \
                        regularization + train_test_split + factsheet_completeness

                    print("methodology Score is:", methodology_score)

                    uploaddic['unsupervised_normalization'] = normalization
                    uploaddic['unsupervised_missing_data'] = missing_data
                    uploaddic['unsupervised_regularization'] = regularization
                    uploaddic['unsupervised_train_test_split'] = train_test_split
                    uploaddic['unsupervised_factsheet_completeness'] = factsheet_completeness

                    uploaddic['unsupervised_methodology_score'] = methodology_score

            trust_score = fairness_score*0.25 + explainability_score * \
                0.25 + robustness_score*0.25 + methodology_score*0.25
            # print("Trust Score is:", trust_score)
            uploaddic['unsupervised_trust_score'] = trust_score
            return trust_score

        path_testdata = os.path.join(
            BASE_DIR, 'apis/TestValues/cblof/test.csv')
        path_traindata = os.path.join(
            BASE_DIR, 'apis/TestValues/cblof/train.csv')
        path_module = os.path.join(
            BASE_DIR, 'apis/TestValues/cblof/model.joblib')
        path_factsheet = os.path.join(
            BASE_DIR, 'apis/TestValues/cblof/factsheet.json')
        outliers_data = os.path.join(
            BASE_DIR, 'apis/TestValues/cblof/outliers.csv')
        config_weights = os.path.join(
            BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
        mappings_config = os.path.join(
            BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
        factsheet = os.path.join(
            BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
        solution_set_path = os.path.join(BASE_DIR, 'apis/TestValues/New_path/')
        print("Final Score unsupervised:", get_final_score_unsupervised(path_module, path_traindata,
              path_testdata, outliers_data, config_weights, mappings_config, path_factsheet, solution_set_path))

        scenarios = Scenario.objects.filter(user_id=userexist.id).values()
        uploaddic['scenarioList'] = scenarios
        uploaddic['solutionList'] = scenarioobj
        return Response(uploaddic)

    def post(self, request):
        return Response("Successfully add!")


class solutiondetail(APIView):
    def get(self, request, id):
        print("id:", id)

        solutionDetail = ScenarioSolution.objects.get(id=id)

        return Response({
            'solution_name': solutionDetail.solution_name,
            'description': solutionDetail.description,
            'solution_type': solutionDetail.solution_type,
            'protected_features': solutionDetail.protected_features,
            'protected_values': solutionDetail.protected_values,
            'target_column': solutionDetail.target_column
        }, status=200)

    def put(self, request):
        solutionDetail = ScenarioSolution.objects.get(
            id=request.data['SolutionId'])

        solutionDetail.solution_name = request.data['NameSolution']
        solutionDetail.description = request.data['DescriptionSolution']
        solutionDetail.training_file = request.data['TrainingFile']
        solutionDetail.test_file = request.data['TestFile']
        solutionDetail.factsheet_file = request.data['FactsheetFile']
        solutionDetail.model_file = request.data['ModelFile']
        solutionDetail.target_column = request.data['Targetcolumn']
        solutionDetail.outlier_data_file = request.data['Outlierdatafile']
        solutionDetail.protected_features = request.data['ProtectedFeature']
        solutionDetail.protected_values = request.data['Protectedvalues']
        solutionDetail.favourable_outcome = request.data['Favourableoutcome']
        solutionDetail.save()

        return Response('successfully changed', 200)


class scenario(APIView):
    def get(self, request, scenarioId):

        print('id:', scenarioId)
        scenario = Scenario.objects.get(id=scenarioId)

        if (scenario is not None):
            return Response({
                'scenarioName': scenario.scenario_name,
                'description': scenario.description,
            }, status=200)
        else:
            return Response("Not Exist", status=201)

    def put(self, request):
        scenario = Scenario.objects.get(
            id=request.data['id'])

        scenario.scenario_name = request.data['name']
        scenario.description = request.data['description']
        scenario.save()

        return Response("successfully changed")


class solution(APIView):
    def get(self, request, email):
        uploaddic = {}

        SolutionName = []
        ModelLinks = []
        LinktoDataset = []
        Description = []

        userexist = CustomUser.objects.filter(email=email)
        if userexist:
            userobj = CustomUser.objects.get(email=email)
            scenarios = Scenario.objects.filter(user_id=userobj.id).values()
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userobj.id).values()

            # print("Scenarioobj:",scenarioobj['response_data'][''])
            if scenarios:
                print("Scenarios data:", scenarios)
            if scenarioobj:
                for i in scenarioobj:
                    # print("Solution is:",str(i.SolutionName))
                    SolutionName.append(i['solution_name'])
                    # ModelLinks.append(str(i.response_data['ModelLinks']).split("['")[1].split("']")[0])
                    # LinktoDataset.append(str(i.response_data['LinktoDataset']).split("['")[1].split("']")[0])
                    # Description.append(str(i.response_data['Description']).split("['")[1].split("']")[0])

            print("SolutionName", SolutionName)
            uploaddic['SolutionName'] = SolutionName
            # uploaddic['ModelLinks'] = ModelLinks
            # uploaddic['LinktoDataset'] = LinktoDataset
            # uploaddic['Description'] = Description
            # # print("User exist",scenarioobj)
            # print("uploaddic",uploaddic)
            return Response(uploaddic)

            # return Response("Successfully Get!")
        else:
            print("User not exist.... Created new")
            return Response("User not exist.... Created new")
        # return Response(uploaddic)
        # return Response("User not exist.... Created new")

    def post(self, request):
        if request.data is not None:
            # # trainingdata=request.data['TrainnigDatafile']
            # # serializer=SolutionSerializer(data=request.data)
            try:
                print('email:', request.data)
                userexist = CustomUser.objects.get(
                    email=request.data['emailid'])
                scenario = Scenario.objects.get(
                    scenario_name=request.data['SelectScenario'])
                print("Solution type:", scenario.id)
                fileupload = ScenarioSolution.objects.create(
                    user=userexist,
                    scenario_id=scenario.id,
                    solution_name=request.data['NameSolution'],
                    description=request.data['DescriptionSolution'],
                    training_file=request.data['TrainingFile'],
                    test_file=request.data['TestFile'],
                    factsheet_file=request.data['FactsheetFile'],
                    model_file=request.data['ModelFile'],
                    target_column=request.data['Targetcolumn'],
                    solution_type=request.data['Solutiontype'],

                    outlier_data_file=request.data['Outlierdatafile'],
                    protected_features=request.data['ProtectedFeature'],
                    protected_values=request.data['Protectedvalues'],
                    favourable_outcome=request.data['Favourableoutcome']
                )
                fileupload.save()

                return Response("Successfully add!", status=200)

            except Exception as e:
                return Response("Error occured", status=201)

            # FactsheetFile=request.data['FactsheetFile'],
            # ModelFile=request.data['ModelFile'],
            # Targetcolumn=request.data['Targetcolumn']

            # Outlierdatafile = request.data['Outlierdatafile']
            # Protectedfeat = request.data['ProtectedFeature']
            # Protectedval = request.data['Protectedvalues']
            # Favoutcome = request.data['Favourableoutcome']

            # print("FactsheetFile:", type(FactsheetFile))
            # print("ModelFile:", type(ModelFile))
            # print("Targetcolumn:", type(Targetcolumn))
            # print("Outlierdatafile:", Outlierdatafile)
            # print("Protectedfeat:", type(Protectedfeat))
            # print("Protectedval:", type(Protectedval))
            # print("Favoutcome:", type(Favoutcome))

            # for i in request.FILES.get['TrainnigDatafile']:
            # print("File i:",i)
            # print("Received Post request:",request.data['Userid'])
            # print("Received SelectScenario request:",request.data['SelectScenario'])
            # print("Received TrainnigDatafile request:",request.data['TrainnigDatafile'])
            # print("Type of file:",type(request.data['file']))


class registerUser(APIView):
    def get(self, request, email):
        uploaddic = {}

        SolutionName = []
        ModelLinks = []
        LinktoDataset = []
        Description = []

        userexist = CustomUser.objects.filter(email=email)
        if userexist:
            userobj = CustomUser.objects.get(email=email)
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userobj.id).values()

            # print("Scenarioobj:",scenarioobj['response_data'][''])
            if scenarioobj:
                for i in scenarioobj:
                    # print("Solution is:",str(i.SolutionName))
                    SolutionName.append(i.SolutionName)
                    # ModelLinks.append(str(i.response_data['ModelLinks']).split("['")[1].split("']")[0])
                    # LinktoDataset.append(str(i.response_data['LinktoDataset']).split("['")[1].split("']")[0])
                    # Description.append(str(i.response_data['Description']).split("['")[1].split("']")[0])

            print("SolutionName", SolutionName)
            uploaddic['SolutionName'] = SolutionName
            # uploaddic['ModelLinks'] = ModelLinks
            # uploaddic['LinktoDataset'] = LinktoDataset
            # uploaddic['Description'] = Description
            # # print("User exist",scenarioobj)
            # print("uploaddic",uploaddic)
            return Response(uploaddic)

            # return Response("Successfully Get!")
        else:
            print("User not exist.... Created new")
            return Response("User not exist.... Created new")
        # return Response(uploaddic)
        # return Response("User not exist.... Created new")

    def post(self, request):
        if request.data is not None:
            fullname = request.data['fullname']
            email = request.data['email']
            password = request.data['password']
            # serializer=SolutionSerializer(data=request.data)
            # userexist = ScenarioUser.objects.get(fullname=request.data['fullname'],
            # email=request.data['email'])
            userexist = CustomUser.objects.filter(username=request.data['fullname'],
                                                  email=request.data['email'])
            if userexist:
                print("User Already Exist!")
                return Response("User Already Exist!")
            else:
                userform = CustomUser.objects.create(
                    name=request.data['fullname'],
                    email=request.data['email'],
                    password=request.data['password'],
                )
                userform.save()
                print("Successfully Created User!")
                return Response("Successfully Created User!")

            # fileupload = ScenarioSolution.objects.create(
            #     user=userexist,
            #     ScenarioName=request.data['SelectScenario'],
            #     SolutionName=request.data['NameSolution'],
            #     SolutionDescription=request.data['DescriptionSolution'],
            #     TrainingFile=request.data['TrainingFile'],
            #     TestFile=request.data['TesdtataFile'],
            #     FactsheetFile=request.data['FactsheetFile'],
            #     ModelFile=request.data['ModelFile'],
            #     Targetcolumn=request.data['Targetcolumn']
            # )
            # fileupload.save()
            # for i in request.FILES.get['TrainnigDatafile']:
            # print("File i:",i)
            # print("Received registerUser data request:",request.data)
            # print("Received Post request:",request.data['Userid'])
            # print("Received SelectScenario request:",request.data['SelectScenario'])
            # print("Received TrainnigDatafile request:",request.data['TrainnigDatafile'])
            # print("Type of file:",type(request.data['file']))
        return Response("Successfully add!")


class userpage(APIView):
    def get(self, request, email):
        uploaddic = {}
        print("I'm in Userpage....")
        SolutionName = []
        ModelLinks = []
        LinktoDataset = []
        Description = []

        userexist = CustomUser.objects.get(email=email)
        print("Is user admin:", userexist.is_admin)
        if userexist.is_admin == True:
            uploaddic['Admin'] = "Admin"

            users = []
            userlist = CustomUser.objects.all()
            for i in userlist:
                print("Each object:", i.email)
                users.append(i.email)
            uploaddic['users'] = users
            print("Users list:", users)
        else:
            uploaddic['Admin'] = "noad"
        # if userexist:
        #     userobj=ScenarioUser.objects.get(user_id=id)
        #     scenarios = userobj.Scenario.all()
        #     scenarioobj = userobj.scenariosolution.all()

        #     # print("Scenarioobj:",scenarioobj['response_data'][''])
        #     if scenarios:
        #         print("Scenarios data:", scenarios)
        #     if scenarioobj:
        #         for i in scenarioobj:
        #             # print("Solution is:",str(i.SolutionName))
        #             SolutionName.append(i.SolutionName)
        #             # ModelLinks.append(str(i.response_data['ModelLinks']).split("['")[1].split("']")[0])
        #             # LinktoDataset.append(str(i.response_data['LinktoDataset']).split("['")[1].split("']")[0])
        #             # Description.append(str(i.response_data['Description']).split("['")[1].split("']")[0])

        #     print("SolutionName",SolutionName)
        #     uploaddic['SolutionName'] = SolutionName
        #     # uploaddic['ModelLinks'] = ModelLinks
        #     # uploaddic['LinktoDataset'] = LinktoDataset
        #     # uploaddic['Description'] = Description
        #     # # print("User exist",scenarioobj)
        #     # print("uploaddic",uploaddic)
        #     return Response(uploaddic)

            # return Response("Successfully Get!")
        # else:
        #     print("User not exist.... Created new")
        #     return Response("User not exist.... Created new")
        return Response(uploaddic)
        # return Response("User not exist.... Created new")

    def post(self, request):
        uploaddic = {}
        print('data:', request.data)
        if request.data is not None:
            # # trainingdata=request.data['TrainnigDatafile']
            # # serializer=SolutionSerializer(data=request.data)
            userexist = CustomUser.objects.get(
                email=request.data['Useremail'])
            scenario = Scenario.objects.filter(user_id=userexist.id).values()
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userexist.id).values()
            ScenarioName = []
            SolutionName = []

            if scenario:
                for i in scenario:
                    print("Response data ScenarioName:",
                          i['scenario_name']),
                    # print("Response data Description:", i.response_data['Description']),
                    # print("Response data LinktoDataset:", i.response_data['LinktoDataset'])

                    ScenarioName.append(i['scenario_name'])
                    # LinktoDataset.append(str(i.response_data['LinktoDataset']).split("['")[1].split("']")[0]),
                    # Description.append(str(i.response_data['Description']).split("['")[1].split("']")[0]),

            uploaddic['ScenarioName'] = ScenarioName

            if scenarioobj:
                for i in scenarioobj:
                    SolutionName.append(i['solution_name'])
        #             # ModelLinks.append(str(i.response_data['ModelLinks']).split("['")[1].split("']")[0])
        #             # LinktoDataset.append(str(i.response_data['LinktoDataset']).split("['")[1].split("']")[0])
        #             # Description.append(str(i.response_data['Description']).split("['")[1].split("']")[0])

        #     print("SolutionName",SolutionName)
            uploaddic['SolutionName'] = SolutionName

            # for i in request.FILES.get['TrainnigDatafile']:
            # print("File i:",i)
            print("Received request.data request:", request.data)
            # print("Received Post request:",request.data['Userid'])
            # print("Received SelectScenario request:",request.data['SelectScenario'])
            # print("Received TrainnigDatafile request:",request.data['TrainnigDatafile'])
            # print("Type of file:",type(request.data['file']))
        return Response(uploaddic)


class analyze(APIView):
    def get(self, request, id):

        print("User not exist.... Created new")
        # return Response(uploaddic)

    def post(self, request):
        uploaddic = {}

        ScenarioName = []
        LinktoDataset = []
        Description = []

        accuracy = []
        globalrecall = []
        classweightedrecall = []
        globalprecision = []
        classweightedprecision = []
        globalf1score = []
        classweightedf1score = []

        ModelType = []
        TrainTestSplit = []
        classweightedrecall = []
        globalprecision = []
        classweightedprecision = []
        globalf1score = []

        print("POST analyze request.data request:", request.data)
        if request.data is not None:
            userexist = CustomUser.objects.get(
                email=request.data['emailid'])
            scenario = Scenario.objects.filter(
                scenario_name=request.data['SelectScenario']).values()
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userexist.id).values()

            if scenario:
                for i in scenario:
                    # print("Response data ScenarioName:", i.response_data['ScenarioName']),
                    # print("Response data SelectScenario:", request.data['SelectScenario']),
                    # print("Response data Description:", i.response_data['Description']),
                    # print("Response data LinktoDataset:", i.response_data['LinktoDataset'])
                    if (i['scenario_name'] == request.data['SelectScenario']):
                        print("Response data ScenarioName:",
                              i['scenario_name']),
                        print("Response data Description:",
                              i['description']),

                        ScenarioName.append(i['scenario_name']),
                        # LinktoDataset.append(
                        # str(i.response_data['LinktoDataset']).split("['")[1].split("']")[0]),
                        Description.append(i['description']),

                    # ScenarioName= i.response_data['ScenarioName'],
                    # LinktoDataset= i.response_data['LinktoDataset'],
                    # Description= i.response_data['Description'],

            print("ScenarioName", ScenarioName)
            print("LinktoDataset", LinktoDataset)
            print("Description", Description)

            uploaddic['ScenarioName'] = ScenarioName
            uploaddic['LinktoDataset'] = LinktoDataset
            uploaddic['Description'] = Description
            # ScenarioName=request.data['SelectScenario'],
            # SolutionName=request.data['SelectSolution'],

            from sklearn import metrics
            import numpy as np
            import tensorflow as tf

            DEFAULT_TARGET_COLUMN_INDEX = -1
            DEFAULT_TARGET_COLUMN_NAME = 'Target'
            import pandas as pd

            def get_performance_metrics(model, test_data, target_column, train_data, factsheet):
                model = pd.read_pickle(model)
                test_data = pd.read_csv(test_data)

                train_data = pd.read_csv(train_data)

                factsheet = os.path.join(BASE_DIR, factsheet)
                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())

                y_test = test_data[target_column]
                y_true = y_test.values.flatten()

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

                performance_metrics = pd.DataFrame({
                    "accuracy": [metrics.accuracy_score(y_true, y_pred)],
                    "global recall": [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted recall": [metrics.recall_score(y_true, y_pred, average="weighted")],
                    "global precision": [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted precision": [metrics.precision_score(y_true, y_pred, average="weighted")],
                    "global f1 score": [metrics.f1_score(y_true, y_pred, average="micro")],
                    "class weighted f1 score": [metrics.f1_score(y_true, y_pred, average="weighted")],
                }).round(decimals=2)

                uploaddic['accuracy'] = (
                    "%.2f" % metrics.accuracy_score(y_true, y_pred))
                uploaddic['globalrecall'] = ("%.2f" % metrics.recall_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedrecall'] = (
                    "%.2f" % metrics.recall_score(y_true, y_pred, average="weighted"))
                uploaddic['globalprecision'] = ("%.2f" % metrics.precision_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedprecision'] = (
                    "%.2f" % metrics.precision_score(y_true, y_pred, average="weighted"))
                uploaddic['globalf1score'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="micro"))
                uploaddic['classweightedf1score'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="weighted"))

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
                    uploaddic['ModelType'] = factsheet["explainability"]["algorithm_class"]["clf_name"][1]
                    uploaddic['TrainTestSplit'] = factsheet["methodology"]["train_test_split"]["train_test_split"][1]
                    uploaddic['DataSize'] = str(
                        train_data.shape[0]) + " samples / " + str(test_data.shape[0]) + " samples"
                    uploaddic['RegularizationTechnique'] = factsheet["methodology"]["regularization"]["regularization_technique"][1]
                    uploaddic['NormalizationTechnique'] = factsheet["methodology"]["normalization"]["normalization"][1]
                    uploaddic['NumberofFeatures'] = factsheet["explainability"]["model_size"]["n_features"][1]
                    # print("Model type:", factsheet["explainability"]["algorithm_class"]["clf_name"][1])
                    properties = properties.transpose()
                    properties = properties.reset_index()
                    properties['index'] = properties['index'].str.title()
                    properties.rename(
                        columns={"index": "key", 0: "value"}, inplace=True)
                performance_metrics = performance_metrics.transpose()
                performance_metrics = performance_metrics.reset_index()
                performance_metrics['index'] = performance_metrics['index'].str.title(
                )
                performance_metrics.rename(
                    columns={"index": "key", 0: "value"}, inplace=True)

                # print("Performance Metrics:", performance_metrics)
                return performance_metrics

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            print('sca:', scenario)
            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario[0]['id'] and i['solution_name'] == request.data['SelectSolution']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']

            print("Performance_Metrics reslt:", get_performance_metrics(
                path_module, path_testdata, 'Target', path_traindata, path_factsheet))

            def get_factsheet_completeness_score(factsheet):
                propdic = {}
                import collections
                info = collections.namedtuple('info', 'description value')
                result = collections.namedtuple('result', 'score properties')

                factsheet = os.path.join(BASE_DIR, factsheet)
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

            # print("Factsheet_completeness Score result:", get_factsheet_completeness_score(path_module, path_traindata, path_testdata, path_factsheet, mappings_config))

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario[0]['id'] and i['solution_name'] == request.data['SelectSolution']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']
                        Target = i['target_column']
                        print("Factsheet file:", i.FactsheetFile)
                        print("ScenarioSolution data:", i.SolutionName)

            completeness_prop = get_factsheet_completeness_score(
                path_factsheet)
            # print("completeness Score:######", completeness_prop[1]['model_name'][1])
            # print("completeness Score:######", completeness_prop[1]['model_name'][1])

            uploaddic['modelname'] = completeness_prop[1]['model_name'][1]
            uploaddic['purposedesc'] = completeness_prop[1]['purpose_description'][1]
            uploaddic['trainingdatadesc'] = completeness_prop[1]['training_data_description'][1]
            uploaddic['modelinfo'] = completeness_prop[1]['model_information'][1]
            uploaddic['authors'] = completeness_prop[1]['authors'][1]
            uploaddic['contactinfo'] = completeness_prop[1]['contact_information'][1]

            def get_final_score(model, train_data, test_data, config_weights, mappings_config, factsheet, recalc=False):
                mappingConfig1 = mappings_config

                with open(mappings_config, 'r') as f:
                    mappings_config = json.loads(f.read())
                # mappings_config=pd.read_json(mappings_config)

                config_fairness = mappings_config["fairness"]
                config_explainability = mappings_config["explainability"]
                config_robustness = mappings_config["robustness"]
                config_methodology = mappings_config["methodology"]

                methodology_config = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
                config_explainability = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
                config_fairness = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/fairness/default.json')
                config_robustness = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')

                def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config):
                    # if "scores" in factsheet.keys() and "properties" in factsheet.keys():
                    #     scores = factsheet["scores"]
                    #     properties = factsheet["properties"]
                    # else:
                    output = dict(
                        fairness=analyse_fairness(
                            model, train_data, test_data, factsheet, config_fairness),
                        explainability=analyse_explainability(
                            model, train_data, test_data, config_explainability, factsheet),
                        robustness=analyse_robustness(
                            model, train_data, test_data, config_robustness, factsheet),
                        methodology=analyse_methodology(
                            model, train_data, test_data, factsheet, methodology_config)
                    )
                    scores = dict((k, v.score) for k, v in output.items())
                    properties = dict((k, v.properties)
                                      for k, v in output.items())
                    # factsheet["scores"] = scores
                    # factsheet["properties"] = properties
                    # write_into_factsheet(factsheet, solution_set_path)

                    return result(score=scores, properties=properties)

                with open(mappingConfig1, 'r') as f:
                    default_map = json.loads(f.read())
                # default_map=pd.read_json(mappingConfig1)

                factsheet = os.path.join(BASE_DIR, factsheet)
                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())
                # factsheet=pd.read_json(factsheet)

                # print("mapping is default:")
                # print(default_map == mappings_config)
                if default_map == mappings_config:
                    if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
                        scores = factsheet["scores"]
                        properties = factsheet["properties"]
                else:
                    result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness,
                                                config_explainability, config_robustness, config_methodology)
                    scores = result.score
                    factsheet["scores"] = scores
                    properties = result.properties
                    factsheet["properties"] = properties
                # try:
                #     write_into_factsheet(factsheet, solution_set_path)
                # except Exception as e:
                #     print("ERROR in write_into_factsheet: {}".format(e))
                # else:
                #     result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology, solution_set_path)
                #     scores = result.score
                #     properties = result.properties

                final_scores = dict()
                # scores = tuple(scores)
                print("Scores is:", scores)
                with open(config_weights, 'r') as n:
                    config_weights = json.loads(n.read())
                # config_weights=pd.read_json(config_weights)
                # configdict = {}
                print("Config weight:", config_weights)

                fairness_score = 0
                explainability_score = 0
                robustness_score = 0
                methodology_score = 0
                for pillar in scores.items():
                    # print("Pillars:", pillar)
                    # pillar = {'pillar':pillar}

                    if pillar[0] == 'fairness':
                        # print("Pillar fairness is:", pillar[1])
                        uploaddic['underfitting'] = int(
                            pillar[1]['underfitting'])
                        uploaddic['overfitting'] = int(
                            pillar[1]['overfitting'])
                        uploaddic['statistical_parity_difference'] = int(
                            pillar[1]['statistical_parity_difference'])
                        uploaddic['equal_opportunity_difference'] = int(
                            pillar[1]['equal_opportunity_difference'])
                        uploaddic['average_odds_difference'] = int(
                            pillar[1]['average_odds_difference'])
                        uploaddic['disparate_impact'] = int(
                            pillar[1]['disparate_impact'])
                        uploaddic['class_balance'] = int(
                            pillar[1]['class_balance'])

                        fairness_score = int(
                            pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
                        + int(pillar[1]['statistical_parity_difference'])*0.15 + \
                            int(pillar[1]['equal_opportunity_difference'])*0.2
                        + int(pillar[1]['average_odds_difference']) * \
                            0.1 + int(pillar[1]['disparate_impact'])*0.1
                        + int(pillar[1]['class_balance'])*0.1

                        uploaddic['fairness_score'] = fairness_score
                        print("Fairness Score is:", fairness_score)

                    if pillar[0] == 'explainability':
                        # print("Pillar explainability is:", pillar[1]['algorithm_class'])
                        algorithm_class = 0
                        correlated_features = 0
                        model_size = 0
                        feature_relevance = 0

                        if str(pillar[1]['algorithm_class']) != 'nan':
                            algorithm_class = int(
                                pillar[1]['algorithm_class'])*0.55

                        if str(pillar[1]['correlated_features']) != 'nan':
                            correlated_features = int(
                                pillar[1]['correlated_features'])*0.15

                        if str(pillar[1]['model_size']) != 'nan':
                            model_size = int(pillar[1]['model_size'])*5

                        if str(pillar[1]['feature_relevance']) != 'nan':
                            feature_relevance = int(
                                pillar[1]['feature_relevance'])*0.15

                        explainability_score = algorithm_class + \
                            correlated_features + model_size + feature_relevance

                        uploaddic['algorithm_class'] = algorithm_class
                        uploaddic['correlated_features'] = correlated_features
                        uploaddic['model_size'] = model_size
                        uploaddic['feature_relevance'] = feature_relevance
                        uploaddic['explainability_score'] = explainability_score
                        print("explainability Score is:", explainability_score)

                    if pillar[0] == 'robustness':
                        # print("Pillar robustness is:", pillar[1])

                        confidence_score = 0
                        clique_method = 0
                        loss_sensitivity = 0
                        clever_score = 0
                        er_fast_gradient_attack = 0
                        er_carlini_wagner_attack = 0
                        er_deepfool_attack = 0

                        if str(pillar[1]['confidence_score']) != 'nan':
                            confidence_score = int(
                                pillar[1]['confidence_score'])*0.2

                        if str(pillar[1]['clique_method']) != 'nan':
                            clique_method = int(pillar[1]['clique_method'])*0.2

                        if str(pillar[1]['loss_sensitivity']) != 'nan':
                            loss_sensitivity = int(
                                pillar[1]['loss_sensitivity'])*0.2

                        if str(pillar[1]['clever_score']) != 'nan':
                            clever_score = int(pillar[1]['clever_score'])*0.2

                        if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
                            er_fast_gradient_attack = int(
                                pillar[1]['er_fast_gradient_attack'])*0.2

                        if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
                            er_carlini_wagner_attack = int(
                                pillar[1]['er_carlini_wagner_attack'])*0.2

                        if str(pillar[1]['er_deepfool_attack']) != 'nan':
                            er_deepfool_attack = int(
                                pillar[1]['er_deepfool_attack'])*0.2

                        robustness_score = confidence_score + clique_method + loss_sensitivity + \
                            clever_score + er_fast_gradient_attack + \
                            er_carlini_wagner_attack + er_deepfool_attack

                        uploaddic['confidence_score'] = confidence_score
                        uploaddic['clique_method'] = clique_method
                        uploaddic['loss_sensitivity'] = loss_sensitivity
                        uploaddic['clever_score'] = clever_score
                        uploaddic['er_fast_gradient_attack'] = er_fast_gradient_attack
                        uploaddic['er_carlini_wagner_attack'] = er_carlini_wagner_attack
                        uploaddic['er_deepfool_attack'] = er_deepfool_attack
                        uploaddic['robustness_score'] = robustness_score
                        print("robustness Score is:", robustness_score)

                    if pillar[0] == 'methodology':
                        # print("Pillar methodology is:", pillar[1])
                        normalization = 0
                        missing_data = 0
                        regularization = 0
                        train_test_split = 0
                        factsheet_completeness = 0

                        if str(pillar[1]['normalization']) != 'nan':
                            normalization = int(pillar[1]['normalization'])*0.2

                        if str(pillar[1]['missing_data']) != 'nan':
                            missing_data = int(pillar[1]['missing_data'])*0.2

                        if str(pillar[1]['regularization']) != 'nan':
                            regularization = int(
                                pillar[1]['regularization'])*0.2

                        if str(pillar[1]['train_test_split']) != 'nan':
                            train_test_split = int(
                                pillar[1]['train_test_split'])*0.2

                        if str(pillar[1]['factsheet_completeness']) != 'nan':
                            factsheet_completeness = int(
                                pillar[1]['factsheet_completeness'])*0.2

                        methodology_score = normalization + missing_data + \
                            regularization + train_test_split + factsheet_completeness

                        uploaddic['normalization'] = normalization
                        uploaddic['missing_data'] = missing_data
                        uploaddic['regularization'] = regularization
                        uploaddic['train_test_split'] = train_test_split
                        uploaddic['factsheet_completeness'] = factsheet_completeness
                        uploaddic['methodology_score'] = (
                            "%.2f" % methodology_score)
                        print("methodology Score is:", methodology_score)

                trust_score = fairness_score*0.25 + explainability_score * \
                    0.25 + robustness_score*0.25 + methodology_score*0.25
                uploaddic['trust_score'] = trust_score
                print("Trust Score is:", trust_score)
                #     config = config_weights[pillar]
                #     weighted_scores = list(map(lambda x: scores[pillar][x] * config[x], scores[pillar].keys()))
                #     sum_weights = np.nansum(np.array(list(config.values()))[~np.isnan(weighted_scores)])
                # if sum_weights == 0:
                #     result = 0
                # else:
                #     result = round(np.nansum(weighted_scores)/sum_weights,1)
                #     final_scores[pillar] = result

                # return scores, properties

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            config_weights = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
            factsheet = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            if scenarioobj:
                for i in scenarioobj:
                    if (i['scenario_id'] == scenario[0]['id'] and i['solution_name'] == request.data['SelectSolution']):
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']

            # solution_set_path,
            # path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
            # path_mapping_fairness=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
            print("Final Score result:", get_final_score(path_module, path_traindata,
                  path_testdata, config_weights, mappings_config, path_factsheet))

        # return result(score=score, properties=properties)
        return Response(uploaddic)

        # def get_factsheet_completeness_score(model, training_dataset, test_dataset, factsheet, methodology_config):
        #     import collections
        #     info = collections.namedtuple('info', 'description value')
        #     result = collections.namedtuple('result', 'score properties')

        #     score = 0
        #     properties= {"dep" :info('Depends on','Factsheet')}
        #     GENERAL_INPUTS = ["model_name", "purpose_description", "domain_description", "training_data_description", "model_information", "authors", "contact_information"]

        #     n = len(GENERAL_INPUTS)
        #     ctr = 0
        #     for e in GENERAL_INPUTS:
        #         if "general" in factsheet and e in factsheet["general"]:
        #             ctr+=1
        #             properties[e] = info("Factsheet Property {}".format(e.replace("_"," ")), "present")
        #         else:
        #             properties[e] = info("Factsheet Property {}".format(e.replace("_"," ")), "missing")
        #     score = round(ctr/n*5)
        #     # return result(score=score, properties=properties)
        #     return Response("Successfully add!")


class compare(APIView):
    def get(self, request, id):

        print("compare get.... Created new")
        # return Response(uploaddic)

    def post(self, request):
        uploaddic = {}

        ScenarioName = []
        LinktoDataset = []
        Description = []

        accuracy = []
        globalrecall = []
        classweightedrecall = []
        globalprecision = []
        classweightedprecision = []
        globalf1score = []
        classweightedf1score = []

        print("POST compare request.data request:", request.data)

        if request.data is not None:
            userexist = CustomUser.objects.get(
                email=request.data['emailid'])
            scenario = Scenario.objects.get(
                scenario_name=request.data['SelectScenario'])
            scenarioobj = ScenarioSolution.objects.filter(
                user_id=userexist.id).values()
            scenarioobj1 = ScenarioSolution.objects.filter().values()

            from sklearn import metrics
            import numpy as np
            import tensorflow as tf

            DEFAULT_TARGET_COLUMN_INDEX = -1
            DEFAULT_TARGET_COLUMN_NAME = 'Target'
            import pandas as pd

            def get_performance_metrics(model, test_data, target_column):
                model = pd.read_pickle(model)
                test_data = pd.read_csv(test_data)

                y_test = test_data[target_column]
                y_true = y_test.values.flatten()

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

                performance_metrics = pd.DataFrame({
                    "accuracy": [metrics.accuracy_score(y_true, y_pred)],
                    "global recall": [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted recall": [metrics.recall_score(y_true, y_pred, average="weighted")],
                    "global precision": [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted precision": [metrics.precision_score(y_true, y_pred, average="weighted")],
                    "global f1 score": [metrics.f1_score(y_true, y_pred, average="micro")],
                    "class weighted f1 score": [metrics.f1_score(y_true, y_pred, average="weighted")],
                }).round(decimals=2)

                uploaddic['accuracy'] = (
                    "%.2f" % metrics.accuracy_score(y_true, y_pred))
                uploaddic['globalrecall'] = ("%.2f" % metrics.recall_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedrecall'] = (
                    "%.2f" % metrics.recall_score(y_true, y_pred, average="weighted"))
                uploaddic['globalprecision'] = ("%.2f" % metrics.precision_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedprecision'] = (
                    "%.2f" % metrics.precision_score(y_true, y_pred, average="weighted"))
                uploaddic['globalf1score'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="micro"))
                uploaddic['classweightedf1score'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="weighted"))

                performance_metrics = performance_metrics.transpose()
                performance_metrics = performance_metrics.reset_index()
                performance_metrics['index'] = performance_metrics['index'].str.title(
                )
                performance_metrics.rename(
                    columns={"index": "key", 0: "value"}, inplace=True)

                # print("Performance Metrics:", performance_metrics)
                return performance_metrics

            def get_performance_metrics2(model, test_data, target_column):
                model = pd.read_pickle(model)
                test_data = pd.read_csv(test_data)

                y_test = test_data[target_column]
                y_true = y_test.values.flatten()

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

                performance_metrics = pd.DataFrame({
                    "accuracy": [metrics.accuracy_score(y_true, y_pred)],
                    "global recall": [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted recall": [metrics.recall_score(y_true, y_pred, average="weighted")],
                    "global precision": [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
                    "class weighted precision": [metrics.precision_score(y_true, y_pred, average="weighted")],
                    "global f1 score": [metrics.f1_score(y_true, y_pred, average="micro")],
                    "class weighted f1 score": [metrics.f1_score(y_true, y_pred, average="weighted")],
                }).round(decimals=2)

                uploaddic['accuracy2'] = (
                    "%.2f" % metrics.accuracy_score(y_true, y_pred))
                uploaddic['globalrecall2'] = ("%.2f" % metrics.recall_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedrecall2'] = (
                    "%.2f" % metrics.recall_score(y_true, y_pred, average="weighted"))
                uploaddic['globalprecision2'] = ("%.2f" % metrics.precision_score(
                    y_true, y_pred, labels=labels, average="micro"))
                uploaddic['classweightedprecision2'] = (
                    "%.2f" % metrics.precision_score(y_true, y_pred, average="weighted"))
                uploaddic['globalf1score2'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="micro"))
                uploaddic['classweightedf1score2'] = (
                    "%.2f" % metrics.f1_score(y_true, y_pred, average="weighted"))

                performance_metrics = performance_metrics.transpose()
                performance_metrics = performance_metrics.reset_index()
                performance_metrics['index'] = performance_metrics['index'].str.title(
                )
                performance_metrics.rename(
                    columns={"index": "key", 0: "value"}, inplace=True)

                # print("Performance Metrics:", performance_metrics)
                return performance_metrics

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            # path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
            # path_mapping_fairness=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution1']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        # print("ScenarioSolution data:", i.SolutionName)
            # print("Performance_Metrics reslt:", get_performance_metrics(
            #     path_module, path_testdata, 'Target'))

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution2']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        # print("ScenarioSolution data:", i.SolutionName)
            # print("Performance_Metrics reslt:", get_performance_metrics2(
                # path_module, path_testdata, 'Target'))

            def get_factsheet_completeness_score(factsheet):
                propdic = {}
                import collections
                info = collections.namedtuple('info', 'description value')
                result = collections.namedtuple('result', 'score properties')
                print("Factsheet:", factsheet)
                factsheet = os.path.join(BASE_DIR, factsheet)
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

            # print("Factsheet_completeness Score result:", get_factsheet_completeness_score(path_module, path_traindata, path_testdata, path_factsheet, mappings_config))

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            if scenarioobj1:
                for i in scenarioobj1:
                    print('idsafasdf:', i)
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution1']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']
                        Target = i['target_column']
                        # print("Factsheet file:", i.FactsheetFile)
                        # print("ScenarioSolution data:", i.SolutionName)

            completeness_prop = get_factsheet_completeness_score(
                path_factsheet)
            # print("completeness Score:######", completeness_prop[1]['model_name'][1])
            # print("completeness Score:######", completeness_prop[1]['model_name'][1])

            uploaddic['modelname1'] = completeness_prop[1]['model_name'][1]
            uploaddic['purposedesc1'] = completeness_prop[1]['purpose_description'][1]
            uploaddic['trainingdatadesc1'] = completeness_prop[1]['training_data_description'][1]
            uploaddic['modelinfo1'] = completeness_prop[1]['model_information'][1]
            uploaddic['authors1'] = completeness_prop[1]['authors'][1]
            uploaddic['contactinfo1'] = completeness_prop[1]['contact_information'][1]

            if scenarioobj1:
                for i in scenarioobj1:
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution2']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']
                        Target = i['target_column']
                        # print("Factsheet file:", i.FactsheetFile)
                        # print("ScenarioSolution data:", i.SolutionName)

            completeness_prop = get_factsheet_completeness_score(
                path_factsheet)
            # print("completeness Score:######", completeness_prop[1]['model_name'][1])
            # print("completeness Score:######", completeness_prop[1]['model_name'][1])

            uploaddic['modelname2'] = completeness_prop[1]['model_name'][1]
            uploaddic['purposedesc2'] = completeness_prop[1]['purpose_description'][1]
            uploaddic['trainingdatadesc2'] = completeness_prop[1]['training_data_description'][1]
            uploaddic['modelinfo2'] = completeness_prop[1]['model_information'][1]
            uploaddic['authors2'] = completeness_prop[1]['authors'][1]
            uploaddic['contactinfo2'] = completeness_prop[1]['contact_information'][1]

            def get_final_score1(model, train_data, test_data, config_weights, mappings_config, factsheet, recalc=False):
                mappingConfig1 = mappings_config

                with open(mappings_config, 'r') as f:
                    mappings_config = json.loads(f.read())
                # mappings_config=pd.read_json(mappings_config)

                config_fairness = mappings_config["fairness"]
                config_explainability = mappings_config["explainability"]
                config_robustness = mappings_config["robustness"]
                config_methodology = mappings_config["methodology"]

                methodology_config = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
                config_explainability = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
                config_fairness = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/fairness/default.json')
                config_robustness = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')

                def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config):
                    # if "scores" in factsheet.keys() and "properties" in factsheet.keys():
                    #     scores = factsheet["scores"]
                    #     properties = factsheet["properties"]
                    # else:
                    output = dict(
                        fairness=analyse_fairness(
                            model, train_data, test_data, factsheet, config_fairness),
                        explainability=analyse_explainability(
                            model, train_data, test_data, config_explainability, factsheet),
                        robustness=analyse_robustness(
                            model, train_data, test_data, config_robustness, factsheet),
                        methodology=analyse_methodology(
                            model, train_data, test_data, factsheet, methodology_config)
                    )
                    scores = dict((k, v.score) for k, v in output.items())
                    properties = dict((k, v.properties)
                                      for k, v in output.items())
                    # factsheet["scores"] = scores
                    # factsheet["properties"] = properties
                    # write_into_factsheet(factsheet, solution_set_path)

                    return result(score=scores, properties=properties)

                with open(mappingConfig1, 'r') as f:
                    default_map = json.loads(f.read())
                # default_map=pd.read_json(mappingConfig1)

                # factsheet=os.path.join(BASE_DIR,'media/' + str(factsheet))
                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())
                # factsheet=pd.read_json(factsheet)

                # print("mapping is default:")
                # print(default_map == mappings_config)
                if default_map == mappings_config:
                    if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
                        scores = factsheet["scores"]
                        properties = factsheet["properties"]
                else:
                    result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness,
                                                config_explainability, config_robustness, config_methodology)
                    scores = result.score
                    factsheet["scores"] = scores
                    properties = result.properties
                    factsheet["properties"] = properties
                # try:
                #     write_into_factsheet(factsheet, solution_set_path)
                # except Exception as e:
                #     print("ERROR in write_into_factsheet: {}".format(e))
                # else:
                #     result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology, solution_set_path)
                #     scores = result.score
                #     properties = result.properties

                final_scores = dict()
                # scores = tuple(scores)
                print("Scores is:", scores)
                with open(config_weights, 'r') as n:
                    config_weights = json.loads(n.read())
                # config_weights=pd.read_json(config_weights)
                # configdict = {}
                print("Config weight:", config_weights)

                fairness_score = 0
                explainability_score = 0
                robustness_score = 0
                methodology_score = 0
                for pillar in scores.items():
                    # print("Pillars:", pillar)
                    # pillar = {'pillar':pillar}

                    if pillar[0] == 'fairness':
                        # print("Pillar fairness is:", pillar[1])
                        uploaddic['underfitting'] = int(
                            pillar[1]['underfitting'])
                        uploaddic['overfitting'] = int(
                            pillar[1]['overfitting'])
                        uploaddic['statistical_parity_difference'] = int(
                            pillar[1]['statistical_parity_difference'])
                        uploaddic['equal_opportunity_difference'] = int(
                            pillar[1]['equal_opportunity_difference'])
                        uploaddic['average_odds_difference'] = int(
                            pillar[1]['average_odds_difference'])
                        uploaddic['disparate_impact'] = int(
                            pillar[1]['disparate_impact'])
                        uploaddic['class_balance'] = int(
                            pillar[1]['class_balance'])

                        fairness_score = int(
                            pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
                        + int(pillar[1]['statistical_parity_difference'])*0.15 + \
                            int(pillar[1]['equal_opportunity_difference'])*0.2
                        + int(pillar[1]['average_odds_difference']) * \
                            0.1 + int(pillar[1]['disparate_impact'])*0.1
                        + int(pillar[1]['class_balance'])*0.1

                        uploaddic['fairness_score1'] = fairness_score
                        print("Fairness Score is:", fairness_score)

                    if pillar[0] == 'explainability':
                        # print("Pillar explainability is:", pillar[1]['algorithm_class'])
                        algorithm_class = 0
                        correlated_features = 0
                        model_size = 0
                        feature_relevance = 0

                        if str(pillar[1]['algorithm_class']) != 'nan':
                            algorithm_class = int(
                                pillar[1]['algorithm_class'])*0.55

                        if str(pillar[1]['correlated_features']) != 'nan':
                            correlated_features = int(
                                pillar[1]['correlated_features'])*0.15

                        if str(pillar[1]['model_size']) != 'nan':
                            model_size = int(pillar[1]['model_size'])*5

                        if str(pillar[1]['feature_relevance']) != 'nan':
                            feature_relevance = int(
                                pillar[1]['feature_relevance'])*0.15

                        explainability_score = algorithm_class + \
                            correlated_features + model_size + feature_relevance

                        uploaddic['algorithm_class'] = algorithm_class
                        uploaddic['correlated_features'] = correlated_features
                        uploaddic['model_size'] = model_size
                        uploaddic['feature_relevance'] = feature_relevance
                        uploaddic['explainability_score1'] = explainability_score
                        print("explainability Score is:", explainability_score)

                    if pillar[0] == 'robustness':
                        # print("Pillar robustness is:", pillar[1])

                        confidence_score = 0
                        clique_method = 0
                        loss_sensitivity = 0
                        clever_score = 0
                        er_fast_gradient_attack = 0
                        er_carlini_wagner_attack = 0
                        er_deepfool_attack = 0

                        if str(pillar[1]['confidence_score']) != 'nan':
                            confidence_score = int(
                                pillar[1]['confidence_score'])*0.2

                        if str(pillar[1]['clique_method']) != 'nan':
                            clique_method = int(pillar[1]['clique_method'])*0.2

                        if str(pillar[1]['loss_sensitivity']) != 'nan':
                            loss_sensitivity = int(
                                pillar[1]['loss_sensitivity'])*0.2

                        if str(pillar[1]['clever_score']) != 'nan':
                            clever_score = int(pillar[1]['clever_score'])*0.2

                        if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
                            er_fast_gradient_attack = int(
                                pillar[1]['er_fast_gradient_attack'])*0.2

                        if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
                            er_carlini_wagner_attack = int(
                                pillar[1]['er_carlini_wagner_attack'])*0.2

                        if str(pillar[1]['er_deepfool_attack']) != 'nan':
                            er_deepfool_attack = int(
                                pillar[1]['er_deepfool_attack'])*0.2

                        robustness_score = confidence_score + clique_method + loss_sensitivity + \
                            clever_score + er_fast_gradient_attack + \
                            er_carlini_wagner_attack + er_deepfool_attack

                        uploaddic['confidence_score'] = confidence_score
                        uploaddic['clique_method'] = clique_method
                        uploaddic['loss_sensitivity'] = loss_sensitivity
                        uploaddic['clever_score'] = clever_score
                        uploaddic['er_fast_gradient_attack'] = er_fast_gradient_attack
                        uploaddic['er_carlini_wagner_attack'] = er_carlini_wagner_attack
                        uploaddic['er_deepfool_attack'] = er_deepfool_attack
                        uploaddic['robustness_score1'] = robustness_score
                        print("robustness Score is:", robustness_score)

                    if pillar[0] == 'methodology':
                        # print("Pillar methodology is:", pillar[1])
                        normalization = 0
                        missing_data = 0
                        regularization = 0
                        train_test_split = 0
                        factsheet_completeness = 0

                        if str(pillar[1]['normalization']) != 'nan':
                            normalization = int(pillar[1]['normalization'])*0.2

                        if str(pillar[1]['missing_data']) != 'nan':
                            missing_data = int(pillar[1]['missing_data'])*0.2

                        if str(pillar[1]['regularization']) != 'nan':
                            regularization = int(
                                pillar[1]['regularization'])*0.2

                        if str(pillar[1]['train_test_split']) != 'nan':
                            train_test_split = int(
                                pillar[1]['train_test_split'])*0.2

                        if str(pillar[1]['factsheet_completeness']) != 'nan':
                            factsheet_completeness = int(
                                pillar[1]['factsheet_completeness'])*0.2

                        methodology_score = normalization + missing_data + \
                            regularization + train_test_split + factsheet_completeness

                        uploaddic['normalization'] = normalization
                        uploaddic['missing_data'] = missing_data
                        uploaddic['regularization'] = regularization
                        uploaddic['train_test_split'] = train_test_split
                        uploaddic['factsheet_completeness'] = factsheet_completeness
                        uploaddic['methodology_score1'] = (
                            "%.2f" % methodology_score)
                        print("methodology Score is:", methodology_score)

                trust_score = fairness_score*0.25 + explainability_score * \
                    0.25 + robustness_score*0.25 + methodology_score*0.25
                uploaddic['trust_score1'] = trust_score
                print("Trust Score is:", trust_score)

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            config_weights = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
            factsheet = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution1']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']

            # solution_set_path,
            # path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
            # path_mapping_fairness=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
            print("Final Score result:", get_final_score1(path_module, path_traindata,
                  path_testdata, config_weights, mappings_config, path_factsheet))

            def get_final_score2(model, train_data, test_data, config_weights, mappings_config, factsheet, recalc=False):
                mappingConfig1 = mappings_config

                with open(mappings_config, 'r') as f:
                    mappings_config = json.loads(f.read())
                # mappings_config=pd.read_json(mappings_config)

                config_fairness = mappings_config["fairness"]
                config_explainability = mappings_config["explainability"]
                config_robustness = mappings_config["robustness"]
                config_methodology = mappings_config["methodology"]

                methodology_config = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
                config_explainability = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
                config_fairness = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/fairness/default.json')
                config_robustness = os.path.join(
                    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')

                def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config):
                    # if "scores" in factsheet.keys() and "properties" in factsheet.keys():
                    #     scores = factsheet["scores"]
                    #     properties = factsheet["properties"]
                    # else:
                    output = dict(
                        fairness=analyse_fairness(
                            model, train_data, test_data, factsheet, config_fairness),
                        explainability=analyse_explainability(
                            model, train_data, test_data, config_explainability, factsheet),
                        robustness=analyse_robustness(
                            model, train_data, test_data, config_robustness, factsheet),
                        methodology=analyse_methodology(
                            model, train_data, test_data, factsheet, methodology_config)
                    )
                    scores = dict((k, v.score) for k, v in output.items())
                    properties = dict((k, v.properties)
                                      for k, v in output.items())
                    # factsheet["scores"] = scores
                    # factsheet["properties"] = properties
                    # write_into_factsheet(factsheet, solution_set_path)

                    return result(score=scores, properties=properties)

                with open(mappingConfig1, 'r') as f:
                    default_map = json.loads(f.read())
                # default_map=pd.read_json(mappingConfig1)

                # factsheet=os.path.join(BASE_DIR,'media/' + str(factsheet))
                with open(factsheet, 'r') as g:
                    factsheet = json.loads(g.read())
                # factsheet=pd.read_json(factsheet)

                # print("mapping is default:")
                # print(default_map == mappings_config)
                if default_map == mappings_config:
                    if "scores" in factsheet.keys() and "properties" in factsheet.keys() and not recalc:
                        scores = factsheet["scores"]
                        properties = factsheet["properties"]
                else:
                    result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness,
                                                config_explainability, config_robustness, config_methodology)
                    scores = result.score
                    factsheet["scores"] = scores
                    properties = result.properties
                    factsheet["properties"] = properties
                # try:
                #     write_into_factsheet(factsheet, solution_set_path)
                # except Exception as e:
                #     print("ERROR in write_into_factsheet: {}".format(e))
                # else:
                #     result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology, solution_set_path)
                #     scores = result.score
                #     properties = result.properties

                final_scores = dict()
                # scores = tuple(scores)
                print("Scores is:", scores)
                with open(config_weights, 'r') as n:
                    config_weights = json.loads(n.read())
                # config_weights=pd.read_json(config_weights)
                # configdict = {}
                print("Config weight:", config_weights)

                fairness_score = 0
                explainability_score = 0
                robustness_score = 0
                methodology_score = 0
                for pillar in scores.items():
                    # print("Pillars:", pillar)
                    # pillar = {'pillar':pillar}

                    if pillar[0] == 'fairness':
                        # print("Pillar fairness is:", pillar[1])
                        uploaddic['underfitting2'] = int(
                            pillar[1]['underfitting'])
                        uploaddic['overfitting2'] = int(
                            pillar[1]['overfitting'])
                        uploaddic['statistical_parity_difference2'] = int(
                            pillar[1]['statistical_parity_difference'])
                        uploaddic['equal_opportunity_difference2'] = int(
                            pillar[1]['equal_opportunity_difference'])
                        uploaddic['average_odds_difference2'] = int(
                            pillar[1]['average_odds_difference'])
                        uploaddic['disparate_impact2'] = int(
                            pillar[1]['disparate_impact'])
                        uploaddic['class_balance2'] = int(
                            pillar[1]['class_balance'])

                        fairness_score = int(
                            pillar[1]['underfitting'])*0.35 + int(pillar[1]['overfitting'])*0.15
                        + int(pillar[1]['statistical_parity_difference'])*0.15 + \
                            int(pillar[1]['equal_opportunity_difference'])*0.2
                        + int(pillar[1]['average_odds_difference']) * \
                            0.1 + int(pillar[1]['disparate_impact'])*0.1
                        + int(pillar[1]['class_balance'])*0.1

                        uploaddic['fairness_score2'] = fairness_score
                        print("Fairness Score is:", fairness_score)

                    if pillar[0] == 'explainability':
                        # print("Pillar explainability is:", pillar[1]['algorithm_class'])
                        algorithm_class = 0
                        correlated_features = 0
                        model_size = 0
                        feature_relevance = 0

                        if str(pillar[1]['algorithm_class']) != 'nan':
                            algorithm_class = int(
                                pillar[1]['algorithm_class'])*0.55

                        if str(pillar[1]['correlated_features']) != 'nan':
                            correlated_features = int(
                                pillar[1]['correlated_features'])*0.15

                        if str(pillar[1]['model_size']) != 'nan':
                            model_size = int(pillar[1]['model_size'])*5

                        if str(pillar[1]['feature_relevance']) != 'nan':
                            feature_relevance = int(
                                pillar[1]['feature_relevance'])*0.15

                        explainability_score = algorithm_class + \
                            correlated_features + model_size + feature_relevance

                        uploaddic['algorithm_class2'] = algorithm_class
                        uploaddic['correlated_features2'] = correlated_features
                        uploaddic['model_size2'] = model_size
                        uploaddic['feature_relevance2'] = feature_relevance

                        uploaddic['explainability_score2'] = explainability_score
                        print("explainability Score is:", explainability_score)

                    if pillar[0] == 'robustness':
                        # print("Pillar robustness is:", pillar[1])

                        confidence_score = 0
                        clique_method = 0
                        loss_sensitivity = 0
                        clever_score = 0
                        er_fast_gradient_attack = 0
                        er_carlini_wagner_attack = 0
                        er_deepfool_attack = 0

                        if str(pillar[1]['confidence_score']) != 'nan':
                            confidence_score = int(
                                pillar[1]['confidence_score'])*0.2

                        if str(pillar[1]['clique_method']) != 'nan':
                            clique_method = int(pillar[1]['clique_method'])*0.2

                        if str(pillar[1]['loss_sensitivity']) != 'nan':
                            loss_sensitivity = int(
                                pillar[1]['loss_sensitivity'])*0.2

                        if str(pillar[1]['clever_score']) != 'nan':
                            clever_score = int(pillar[1]['clever_score'])*0.2

                        if str(pillar[1]['er_fast_gradient_attack']) != 'nan':
                            er_fast_gradient_attack = int(
                                pillar[1]['er_fast_gradient_attack'])*0.2

                        if str(pillar[1]['er_carlini_wagner_attack']) != 'nan':
                            er_carlini_wagner_attack = int(
                                pillar[1]['er_carlini_wagner_attack'])*0.2

                        if str(pillar[1]['er_deepfool_attack']) != 'nan':
                            er_deepfool_attack = int(
                                pillar[1]['er_deepfool_attack'])*0.2

                        robustness_score = confidence_score + clique_method + loss_sensitivity + \
                            clever_score + er_fast_gradient_attack + \
                            er_carlini_wagner_attack + er_deepfool_attack

                        uploaddic['confidence_score2'] = confidence_score
                        uploaddic['clique_method2'] = clique_method
                        uploaddic['loss_sensitivity2'] = loss_sensitivity
                        uploaddic['clever_score2'] = clever_score
                        uploaddic['er_fast_gradient_attack2'] = er_fast_gradient_attack
                        uploaddic['er_carlini_wagner_attack2'] = er_carlini_wagner_attack
                        uploaddic['er_deepfool_attack2'] = er_deepfool_attack
                        uploaddic['robustness_score2'] = robustness_score
                        print("robustness Score is:", robustness_score)

                    if pillar[0] == 'methodology':
                        # print("Pillar methodology is:", pillar[1])
                        normalization = 0
                        missing_data = 0
                        regularization = 0
                        train_test_split = 0
                        factsheet_completeness = 0

                        if str(pillar[1]['normalization']) != 'nan':
                            normalization = int(pillar[1]['normalization'])*0.2

                        if str(pillar[1]['missing_data']) != 'nan':
                            missing_data = int(pillar[1]['missing_data'])*0.2

                        if str(pillar[1]['regularization']) != 'nan':
                            regularization = int(
                                pillar[1]['regularization'])*0.2

                        if str(pillar[1]['train_test_split']) != 'nan':
                            train_test_split = int(
                                pillar[1]['train_test_split'])*0.2

                        if str(pillar[1]['factsheet_completeness']) != 'nan':
                            factsheet_completeness = int(
                                pillar[1]['factsheet_completeness'])*0.2

                        methodology_score = normalization + missing_data + \
                            regularization + train_test_split + factsheet_completeness

                        uploaddic['normalization2'] = normalization
                        uploaddic['missing_data2'] = missing_data
                        uploaddic['regularization2'] = regularization
                        uploaddic['train_test_split2'] = train_test_split
                        uploaddic['factsheet_completeness2'] = factsheet_completeness
                        uploaddic['methodology_score2'] = (
                            "%.2f" % methodology_score)
                        print("methodology Score is:", methodology_score)

                trust_score = fairness_score*0.25 + explainability_score * \
                    0.25 + robustness_score*0.25 + methodology_score*0.25
                uploaddic['trust_score2'] = trust_score
                print("Trust Score is:", trust_score)

            path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
            path_traindata = os.path.join(
                BASE_DIR, 'apis/TestValues/train.csv')
            path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
            path_factsheet = os.path.join(
                BASE_DIR, 'apis/TestValues/factsheet.json')
            config_weights = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Weights/default.json')
            mappings_config = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')
            factsheet = os.path.join(
                BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution2']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        path_traindata = i['training_file']
                        path_factsheet = i['factsheet_file']

            # solution_set_path,
            # path_mapping_accountabiltiy=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/Accountability/default.json')
            # path_mapping_fairness=os.path.join(BASE_DIR,'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
            print("Final Score result:", get_final_score2(path_module, path_traindata,
                  path_testdata, config_weights, mappings_config, path_factsheet))

        return Response(uploaddic)


# class dataList1(APIView):
#     def get(self,request,id):
#         stockInfoDic={}
#         indexDic={}

#         ################# User data ####################
#         # stockNames=['GME','AAPL','AMC']
#         # indices=['^GSPC','^W5000']
#         # quantitiesPurchased=[15, 2, 10]
#         # # prchsdDts=['2020-02-19','2021-12-17','2022-01-12','2035-02-21']
#         # prchsdDts= []
#         ################################################

#         stockNames=['AMZN','TSLA','PYPL','BABA']
#         indices=['^GSPC','^W5000']
#         quantitiesPurchased=[1,2,3,1]
#         prchsdDts=[]
#         uploadDict = {}
#         # print("stocknames before",stockNames)
#         user=CustomUser.objects.get(user_id=id)
#         print('User Get=====>',user)
#         stock_data=FirstStock.objects.filter(user=user).order_by('-id')[:1]
#         print("User stock:",stock_data)
#         serializer =FirstStockSerializer2(stock_data,many=True)
#         # print("Serializer data:",serializer.data)
#         # serializer = FirstStockSerializer(stock_data,many=True)

#         corrlist = []
#         dailyreturnsportf = []
#         PreviousInv = []
#         PercentageAR = []
#         PercentAR_t = []
#         SP500listt = []
#         W5000listt = []

#         TotalInvestment = []
#         TotalReturn = []

#         FullLoc1 = []
#         FullLoc0 = []
#         Full1Loc1 = []
#         Full1Loc0 = []
#         if stock_data is not None:
#             for i in stock_data:
#                 # print("response Data:",i.response_data['PreviousInv'])
#                 stockNames = i.form_data['stockNames']
#                 # prchsdDts = i.form_data['purchaseDate']
#                 # print("prchsdDts:",i.form_data['purchaseDate'])
#                 quantitiesPurchased = i.form_data['quantitiesPurchased']
#                 for j in  i.form_data['purchaseDate']:
#                     prchsdDts.append(dt.strptime(j, "%d/%m/%Y").strftime("%Y-%m-%d"))


#                 corrlist = i.response_data['corrlist']
#                 dailyreturnsportf = i.response_data['dailyreturnsportf']
#                 PreviousInv = i.response_data['PrevInvestments']
#                 PercentageAR = i.response_data['PercentageAR']

#                 TotalInvestment = int(i.response_data['TotalInvestment'])
#                 TotalReturn = int(i.response_data['TotalReturn'])

#                 PercentAR_t = i.response_data['PercentARlistt']
#                 SP500listt = i.response_data['SP500listt']
#                 W5000listt = i.response_data['W5000listt']

#                 FullLoc1 = i.response_data['FullLoc1']
#                 FullLoc0 = i.response_data['FullLoc0']
#                 Full1Loc1 = i.response_data['Full1Loc1']
#                 Full1Loc0 = i.response_data['Full1Loc0']

#         # #             # print("Dates:", dt.strptime(j, "%d/%m/%Y").strftime("%Y-%m-%d"))
#         # #         # print("Stock Data:",i.form_data['stockNames'])
#         # # # print("Serializer Data:",serializer.data)
#         print("stocknames after",stockNames)
#         uploadDict['stockNames'] = stockNames
#         uploadDict['quantitiesPurchased'] = quantitiesPurchased
#         uploadDict['prchsdDts'] = prchsdDts
#         uploadDict['corrlist'] = corrlist
#         uploadDict['dailyreturnsportf'] = dailyreturnsportf
#         uploadDict['PreviousInv'] = PreviousInv
#         uploadDict['PercentageAR'] = PercentageAR
#         uploadDict['PercentAR_t'] = PercentAR_t
#         uploadDict['SP500listt'] = SP500listt
#         uploadDict['W5000listt'] = W5000listt
#         uploadDict['FullLoc1'] = FullLoc1
#         uploadDict['FullLoc0'] = FullLoc0
#         uploadDict['Full1Loc1'] = Full1Loc1
#         uploadDict['Full1Loc0'] = Full1Loc0
#         uploadDict['TotalInvestment'] = TotalInvestment
#         uploadDict['TotalReturn'] = TotalReturn


#         print("Correlation:",corrlist)
#         return Response(uploadDict)

#     def post(self, request):
#         # serializer = DataSerializer(data=request.data,many=False)
#         print('successfull id is:',request.data['user'])
#         stockInfoDic = {}
#         indexDic={}
#         print(request.data)
#         uploaddic = {}
#         # stockNames=request.data['form_data']['stockNames']
#         # quantitiesPurchased=request.data['form_data']["quantitiesPurchased"]
#         # prchsdDts= dt.strptime(request.data['form_data']["purchaseDate"], "%d/%m/%Y").strftime("%Y-%m-%d")
#         stockNames=[]
#         indices=['^GSPC','^W5000']
#         quantitiesPurchased=[]
#         prchsdDts=[]
#         lst = []
#         # print("stocknames before",stockNames)
#         user=CustomUser.objects.get(user_id=request.data['user'])
#         print('User=====>',user)
#         stock_data=FirstStock.objects.filter(user=user).order_by('-id')[:1]
#         serializer =FirstStockSerializer2(stock_data,many=True)
#         # serializer = FirstStockSerializer(stock_data,many=True)
#         if request.data is not None:
#             for i, j, k in zip(request.data['form_data']['stockNames'], request.data['form_data']['purchaseDate'], request.data['form_data']['quantitiesPurchased']):
#                 print("DATA:",i, j, k)
#                 stockNames.append(i)
#                 prchsdDts.append(dt.strptime(j, "%d/%m/%Y").strftime("%Y-%m-%d"))
#                 quantitiesPurchased.append(k)
#                 # stockNames = i.form_data
#                 # # print("Stock Data:",i.form_data)
#                 # stockNames = i.form_data['stockNames']
#                 # # prchsdDts = i.form_data['purchaseDate']
#                 # # print("prchsdDts:",i.form_data['purchaseDate'])
#                 # quantitiesPurchased = i.form_data['quantitiesPurchased']
#                 # for j in  i.form_data['purchaseDate']:
#                 #     prchsdDts.append(dt.strptime(j, "%d/%m/%Y").strftime("%Y-%m-%d"))
#         #             # print("Dates:", dt.strptime(j, "%d/%m/%Y").strftime("%Y-%m-%d"))
#         #         # print("Stock Data:",i.form_data['stockNames'])
#         # # print("Serializer Data:",serializer.data)
#         print("stocknames after",stockNames)
#         uploaddic['stockNames'] = stockNames

#         print("Quantities:",quantitiesPurchased)
#         uploaddic['Quantities'] = quantitiesPurchased

#         print("Dates are:",prchsdDts)
#         uploaddic['prchsdDts'] = prchsdDts

#         if len(stockNames) != len(quantitiesPurchased) or len(stockNames) != len(prchsdDts):
#             return "error, incompatible number of variables input!"

#         for p in range(len(prchsdDts)):
#             if dt.strptime(prchsdDts[p],"%Y-%m-%d").weekday() == 5 or dt.strptime(prchsdDts[p],"%Y-%m-%d").weekday() == 6:
#                 prchsdDts[p]=(dt.strptime(prchsdDts[p],"%Y-%m-%d")-BDay(1)).strftime("%Y-%m-%d")
#             else:
#                 pass

#         ########## last 2 graphs corr_matrix   daily_returns_portf  ################
#         Min_date=min(prchsdDts)
#         Max_date=dt.today()
#                 # qnttsPrchsd = list(map(lambda x: int(x), quantitiesPurchased))
#         qnttsPrchsd = [int(x) for x in quantitiesPurchased]

#         stocksDF = yf.download(tickers = stockNames,start=Min_date,end=Max_date)['Close']
#         portf=stocksDF.sum(axis=1)
#         daily_returns_portf=portf.pct_change(1)*100
#         daily_returns = stocksDF.pct_change(1)
#         corr_matrix=daily_returns.corr()

#         # print("corr_matrix",corr_matrix)
#         corrlist = []
#         for i in corr_matrix:
#             print("corr_matrix",list(corr_matrix[i]))
#             corrlist.append(list(corr_matrix[i]))
#         print("corrlist:",corrlist)
#         uploaddic['corrlist'] = corrlist
#         # print("corr_matrix",list(corr_matrix['AAPL']))
#         # print("corr_matrix",list(corr_matrix['AMC']))
#         # print("corr_matrix",list(corr_matrix['GME']))

#         uploaddic['dailyreturnsportf'] = list(daily_returns_portf)[1:]
#         # print("daily_returns_portf",list(daily_returns_portf))

#         # # Min_date=min(prchsdDts)
#         # # Max_date=dt.today()
#         # #             # qnttsPrchsd = list(map(lambda x: int(x), quantitiesPurchased))
#         # # qnttsPrchsd = [int(x) for x in quantitiesPurchased]

#         # # stocksDF = yf.download(tickers = stockNames,start=Min_date,end=Max_date)['Close']
#         # # if stocksDF.empty:
#         # #     return "Error! no market data available for today"

#         # # daily_returns = stocksDF.pct_change(1)


#         for SN, PD, QP in zip(stockNames, prchsdDts, quantitiesPurchased):

#             Min_date=min(prchsdDts)
#             Max_date=dt.today()
#                     # qnttsPrchsd = list(map(lambda x: int(x), quantitiesPurchased))
#             qnttsPrchsd = [int(x) for x in quantitiesPurchased]

#             stocksDF = yf.download(tickers = stockNames,start=Min_date,end=Max_date)['Close']

#             if stocksDF.empty:
#                 return "Error! no market data available for today"

#             daily_returns = stocksDF[SN].pct_change(1)
#             standard_deviation=np.std(daily_returns)*100

#             stock = yf.Ticker(SN)
#             PD_=dt.strptime(PD,"%Y-%m-%d").strftime("%Y-%m-%d")
#             # retrieving stock information for purchaseDate
#             df_stock = stock.history(start=PD_,
#                 end=dt.strptime(PD_,"%Y-%m-%d") + timedelta(days=1))
#                     #print(df, PD, type(PD))

#             stockValuePD = df_stock.loc[PD_,"Close"]
#                         # print("stockvaluePD",stockValuePD, stockValuePD.corr())
#                         # return Response("Hello")

#             FSV0 = stockValuePD * QP

#             try:
#                 present_date = dt.today()-BDay(1)
#                 #print('Present date: ', present_date)
#                 stockValueToday = stocksDF[SN].loc[present_date.strftime("%Y-%m-%d")]
#                 #print('stockvalueToday',stockValueToday)

#             #except KeyError:
#             #    return "errMessage Error! no market data available for today"
#             except Exception as ex:
#                 template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#                 message = template.format(type(ex).__name__, ex.args)
#                 return message
#                 print("Usman")

#                     # calculating FSV1
#             FSV1 = stockValueToday * QP

#                     # claculating stock return
#             R = FSV1-FSV0

#                     # calculation of percentage return
#             percentAR = (R/FSV0)*100


#             stockInfoDic[SN] = {"FSV0": FSV0, "FSV1": FSV1, "StockName": SN,"PurchaseDate":PD,
#                     "PercentAR": percentAR, "R": R,"StandardDev":standard_deviation}

#             TotalInvestment = 0
#             for x in stockInfoDic.values():
#                 TotalInvestment = TotalInvestment+x["FSV0"]
#                 x['perc_Inv']=100*x['FSV0']/TotalInvestment


#             ###########  4th Bar graph  ####################
#             TotalInvestment = 0
#             TotalReturn=0
#             for x in stockInfoDic.values():
#                 TotalInvestment = TotalInvestment+x["FSV0"]
#                 TotalReturn = TotalReturn+x["R"]
#                 # Calculation of percentage return compared to total investment
#             for x in stockInfoDic.values():
#                 PR = (x["FSV0"]/TotalInvestment)*100
#                 x["prTotalI"] = PR

#             for x in stockInfoDic.values():

#                 indicesDic={}
#                 for i in indices:
#                     indexDF = yf.download(tickers = indices,start=Min_date,end=Max_date)['Close']
#                     index = yf.Ticker(i)

#                     df_index = index.history(start=x['PurchaseDate'],end=dt.strptime(x['PurchaseDate'],"%Y-%m-%d") + timedelta(days=1))
#                     FSV0_index = df_index.loc[x['PurchaseDate'],"Close"]

#                     FSV1_index = indexDF[i].loc[present_date.strftime("%Y-%m-%d")]

#                     R_index = FSV1_index-FSV0_index

#                     percentAR_index = (R_index/FSV0_index)*100

#                     indicesDic[i]=percentAR_index

#                 x['SP500']=indicesDic['^GSPC']
#                 x['W5000']=indicesDic['^W5000']

#         print("TotalInvestment is:",TotalInvestment)
#         uploaddic['TotalInvestment'] = TotalInvestment
#         print("TotalReturn is:",TotalReturn)
#         uploaddic['TotalReturn'] = TotalReturn
#         perc_Invs=[]

#         for i in stockInfoDic.keys():
#             perc_Invs.append(stockInfoDic[i]['perc_Inv'])

#         perc_returns=[]

#         for i in stockInfoDic.keys():
#             perc_returns.append(stockInfoDic[i]['PercentAR'])


#         print("PrevInvestments:",perc_Invs)
#         uploaddic['PrevInvestments'] = perc_Invs

#         print("PercentageAR:",perc_returns)
#         uploaddic['PercentageAR'] = perc_returns

#         # print("stock Names:", stockNames)
#         # uploaddic['perc_returns'] = perc_returns

#         x = np.arange(len(stockNames))  # the label locations
#         width = 0.25  # the width of the bars
#         #fig, ax = plt.subplots()

#         PercentAR_list=[]
#         SP500_list=[]
#         W5000_list=[]

#         for i in stockInfoDic.keys():
#             PercentAR_list.append(stockInfoDic[i]['PercentAR'])
#             SP500_list.append(stockInfoDic[i]['SP500'])
#             W5000_list.append(stockInfoDic[i]['W5000'])

#         PercentAR_list_t=np.array(PercentAR_list).T
#         SP500_list_t=np.array(SP500_list).T
#         W5000_list_t=np.array(W5000_list).T

#         PercentAR_t = []
#         SP500_t = []
#         W5000_t = []

#         for i in range(len(PercentAR_list_t)):
#             PercentAR_t.append(PercentAR_list_t[i])
#             SP500_t.append(SP500_list_t[i])
#             W5000_t.append(W5000_list_t[i])

#         print("PercentAR_list_t:",PercentAR_t)
#         uploaddic['PercentARlistt'] = PercentAR_t

#         print("SP500_list_t:",SP500_t)
#         uploaddic['SP500listt'] = SP500_t

#         print("W5000_list_t:",W5000_t)
#         uploaddic['W5000listt'] = W5000_t

#         # uploaddic['PercentAR_list_t'] = PercentAR_list_t

#         # uploaddic['SP500_list_t'] = SP500_list_t

#         # uploaddic['W5000_list_t'] = W5000_list_t
#                 ##############  4th bar graph End ########################

#         ############# 1st scatter Graph  ###################
#         # perc_returns=[]
#         returns_list=[]
#         std_list=[]

#         for i in stockInfoDic.keys():

#             returns_list.append(stockInfoDic[i]['PercentAR'])
#             std_list.append(stockInfoDic[i]['StandardDev'])

#             full=pd.concat((pd.DataFrame(returns_list).T,pd.DataFrame(std_list).T),axis=0)

#         full.columns=stockInfoDic.keys()

#         print("Full Loc 1:",full.iloc[1])
#         print("Full Loc 0:",full.iloc[0])

#         print("Full Loc 1:",list(full.iloc[1]))
#         uploaddic['FullLoc1'] = list(full.iloc[1])

#         print("Full Loc 0:",list(full.iloc[0])) ### For Scatter
#         uploaddic['FullLoc0'] = list(full.iloc[0])

#         ######################### 1st scatter End ###################################

#         ################  Second Scatter Graph  #############################################
#         for SN, I, PD, QP in zip(stockNames,indices, prchsdDts, quantitiesPurchased):

#             Min_date=min(prchsdDts)
#             Max_date=dt.today()
#                     # qnttsPrchsd = list(map(lambda x: int(x), quantitiesPurchased))
#             qnttsPrchsd = [int(x) for x in quantitiesPurchased]

#             stocksDF = yf.download(tickers = stockNames,start=Min_date,end=Max_date)['Close']
#             indexDF=yf.download(tickers = indices,start=Min_date,end=Max_date)['Close']

#             daily_returns_index = indexDF[I].pct_change(1)
#             standard_deviation_index=np.std(daily_returns_index)*100

#             portf=stocksDF.sum(axis=1)
#             daily_returns_portf=portf.pct_change(1)
#             standard_deviation_portf=np.std(daily_returns_portf)*100

#             if stocksDF.empty:
#                 return "Error! no market data available for today"

#             stock = yf.Ticker(SN)
#             index = yf.Ticker(I)
#             PD_=dt.strptime(PD,"%Y-%m-%d").strftime("%Y-%m-%d")
#             # retrieving stock information for purchaseDate
#             df_stock = stock.history(start=PD_,
#                 end=dt.strptime(PD_,"%Y-%m-%d") + timedelta(days=1))
#                     #print(df, PD, type(PD))
#             df_index = index.history(start=PD_,
#                 end=dt.strptime(PD_,"%Y-%m-%d") + timedelta(days=1))

#             stockValuePD = df_stock.loc[PD_,"Close"]
#             indexValuePD = df_index.loc[PD_,"Close"]

#             FSV0 = stockValuePD * QP
#             FSV0_index = indexValuePD

#             try:
#                 present_date = dt.today()-BDay(1)
#             #print('Present date: ', present_date)
#                 stockValueToday = stocksDF[SN].loc[present_date.strftime("%Y-%m-%d")]
#             #print('stockvalueToday',stockValueToday)
#                 indexValueToday = indexDF[I].loc[present_date.strftime("%Y-%m-%d")]

#             #except KeyError:
#             #    return "errMessage Error! no market data available for today"
#             except Exception as ex:
#                 template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#                 message = template.format(type(ex).__name__, ex.args)
#                 return message
#                 print("Usman")

#                     # calculating FSV1
#             FSV1 = stockValueToday * QP
#             FSV1_index = indexValueToday

#                     # claculating stock return
#             R = FSV1-FSV0
#             R_index = FSV1_index-FSV0_index

#                     # calculation of percentage return
#             percentAR = (R/FSV0)*100
#             percentAR_index = (R_index/FSV0_index)*100


#             stockInfoDic[SN] = {"FSV0": FSV0, "FSV1": FSV1, "StockName": SN,"PurchaseDate":PD,
#                     "PercentAR": percentAR, "R": R}

#             indexDic[I]={"PercentAR":percentAR_index,"StandardDev":standard_deviation_index}


#             for x in stockInfoDic.values():

#                 indicesDic={}
#                 for i in indices:
#                     indexDF = yf.download(tickers = indices,start=Min_date,end=Max_date)['Close']
#                     index = yf.Ticker(i)

#                     df_index = index.history(start=x['PurchaseDate'],end=dt.strptime(x['PurchaseDate'],"%Y-%m-%d") + timedelta(days=1))
#                     FSV0_index = df_index.loc[x['PurchaseDate'],"Close"]

#                     FSV1_index = indexDF[i].loc[present_date.strftime("%Y-%m-%d")]

#                     R_index = FSV1_index-FSV0_index

#                     percentAR_index = (R_index/FSV0_index)*100

#                     indicesDic[i]=percentAR_index

#                 x['SP500']=indicesDic['^GSPC']
#                 x['W5000']=indicesDic['^W5000']

# ##################################################################
#         # returns_list=[]
#         # std_list=[]

#         # for i in stockInfoDic.keys():

#         #     returns_list.append(stockInfoDic[i]['PercentAR'])
#         #     std_list.append(stockInfoDic[i]['StandardDev'])

#         #     full=pd.concat((pd.DataFrame(returns_list).T,pd.DataFrame(std_list).T),axis=0)

#         # full.columns=stockInfoDic.keys()
#         # print("Full Loc 1:",full.iloc[1])
#         # print("Full Loc 0:",full.iloc[0]) ### For Scatter
# ###################################################################################

#         for x in stockInfoDic.values():
#             PercentAR_portf=100*(x['FSV1'].sum()-x['FSV0'].sum())/x['FSV0'].sum()

#         returns_list=[]
#         std_list=[]

#         indexDic["SP500"] = indexDic.pop("^GSPC")
#         indexDic["W5000"] = indexDic.pop("^W5000")

#         for i in indexDic.keys():

#             returns_list.append(indexDic[i]['PercentAR'])
#             std_list.append(indexDic[i]['StandardDev'])

#             full=pd.concat((pd.DataFrame(returns_list).T,pd.DataFrame(std_list).T),axis=0,ignore_index=True)

#         full.columns=indexDic.keys()
#         portf=pd.DataFrame([PercentAR_portf,standard_deviation_portf]).rename(columns={0:'Portfolio'})
#         full1=pd.concat((full,portf),axis=1)
#         print("full1.iloc1",full1.iloc[1])
#         print("full1.iloc0",full1.iloc[0])


#         uploaddic['Full1Loc1'] = list(full1.iloc[1])
#         uploaddic['Full1Loc0'] = list(full1.iloc[0])
#         # # # # for i in stockInfoDic.keys():

#         # # # #     perc_returns.append(stockInfoDic[i]['PercentAR'])
#         # # # print("PercentageAR:",returns_list)
#         # # # print("stock Names:", stockNames)

#         ############################# 2nd Scatter Graph END #########################################
#         # print("Full Loc 1:",full.iloc[1])
#         # print("Full Loc 0:",full.iloc[0]) ### For Scatter
#         # stockInfoDic = {}
#         # stockInfoDic['stocknames'] = stockNames   ## For Hbar and Scatter
#         # stockInfoDic['Percentage'] = returns_list  ## for Hbar and Scatter

#         # ser = FirstStockSerializer(perc_Invs,many=True)

#         user=CustomUser.objects.get(user_id=request.data['user'])
#         request.data['user'] = user.id
#         serializer=FirstStockSerializer(data=request.data)
#         print("Serializer:",serializer)
#         print("UploadDict:",uploaddic)
#         if serializer.is_valid():
#             serializer.save(response_data=uploaddic)
#             # serializer.save(response_data={"PreviousInv": perc_Invs, "PercentageAR": perc_returns, "StockNames": stockNames, "PercentARlistT": PercentAR_t, "SP500listt": SP500_t, "W5000listt": W5000_t, "corrlist": corrlist, "dailyreturnportf": list(daily_returns_portf), "fullloc1": list(full.iloc[1]), "fullloc0": list(full.iloc[0]), "full1loc1": list(full1.iloc[1]), "full1loc0": list(full1.iloc[0])})
#             print("Nice to Add Second!")
#             # return Response(stockInfoDic)
#             return Response(uploaddic)

#         else:
#             print('errors:  ',serializer.errors)

#         # return Response(stockInfoDic)
#         return Response(uploaddic)


# class dataList2(APIView):
#     def get(self,request,id):
#         user=CustomUser.objects.get(user_id=id)
#         stock_data=SecondStock.objects.filter(user=user).order_by('-id')[:1]
#         serializer = SecondStockSerializer2(stock_data,many=True)

#         return Response(serializer.data)

#     def post(self, request):
#         print('into datalist2 ===============================>')
#         # serializer = DataSerializer(data=request.data,many=False)
#         # print(request.data)
#         # user=request.data["user"]
#         # print("user",user)
#         #form_data
#         stockNames=request.data['form_data']['stockNames']
#         quantitiesPurchased=[request.data['form_data']["quantitiesPurchased"]]
#         interval=request.data['form_data']["interval"]

#         if interval.capitalize() == "D":
#             interval = "1d"
#         elif interval.capitalize() == "M":
#             interval = "1mo"
#             return Response("M")
#         elif interval.capitalize() == "W":
#             interval = "1wk"
#         else:
#             errMessage = "Invalid interval type specified in the request"
#             return Response("Invalid")
#         myDf = yf.download(tickers=" ".join(stockNames),  start=datetime.date(datetime.now() - relativedelta(months=14)),
#                        end=datetime.date(datetime.now(
#                        )), group_by="ticker", interval=interval)

#         if myDf.empty:
#             return Response("Error! no market data available for today")


#         srs = []

#         for x in stockNames:
#             s = myDf[x]["Close"].rename("Close "+x)
#             # return Response("success")
#             srs.append(s)

#         ogDf = pd.concat(srs, axis=1)

#        # return Response('here')
#         if interval == "1wk":
#             ogDf = ogDf[ogDf.index.dayofweek == 0]
#         elif interval == "1mo":
#             ogDf = ogDf[ogDf.index.day == 1]

#         df = ogDf.rolling(2).apply(myFunc)


#         means = df.mean()
#         variances = df.var()
#         stds = df.std()*100

#         resultDic = {}
#         for x in stockNames:
#             mean = means["Close {}".format(
#             x)]
#             variance = variances["Close {}".format(x)]
#             std = stds["Close {}".format(x)]
#             posRan = mean+(mean*std)
#             negRan = mean-(mean*std)

#             result = {"Mean": mean, "variances": variance,
#                   "stds": std, "posEnd": posRan, "negEnd": negRan}

#             if math.isnan(result['Mean']):
#                 result['Mean'] = ''
#             if math.isnan(result['variances']):
#                 result['variances'] = ''
#             if math.isnan(result['stds']):
#                 result['stds'] = ''
#             if math.isnan(result['posEnd']):
#                 result['posEnd'] = ''
#             if math.isnan(result['negEnd']):
#                 result['negEnd'] = ''
#             resultDic[x] = result


#         df.dropna(axis=1, how="all", inplace=True)
#         ogDf.dropna(axis=1, how="all", inplace=True)
#         dropNaDf = ogDf.dropna(axis=1, how="any")
#         newDf = pd.DataFrame(index=dropNaDf.index.copy())
#         newDf['totalSumLog'] = np.log(dropNaDf.loc[:, :].sum(axis=1))
#         newDf = newDf.rolling(2).apply(myFunc2)
#         newDf.dropna(inplace=True)
#         rng = np.random.default_rng()
#         rsame = rng.choice(newDf, size=100000, replace=True)
#         VaR = float(np.quantile(rsame, 0.05))
#         VarOut = VaR*100


#     # RStar = np.percentile(retVec,100.*p)
#         Es = float(np.mean(newDf[newDf <= VaR]))
#         esHist = 100*Es

#         resultDic['var'] = VaR
#         resultDic['var_out'] = VarOut
#         resultDic['es'] = Es
#         resultDic['es_hist'] = esHist


#         stocksV = []
#         for SN, QP in zip(stockNames, quantitiesPurchased):
#             try:
#                 stocksV.append(dropNaDf.iloc[-1:]["Close "+SN][0]*QP)
#             except KeyError:
#                 pass
#         VaR = np.quantile(rsame, 0.05)
#         VarOut = VaR*100
#         #RStar = np.percentile(retVec,100.*p)
#         Es = np.mean(newDf[newDf <= VaR])
#         esHist = 100*Es
#         stocksV = []
#         for SN, QP in zip(stockNames, quantitiesPurchased):
#             try:
#                 stocksV.append(dropNaDf.iloc[-1:]["Close "+SN][0]*QP)
#             except KeyError:
#                 pass

#         portfolio_value = sum(stocksV)
#         VaR_value = portfolio_value*(math.e**VaR-1)
#         ES_value = portfolio_value*(math.e**esHist["totalSumLog"]-1)
#         # RDSB[]
#         user=CustomUser.objects.get(user_id=request.data['user'])
#         request.data['user'] = user.id
#         serializer=SecondStockSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save(response_data=resultDic)
#             print("Nice to Add Second!")

#         else:
#             print('errors:  ',serializer.errors)

#         print(resultDic, 'resultDic')

#         return Response(resultDic)


# @api_view(['GET'])
# def marketComparison(request):
#     print("Market Comparison")
#     myDf = yf.download(tickers="^W5000  ^GSPC",  start=datetime.date(datetime.now() - relativedelta(days=5)),
#                        end=datetime.date(datetime.now(
#                        )), group_by="ticker")
#     print("myDf====>",myDf)
#     myDf = myDf.to_dict('records')
#     newData = []
#     for data in myDf:
#         print(data, 'data')
#         newDict = {}
#         for subData, values in data.items():
#             print(subData, values, 'keys')
#             val = ''
#             for key in subData:
#                 val = val + key + ' '
#             newDict[val] = data[subData]
#             print(subData, val)
#         newData.append(newDict)
#     return Response(newData)


def myFunc(x):
    return (x[-1]/x[0])-1
    return Response("success")


def myFunc2(x):
    return x[-1]-x[0]
    return Response("success")


def save_files_return_paths(*args):
    fs = django.core.files.storage.FileSystemStorage()
    paths = []
    for arg in args:
        if arg is None:
            paths.append(None)
        else:
            # file_name = fs.save(arg.name, arg)
            print('arg:', fs.path(arg))
            paths.append(fs.path(arg))
    return tuple(paths)


def handle_request(request, function_name):
    print("REQUEST DATA: ", request.data)
    solution_type = request.data['solution_type']
    if solution_type not in ['supervised', 'unsupervised']:
        return Response({'error': 'Invalid solution type.'}, status=400)

    try:
        model_file = request.FILES['model_file']
        try:
            training_dataset_file = request.FILES['training_dataset']
            factsheet_file = request.FILES['factsheet']

        except:
            pass

        test_dataset_file = request.FILES['test_dataset']
        mappings_file = request.FILES['mappings']
        try:
            thresholds = request.data['thresholds']
        except:
            pass

    except KeyError:
        return Response({'error': 'Missing file(s).'}, status=400)

    model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path = save_files_return_paths(
        model_file, training_dataset_file, test_dataset_file, factsheet_file, mappings_file)
    if None in (model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path):
        return Response({'error': 'Error saving file(s).'}, status=400)

    if solution_type == 'supervised':
        function_name = get_function_name(function_name, '_supervised')
    else:
        function_name = get_function_name(function_name, '_unsupervised')

    function = globals()[function_name]

    try:
        result = function(model=model_path, training_dataset=training_dataset_path,
                          test_dataset=test_dataset_path, factsheet=factsheet_path, mappings=mappings_path)
    except:
        result = function(model=model_path, training_dataset=training_dataset_path, test_dataset=test_dataset_path,
                          factsheet=factsheet_path, mappings=mappings_path, thresholds=thresholds)

    return Response({'score': result.score, 'properties': result.properties}, status=200)


def handle_score_request(type, detailType,  data, user_id):
    solution_name = data['solution_name']
    # solution_type = data['solution_type']

    solution = ScenarioSolution.objects.filter(
        solution_name=solution_name,
        user_id=user_id).values().order_by('id')[:1]

    path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
    path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
    path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')

    if solution:
        for i in solution:
            print('i:', i)
            path_testdata = i["test_file"]
            path_traindata = i["training_file"]
            path_factsheet = i["factsheet_file"]
            path_mapping = i["metrics_mappings_file"]
            solutionType = i['solution_type']
            model_file = i['model_file']
            target_column = i['target_column']
            outliers_data = i['outlier_data_file']

        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data, target_column = save_files_return_paths(
            model_file, path_traindata, path_testdata, path_factsheet, path_mapping, outliers_data, target_column)

        status = 200
        if (type == 'account'):
            if (detailType == 'factsheet'):

                if (solutionType == 'supervised'):
                    result = get_factsheet_completeness_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    result = get_factsheet_completeness_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'missingdata'):
                if (solutionType == 'supervised'):
                    result = get_missing_data_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    result = get_missing_data_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'normalization'):
                if (solutionType == 'supervised'):
                    result = get_normalization_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    result = get_normalization_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'regularization'):
                if (solutionType == 'supervised'):
                    result = get_regularization_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    result = get_regularization_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'train_test'):
                if (solutionType == 'supervised'):
                    result = get_train_test_split_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    result = get_train_test_split_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            else:
                result = 'none'
                status = 201

        elif (type == 'robust'):

            if (detailType == 'clever_score'):
                if (solutionType == 'supervised'):
                    result = get_clever_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    result = get_clever_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'clique_method_score'):
                if (solutionType == 'supervised'):
                    result = get_clique_method_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                else:
                    status = 201
                    result = "clique method isn't applicable for unsupervised solutions"

            elif (detailType == 'confidence_score'):
                if (solutionType == 'supervised'):
                    result = get_confidence_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    status = 201
                    result = "confidence score isn't applicable for unsupervised solutions"

            elif (detailType == 'carliwagnerwttack_score'):
                if (solutionType == 'supervised'):
                    result = get_carliwagnerwttack_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    status = 201
                    result = "carliwagnerwttack score isn't applicable for unsupervised solutions"

            elif (detailType == 'loss_sensitivity_score'):
                if (solutionType == 'supervised'):
                    result = get_loss_sensitivity_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    status = 201
                    result = "carliwagnerwttack score isn't applicable for unsupervised solutions"

        elif (type == 'explain'):
            if (detailType == 'modelsize_score'):
                if (solutionType == 'supervised'):
                    result = get_modelsize_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                else:
                    result = get_modelsize_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'correlated_features_score'):
                if (solutionType == 'supervised'):
                    result = get_correlated_features_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                else:
                    result = get_correlated_features_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

            elif (detailType == 'algorithm_class_score'):
                if (solutionType == 'supervised'):
                    result = get_algorithm_class_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                else:
                    status = 201
                    result = "algorithm_class_score isn't applicated"

            elif (detailType == 'feature_relevance_score'):
                if (solutionType == 'supervised'):
                    result = get_feature_relevance_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                else:
                    status = 201
                    result = "feature_relevance_score isn't applicated"
            elif (detailType == 'permutation_feature_importance_score'):
                if (solutionType == 'supervised'):
                    result = "permutation_feature_importance_score_supervied isn't applicated"
                else:
                    result = get_permutation_feature_importance_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data)

        elif (type == 'fairness'):
            if (detailType == 'disparate_impact_score'):
                if (solutionType == 'supervised'):
                    result = get_disparate_impact_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    result = get_disparate_impact_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'class_balance_score'):
                if (solutionType == 'supervised'):
                    result = get_class_balance_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    status = 201
                    result = "get_class_balance_score_unsupervised unimplemented"

            elif (detailType == 'overfitting_score'):
                if (solutionType == 'supervised'):
                    result = get_overfitting_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    result = get_overfitting_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'underfitting_score'):
                if (solutionType == 'supervised'):
                    result = get_underfitting_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    result = get_underfitting_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'statistical_parity_difference_score'):
                if (solutionType == 'supervised'):
                    result = get_statistical_parity_difference_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    result = get_statistical_parity_difference_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'equal_opportunity_difference_score'):
                if (solutionType == 'supervised'):
                    result = get_equal_opportunity_difference_score(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    status = 201
                    result = "equal_opportunity_difference_score_unsupervied isn't implemented"

            elif (detailType == 'average_odds_difference_score'):
                if (solutionType == 'supervised'):
                    result = get_average_odds_difference_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                else:
                    status = 201
                    result = "avaeraage_odds_difference_score not implemented"

        elif (type == 'pillar'):
            if (detailType == 'accountability_score'):
                if (solutionType == 'supervised'):
                    result = get_accountability_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                else:
                    result = get_accountability_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'robustnesss_score'):
                if (solutionType == 'supervised'):
                    result = get_robustness_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                else:
                    result = get_robustness_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'explainability_score'):
                if (solutionType == 'supervised'):
                    result = get_explainability_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                else:
                    result = get_explainability_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'fairness_score'):
                if (solutionType == 'supervised'):
                    result = get_fairness_score_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                else:
                    result = get_fairness_score_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

        elif (type == 'trust'):
            if (detailType == 'trustscore'):
                if (solutionType == 'supervised'):
                    result = trusting_AI_scores_supervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data).score
                else:
                    result = trusting_AI_scores_unsupervised(
                        model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data).score
            elif (detailType == 'trusting_AI_scores_supervised'):
                result = trusting_AI_scores_supervised(
                    model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

            elif (detailType == 'trusting_AI_scores_unsupervised'):
                result = trusting_AI_scores_unsupervised(
                    model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

        else:
            result = 'none'
            status = 201

        return Response(result, status)

    else:
        return Response("error: No-Exist", status=409)


class CustomAutoSchema(SwaggerAutoSchema):
    def get_request_serializer(self):
        if self.method.lower() == 'post':
            return None
        return super().get_request_serializer()

    def get_query_parameters(self, *args, **kwargs):
        params = super().get_query_parameters(*args, **kwargs)
        if self.method.lower() == 'post':
            params.append(openapi.Parameter(
                'model_file', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter(
                'training_dataset', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter(
                'test_dataset', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter(
                'factsheet', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter(
                'mappings', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter(
                'solution_type', in_=openapi.IN_FORM, type=openapi.TYPE_STRING, required=True))
            params.append(openapi.Parameter(
                'scenario_name', in_=openapi.IN_FORM, type=openapi.TYPE_STRING, required=True))
            params.append(openapi.Parameter(
                'solution_name', in_=openapi.IN_FORM, type=openapi.TYPE_STRING, required=True))

        return params


class CustomAutoSchemaWithFile(CustomAutoSchema):
    def get_request_body(self, path, method):
        base_body = super().get_request_body(path, method)
        base_body['required'].append('file')
        base_body['properties']['file'] = {
            'type': 'string', 'format': 'binary'}
        return base_body

# 1)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_factsheet_completness_score(request):
    return handle_score_request('account', 'factsheet', request.data, request.user.id)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_missing_data_score(request):
    return handle_score_request('account', 'missingdata', request.data, request.user.id)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_normalization_score(request):
    return handle_score_request('account', 'normalization', request.data, request.user.id)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_regularization_score(request):
    return handle_score_request('account', 'regularization', request.data, request.user.id)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_train_test_split_score(request):
    return handle_score_request('account', 'train_test', request.data, request.user.id)

# 1) Clever Score


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_clever_score(request):
    return handle_score_request('robust', 'clever_score', request.data, request.user.id)


# 2)CliqueMethodScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_clique_method_score(request):
    return handle_score_request('robust', 'clique_method_score', request.data, request.user.id)


# 3)ConfidenceScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_confidence_score(request):
    return handle_score_request('robust', 'confidence_score', request.data, request.user.id)


# 4)CarliWagnerAttackScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_carliwagnerwttack_score(request):
    return handle_score_request('robust', 'carliwagnerwttack_score', request.data, request.user.id)


# 5)DeepFoolAttackScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_deepfoolattack_score(request):
    return handle_score_request('robust', 'deepfoolattack_score', request.data, request.user.id)


# 6)ERFastGradientAttackScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_fast_gradient_attack_score(request):
    return handle_score_request('robust', 'fast_gradient_attack_score', request.data, request.user.id)


# 7)LossSensitivityScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_loss_sensitivity_score(request):
    return handle_score_request('robust', 'loss_sensitivity_score', request.data, request.user.id)


# C)Explainability
# 1)ModelSizeScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_modelsize_score(request):
    return handle_score_request('explain', 'modelsize_score', request.data, request.user.id)

# 2)CorrelatedFeaturesScore


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_correlated_features_score(request):
    return handle_score_request('explain', 'correlated_features_score', request.data, request.user.id)


# 3)AlgorithmClassScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_algorithm_class_score(request):
    return handle_score_request('explain', 'algorithm_class_score', request.data, request.user.id)


# 4)FeatureRelevanceScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_feature_relevance_score(request):
    return handle_score_request('explain', 'feature_relevance_score', request.data, request.user.id)


# 5)PermutationFeatures

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_permutation_feature_importance_score(request):
    return handle_score_request('explain', 'permutation_feature_importance_score', request.data, request.user.id)


# D)Fairness
# 1)DisparateImpactScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_disparate_impact_score(request):
    return handle_score_request('fairness', 'disparate_impact_score', request.data, request.user.id)


# 2)ClassBalanceScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_class_balance_score(request):
    return handle_score_request('fairness', 'disparate_impact_score', request.data, request.user.id)


# 3)OverfittingScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_overfitting_score(request):
    return handle_score_request('fairness', 'overfitting_score', request.data, request.user.id)


# 4)UnderfittingScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_underfitting_score(request):
    return handle_score_request('fairness', 'underfitting_score', request.data, request.user.id)


# 5)StatisticalParityDifferenceScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_statistical_parity_difference_score(request):
    return handle_score_request('fairness', 'statistical_parity_difference_score', request.data, request.user.id)


# 6)EqualOpportunityDifferenceScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_equal_opportunity_difference_score(request):
    return handle_score_request('fairness', 'equal_opportunity_difference_score', request.data, request.user.id)


# 7)AverageOddsDifferenceScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_average_odds_difference_score(request):
    return handle_score_request('fairness', 'average_odds_difference_score', request.data, request.user.id)


# E)PillarScores
# 1)AccountabilityScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_accountability_score(request):
    return handle_score_request('pillar', 'accountability_score', request.data, request.user.id)


# 2)RobustnessScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_robustness_score(request):
    return handle_score_request('pillar', 'robustnesss_score', request.data, request.user.id)


# 3)ExplainabilityScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_explainability_score(request):
    return handle_score_request('pillar', 'explainability_score', request.data, request.user.id)


# 4)FairnessScore
@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_fairness_score(request):
    return handle_score_request('pillar', 'fairness_score', request.data, request.user.id)


# 5)TrustScore

@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trust_score(request):
    return handle_score_request('trust', 'trustscore', request.data, request.user.id)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trusting_AI_scores_supervised(request):
    return handle_score_request('trust', 'trusting_AI_scores_supervised', request.data, request.user.id)


@ swagger_auto_schema(method='post', request_body=SolutionSerializer, auto_schema=CustomAutoSchemaWithFile)
@ parser_classes([MultiPartParser, FormParser])
@ api_view(['POST'])
@ authentication_classes([CustomUserAuthentication])
def get_trusting_AI_scores_unsupervised(request):
    return handle_score_request('trust', 'trusting_AI_scores_unsupervised', request.data, request.user.id)
