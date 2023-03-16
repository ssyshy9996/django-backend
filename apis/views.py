from passlib.handlers.django import django_pbkdf2_sha256
from rest_framework import permissions
from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import api_view, permission_classes
from django.http import Http404, HttpResponse
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
from .serilizers import UserSerializer, SolutionSerializer,ScenarioSerializer
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


class auth(APIView):
    def get(self, request):  # when login
        email = request.query_params['email']
        password = request.query_params['password']
        user = CustomUser.objects.get(email=email)
        password_verify = django_pbkdf2_sha256.verify(password, user.password)

        if (password_verify):
            return Response({
                'email': email,
                'password': password,
            }, status=200)
        else:
            return Response('Login Failed', status=400)

    def post(self, request):  # when register
        print('register:', request)
        email = request.data['email']
        password = request.data['password']

        if email is None or password is None:
            return Response('register Error', status=400)

        password = make_password(password)
        print('ne pass:', password)
        newUser = CustomUser.objects.create(email=email, password=password)
        newUser.save()

        return Response('register', status=200)


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

from algorithms.TrustScore.TrustScore import trusting_AI_scores_supervised

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

        scenarios = Scenario.objects.filter(user_id=userexist.id).values()
        uploaddic['scenarioList'] = scenarios
        uploaddic['solutionList'] = scenarioobj


        def unsupervised_FinalScore(model_path, training_dataset_path, test_dataset_path, outliers_dataset_path, facsheet_path, metrics_mappings_path,weights_metrics_path=None, weights_pillars_path=None):
            from algorithms.TrustScore.TrustScore import trusting_AI_scores_unsupervised as finalScore_unsupervised
            import pandas as pd

            finalScore_unsup = dict_trusting_score_unsup=finalScore_unsupervised(model=model_path, training_dataset=training_dataset_path,
            test_dataset=test_dataset_path, outliers_data=outliers_dataset_path,factsheet=facsheet_path, mappings= metrics_mappings_path)
            
            dict_trusting_score_unsup=dict_trusting_score_unsup[0]
            print("DICT TRUSTING SCORE Robustness: ", dict_trusting_score_unsup["robustness"])
            foo1,foo2,foo3,foo4=dict_trusting_score_unsup["accountability"],dict_trusting_score_unsup["explainability"],dict_trusting_score_unsup["fairness"],dict_trusting_score_unsup["robustness"]
            
            factsheetcompletness_score, missing_data_score, normalization_score, regularization_score, train_test_split_score=foo1["factsheet_completeness"],foo1["missing_data"],foo1["normalization"],foo1["regularization"], foo1["train_test_split"]
            
            print('foo2:', foo2)
            correlated_features_score,model_size_score, permutation_feature_importance_score=foo2["correlated_features"],foo2["model_size"],foo2["permutation_feature_importance"]
            
            disparate_impact_score, overfitting_score, statistical_parity_difference_score, underfitting_score=foo3["disparate_impact"],foo3["overfitting"],foo3["statistical_parity_difference"],foo3["underfitting"]
            
            clever_score=foo4["clever_score"]
            #wait need to find weights metrics for unsup

            if (weights_metrics_path is None):
                weights_metrics_unsup={
                "fairness": {
                    "underfitting": 0.35,
                    "overfitting": 0.15,
                    "statistical_parity_difference": 0.15,
                    "disparate_impact": 0.1
                },
                "explainability": {
                    "correlated_features": 0.15,
                    "model_size": 0.15,
                    "permutation_feature_importance": 0.15
                },
                "robustness": {
                    "clever_score": 0.2
                },
                "methodology": {
                    "normalization": 0.2,
                    "missing_data": 0.2,
                    "regularization": 0.2,
                    "train_test_split": 0.2,
                    "factsheet_completeness": 0.2
                },
                "pillars": {
                    "fairness": 0.25,
                    "explainability": 0.25,
                    "robustness": 0.25,
                    "methodology": 0.25
                }
            }
            else:
                weights_metrics_unsup=pd.read_json(weights_metrics_unsup)

                        
            if (weights_pillars_path is None):
                weights_pillars_unsup={"pillars": {
                    "fairness": 0.25,
                    "explainability": 0.25,
                    "robustness": 0.25,
                    "methodology": 0.25
                }}
            else:
                weights_pillars_unsup=pd.read_json(weights_pillars_unsup)


            #default weights pillar scores calculations
            print("DICT WEIGHTS METRICS: ",weights_metrics_unsup)
            try:
                foo5=weights_metrics_unsup["accountability"]
            except:
                foo5=weights_metrics_unsup["methodology"]

            foo6,foo7,foo8=weights_metrics_unsup["explainability"],weights_metrics_unsup["fairness"],weights_metrics_unsup["robustness"]
            

            print("Foo8: ",foo8)

           

            accountability_score_unsupervised=foo5["factsheet_completeness"]*factsheetcompletness_score+foo5["missing_data"]*missing_data_score+foo5["normalization"]*normalization_score+foo5["regularization"]*regularization_score+foo5["train_test_split"]*train_test_split_score
            explainability_score_unsupervised=foo6["correlated_features"]*correlated_features_score+foo6["model_size"]*model_size_score+foo6["permutation_feature_importance"]*permutation_feature_importance_score
            fairness_score_unsupervised=foo7["underfitting"]*underfitting_score+foo7["overfitting"]*overfitting_score+foo7["statistical_parity_difference"]*statistical_parity_difference_score+ foo7["disparate_impact"]*disparate_impact_score
            import numpy as np
            try:
                robustnesss_score_unsupervised=clever_score
            except:
                try:
                    if(clever_score is None):
                        clever_score=1
                except:
                    pass
                
                robustnesss_score_unsupervised=clever_score
            
            weights_pillars_unsup=weights_pillars_unsup["pillars"] 
            try:
                weight_accountability=weights_pillars_unsup["accountability"]
            except:
                weight_accountability=weights_pillars_unsup["methodology"]

            trust_score_unsupervised=weight_accountability*accountability_score_unsupervised+weights_pillars_unsup["explainability"]*explainability_score_unsupervised+weights_pillars_unsup["fairness"]*fairness_score_unsupervised+weights_pillars_unsup["robustness"]*robustnesss_score_unsupervised
            
            dict_accountabiltiy_metric_scores={"Factsheecompletnessscore":factsheetcompletness_score,"Missingdatascore":missing_data_score, "Normalizationscore":normalization_score, "Regularizationscore":regularization_score,"Traintestsplitscore":train_test_split_score}
            dict_explainabiltiy_metric_scores={"Correlatedfeaturesscore":correlated_features_score,"Modelsizescore":model_size_score,"Permutationfeatureimportancescore":permutation_feature_importance_score}
            dict_fairness_metric_scores={"Underfittingscore":underfitting_score, "Overfittingscore":overfitting_score,"Statisticalparitydifferencescore":statistical_parity_difference_score,"Disparateimpactscore":disparate_impact_score}            
            dict_robustness_metric_scores={"Cleverscore":clever_score}
            
            dict_metric_scores={"Metricscores":{"Accountabilityscore":dict_accountabiltiy_metric_scores,"Explainabilityscore":dict_explainabiltiy_metric_scores,"Fairnessscore":dict_fairness_metric_scores,"Robustnessscore":dict_robustness_metric_scores}}
            
            dict_pillars_scores={"Accountabilityscore":accountability_score_unsupervised,"Explainabilityscore":explainability_score_unsupervised,"Fairnessscore":fairness_score_unsupervised,"Robustnessscore":robustnesss_score_unsupervised}

            dict_result_unsupervised={"Metricscores":dict_metric_scores,"Pillarscores":dict_pillars_scores,"Trustscore":trust_score_unsupervised}
            print("DICT RESULT UNSUP: ",dict_result_unsupervised)
            return dict_result_unsupervised


        def finalScore_supervised(model_path, training_dataset_path, test_dataset_path, facsheet_path, metrics_mappings_path,weights_metrics_path=None, weights_pillars_path=None):
            from algorithms.TrustScore.TrustScore import trusting_AI_scores_supervised
            import pandas as pd

            finalScore = dict_trusting_score=trusting_AI_scores_supervised(model=model_path, training_dataset=training_dataset_path,
            test_dataset=test_dataset_path, factsheet=facsheet_path, mappings= metrics_mappings_path)
            
            dict_trusting_score=dict_trusting_score[0]
            print("DICT TRUSTING SCORE Robustness: ", dict_trusting_score["robustness"])
            foo1,foo2,foo3,foo4=dict_trusting_score["accountability"],dict_trusting_score["explainability"],dict_trusting_score["fairness"],dict_trusting_score["robustness"]
            factsheetcompletness_score, missing_data_score, normalization_score, regularization_score, train_test_split_score=foo1["factsheet_completeness"],foo1["missing_data"],foo1["normalization"],foo1["regularization"], foo1["train_test_split"]
            algorithm_class_score, correlated_features_score, feature_relevance_score, model_size_score=foo2["algorithm_class"],foo2["correlated_features"],foo2["feature_relevance"],foo2["model_size"]
            average_odds_difference_score, class_balance_score, disparate_impact_score, equal_opportunity_score, overfitting_score, statistical_parity_difference_score, underfitting_score=foo3["average_odds_difference"],foo3["class_balance"],foo3["disparate_impact"],foo3["equal_opportunity_difference"],foo3["overfitting"],foo3["statistical_parity_difference"],foo3["underfitting"]
            clever_score, clique_method_score, confidence_score, er_carlini_wagner_score, er_deep_fool_attack_score, er_fast_gradient_attack_score, loss_sensitivity_score=foo4["clever_score"],foo4["clique_method"],foo4["confidence_score"],foo4["er_carlini_wagner_attack"],foo4["er_deepfool_attack"],foo4["er_fast_gradient_attack"],foo4["loss_sensitivity"]
            
            

            if (weights_metrics_path is None):
                weights_metrics={
                "fairness": {
                    "underfitting": 0.35,
                    "overfitting": 0.15,
                    "statistical_parity_difference": 0.15,
                    "equal_opportunity_difference": 0.2,
                    "average_odds_difference": 0.1,
                    "disparate_impact": 0.1,
                    "class_balance": 0.1
                },
                "explainability": {
                    "algorithm_class": 0.55,
                    "correlated_features": 0.15,
                    "model_size": 0.15,
                    "feature_relevance": 0.15
                },
                "robustness": {
                    "confidence_score": 0.2,
                    "clique_method": 0.2,
                    "loss_sensitivity": 0.2,
                    "clever_score": 0.2,
                    "er_fast_gradient_attack": 0.2,
                    "er_carlini_wagner_attack": 0.2,
                    "er_deepfool_attack": 0.2
                },
                "methodology": {
                    "normalization": 0.2,
                    "missing_data": 0.2,
                    "regularization": 0.2,
                    "train_test_split": 0.2,
                    "factsheet_completeness": 0.2
                },
                "pillars": {
                    "fairness": 0.25,
                    "explainability": 0.25,
                    "robustness": 0.25,
                    "methodology": 0.25
                }
            }
            else:
                weights_metrics=pd.read_json(weights_metrics)

                        
            if (weights_pillars_path is None):
                weights_pillars={"pillars": {
                    "fairness": 0.25,
                    "explainability": 0.25,
                    "robustness": 0.25,
                    "methodology": 0.25
                }}
            else:
                weights_pillars=pd.read_json(weights_pillars_path)


            #default weights pillar scores calculations
            print("DICT WEIGHTS METRICS: ",weights_metrics)
            try:
                foo5=weights_metrics["accountability"]
            except:
                foo5=weights_metrics["methodology"]

            foo6,foo7,foo8=weights_metrics["explainability"],weights_metrics["fairness"],weights_metrics["robustness"]
            

            print("Foo8: ",foo8)
            accountability_score_supervised=foo5["factsheet_completeness"]*factsheetcompletness_score+foo5["missing_data"]*missing_data_score+foo5["normalization"]*normalization_score+foo5["regularization"]*regularization_score+foo5["train_test_split"]*train_test_split_score
            explainability_score_supervised=foo6["algorithm_class"]*algorithm_class_score+foo6["correlated_features"]*correlated_features_score+foo6["model_size"]*model_size_score+foo6["feature_relevance"]*feature_relevance_score
            fairness_score_supervised=foo7["underfitting"]*underfitting_score+foo7["overfitting"]*overfitting_score+foo7["statistical_parity_difference"]*statistical_parity_difference_score+ foo7["equal_opportunity_difference"]*equal_opportunity_score+foo7["average_odds_difference"]*average_odds_difference_score+foo7["disparate_impact"]*disparate_impact_score+foo7["class_balance"]*class_balance_score
            import numpy as np
            try:
                robustnesss_score_supervised=foo8["clever_score"]*clever_score+foo8["clique_method"]*clique_method_score+foo8["confidence_score"]*confidence_score+foo8["er_carlini_wagner_attack"]*er_carlini_wagner_score+foo8["er_deepfool_attack"]*er_deep_fool_attack_score+foo8["er_fast_gradient_attack"]*er_fast_gradient_attack_score+foo8["loss_sensitivity"]*loss_sensitivity_score

            except:
                try:
                    if(clique_method_score is None):
                        clique_method_score=1
                except:
                    pass
                if(er_carlini_wagner_score is None):
                    er_carlini_wagner_score=1
                if(er_deep_fool_attack_score is None):
                    er_deep_fool_attack_score=1
                if(er_fast_gradient_attack_score is None):
                    er_fast_gradient_attack_score = 1
                if(loss_sensitivity_score is None):
                    loss_sensitivity_score=1
                robustnesss_score_supervised=foo8["clever_score"]*clever_score+foo8["clique_method"]*clique_method_score+foo8["er_carlini_wagner_attack"]*er_carlini_wagner_score+foo8["er_deepfool_attack"]*er_deep_fool_attack_score+foo8["er_fast_gradient_attack"]*er_fast_gradient_attack_score+foo8["loss_sensitivity"]*loss_sensitivity_score
            
            weights_pillars=weights_pillars["pillars"] 
            try:
                weight_accountability=weights_pillars["accountability"]
            except:
                weight_accountability=weights_pillars["methodology"]

            trust_score_supervised=weight_accountability*accountability_score_supervised+weights_pillars["explainability"]*explainability_score_supervised+weights_pillars["fairness"]*fairness_score_supervised+weights_pillars["robustness"]*robustnesss_score_supervised
            
            dict_accountabiltiy_metric_scores={"Factsheecompletnessscore":factsheetcompletness_score,"Missingdatascore":missing_data_score, "Normalizationscore":normalization_score, "Regularizationscore":regularization_score,"Traintestsplitscore":train_test_split_score}
            dict_explainabiltiy_metric_scores={"Algorithmclassscore":algorithm_class_score,"Correlatedfeaturesscore":correlated_features_score,"Modelsizescore":model_size_score,"Featurerevelancescore":feature_relevance_score}
            dict_fairness_metric_scores={"Underfittingscore":underfitting_score, "Overfittingscore":overfitting_score,"Statisticalparitydifferencescore":statistical_parity_difference_score,"Equalopportunityscore":equal_opportunity_score,"Averageoddsdifferencescore":average_odds_difference_score,"Disparateimpactscore":disparate_impact_score,"Classbalancescore":class_balance_score}            
            dict_robustness_metric_scores={"Cleverscore":clever_score,"Cliquemethodscore":clique_method_score,"Confidencescore":confidence_score,"Ercarliniwagnerscore":er_carlini_wagner_score,"Erdeepfoolattackscore":er_deep_fool_attack_score,"Erfastgradientattack":er_fast_gradient_attack_score,"Losssensitivityscore":loss_sensitivity_score}
            
            dict_metric_scores={"Metricscores":{"Accountabilityscore":dict_accountabiltiy_metric_scores,"Explainabilityscore":dict_explainabiltiy_metric_scores,"Fairnessscore":dict_fairness_metric_scores,"Robustnessscore":dict_robustness_metric_scores}}
            
            dict_pillars_scores={"Accountabilityscore":accountability_score_supervised,"Explainabilityscore":explainability_score_supervised,"Fairnessscore":fairness_score_supervised,"Robustnessscore":robustnesss_score_supervised}

            dict_result_supervised={"Metricscores":dict_metric_scores,"Pillarscores":dict_pillars_scores,"Trustscore":trust_score_supervised}
            print("DICT RESULT SUP: ",dict_result_supervised)
            return dict_result_supervised

        if scenarioobj:
            for i in scenarioobj:
                path_testdata = i["test_file"]
                path_module = i["model_file"]
                path_traindata = i["training_file"]
                path_factsheet = i["factsheet_file"]
                path_outliersdata = i['outlier_data_file']
                soulutionType = i['solution_type']
                try:
                    mappings_config = save_files_return_paths(i['metrics_mapping_file'])[0]
                except:
                    mappings_config = os.path.join(BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/default.json')

            path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata = save_files_return_paths(path_module, path_traindata, path_testdata, path_factsheet, path_outliersdata)
            if(soulutionType == 'supervised'):
                resultSuper = finalScore_supervised(path_module, path_traindata, path_testdata, path_factsheet, mappings_config)

                uploaddic['fairness_score'] = resultSuper['Pillarscores']['Fairnessscore']
                try:
                    uploaddic['methodology_score'] = resultSuper['Pillarscores']['Accountabilityscore']
                except:
                    uploaddic['accountability_score'] = resultSuper['Pillarscores']['Accountabilityscore']

                uploaddic['trust_score'] = resultSuper['Trustscore']['Trustscore']


                uploaddic['explainability_score'] = resultSuper['Pillarscores']['Explainabilityscore']
                uploaddic['robustness_score'] = resultSuper['Pillarscores']['Robustnessscore']
                uploaddic['underfitting'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
                uploaddic['overfitting'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
                uploaddic['statistical_parity_difference'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
                uploaddic['equal_opportunity_difference'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Equalopportunityscore']
                uploaddic['average_odds_difference'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Averageoddsdifferencescore']
                uploaddic['disparate_impact'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
                uploaddic['class_balance'] = resultSuper['Metricscores']['Metricscores']['Fairnessscore']['Classbalancescore']
                uploaddic['algorithm_class'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Algorithmclassscore']
                uploaddic['correlated_features'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
                uploaddic['model_size'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
                uploaddic['feature_relevance'] = resultSuper['Metricscores']['Metricscores']['Explainabilityscore']['Featurerevelancescore']
                uploaddic['confidence_score'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Confidencescore']
                uploaddic['clique_method'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Cliquemethodscore']
                uploaddic['loss_sensitivity'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Losssensitivityscore']
                uploaddic['clever_score'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Cleverscore']
                uploaddic['er_fast_gradient_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Erfastgradientattack']
                uploaddic['er_carlini_wagner_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Ercarliniwagnerscore']
                uploaddic['er_deepfool_attack'] = resultSuper['Metricscores']['Metricscores']['Robustnessscore']['Erdeepfoolattackscore']
                uploaddic['normalization'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Normalizationscore']
                uploaddic['missing_data'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
                uploaddic['regularization'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Regularizationscore']
                uploaddic['train_test_split'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Traintestsplitscore']
                uploaddic['factsheet_completeness'] = resultSuper['Metricscores']['Metricscores']['Accountabilityscore']['Factsheecompletnessscore']

            elif(soulutionType == 'unsupervised'):
                print('is this called?')
                resultUnsuper = unsupervised_FinalScore(path_module, path_traindata, path_testdata, path_outliersdata, path_factsheet, mappings_config)

                uploaddic['unsupervised_fairness_score'] = resultUnsuper['Pillarscores']['Fairnessscore']#here
                try:
                    uploaddic['unsupervised_methodology_score'] = resultUnsuper['Pillarscores']['Accountabilityscore']
                except:
                    uploaddic['accountability_score'] = resultUnsuper['Pillarscores']['Accountabilityscore']
                
                print('result:', resultUnsuper)
                
                uploaddic['unsupervised_trust_score'] = resultUnsuper['Trustscore']

                uploaddic['unsupervised_explainability_score'] = resultUnsuper['Pillarscores']['Explainabilityscore']
                uploaddic['unsupervised_robustness_score'] = resultUnsuper['Pillarscores']['Robustnessscore']

                uploaddic['unsupervised_underfitting'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Underfittingscore']
                uploaddic['unsupervised_overfitting'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Overfittingscore']
                uploaddic['unsupervised_statistical_parity_difference'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Statisticalparitydifferencescore']
                uploaddic['unsupervised_disparate_impact'] = resultUnsuper['Metricscores']['Metricscores']['Fairnessscore']['Disparateimpactscore']
                uploaddic['unsupervised_correlated_features'] = resultUnsuper['Metricscores']['Metricscores']['Explainabilityscore']['Correlatedfeaturesscore']
                uploaddic['unsupervised_permutation_importance'] = resultUnsuper['Metricscores']['Metricscores']['Explainabilityscore']['Permutationfeatureimportancescore']
                uploaddic['unsupervised_model_size'] = resultUnsuper['Metricscores']['Metricscores']['Explainabilityscore']['Modelsizescore']
                uploaddic['unsupervised_clever_score'] = resultUnsuper['Metricscores']['Metricscores']['Robustnessscore']['Cleverscore']
                uploaddic['unsupervised_normalization'] = resultUnsuper['Metricscores']['Metricscores']['Accountabilityscore']['Normalizationscore']
                uploaddic['unsupervised_missing_data'] = resultUnsuper['Metricscores']['Metricscores']['Accountabilityscore']['Missingdatascore']
                uploaddic['unsupervised_regularization'] = resultUnsuper['Metricscores']['Metricscores']['Accountabilityscore']['Regularizationscore']
                uploaddic['unsupervised_train_test_split'] = resultUnsuper['Metricscores']['Metricscores']['Accountabilityscore']['Traintestsplitscore']
                uploaddic['unsupervised_factsheet_completeness'] = resultUnsuper['Metricscores']['Metricscores']['Accountabilityscore']['Factsheecompletnessscore']

                

            FACTSHEET_NAME = "Newfact"
            return Response(uploaddic)
        else:
            return Response('No Solution')

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
        print('request:', request.data)
        solutionDetail.solution_name = request.data['NameSolution']
        solutionDetail.description = request.data['DescriptionSolution']
        if (request.data['TrainingFile'] != 'undefined'):
            print('asdfasdfasdf')
        if (request.data['TrainingFile'] != 'undefined'):
            solutionDetail.training_file = request.data['TrainingFile']
        if (request.data['TestFile'] != 'undefined'):
            solutionDetail.test_file = request.data['TestFile']
        if (request.data['FactsheetFile'] != 'undefined'):
            solutionDetail.factsheet_file = request.data['FactsheetFile']
        if (request.data['ModelFile'] != 'undefined'):
            solutionDetail.model_file = request.data['ModelFile']
        if (len(request.data['Targetcolumn']) <= 0):
            solutionDetail.target_column = request.data['Targetcolumn']
        if (request.data['Outlierdatafile'] != 'undefined'):
            solutionDetail.outlier_data_file = request.data['Outlierdatafile']
        if (len(request.data['ProtectedFeature']) <= 0):
            solutionDetail.protected_features = request.data['ProtectedFeature']
        if (len(request.data['Protectedvalues']) <= 0):
            solutionDetail.protected_values = request.data['Protectedvalues']
        if (len(request.data['Favourableoutcome']) <= 0):
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

            mapFile = ''
            if request.data['MapFile'] is None or request.data['MapFile'] == 'undefined':
                mapFile = 'files/mapping_metrics_default.json'
            else:
                mapFile = request.data['MapFile']

            try:
                print('email:', request.data)
                userexist = CustomUser.objects.get(
                    email=request.data['emailid'])
                scenario = Scenario.objects.get(
                    scenario_name=request.data['SelectScenario'])
                print("Solution type:", scenario.id)
                fileupload = ScenarioSolution.objects.create(
                    user_id=userexist.id,
                    scenario_id=scenario.id,
                    solution_name=request.data['NameSolution'],
                    description=request.data['DescriptionSolution'],
                    training_file=request.data['TrainingFile'],
                    metrics_mappings_file=mapFile,
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

            path_module, path_testdata, path_traindata, path_factsheet = save_files_return_paths(path_module, path_testdata, path_traindata, path_factsheet)
            print("Performance_Metrics reslt:", get_performance_metrics(
                path_module, path_testdata, 'Target', path_traindata, path_factsheet))

            def get_factsheet_completeness_score(factsheet):
                propdic = {}
                import collections
                info = collections.namedtuple('info', 'description value')
                result = collections.namedtuple('result', 'score properties')

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

                factsheet = save_files_return_paths(factsheet)[0]
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
                model, test_data = save_files_return_paths(model, test_data)
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
                model, test_data = save_files_return_paths(model, test_data)
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
            print("Performance_Metrics reslt:", get_performance_metrics(
                path_module, path_testdata, 'Target'))

            if scenarioobj:
                for i in scenarioobj:
                    if i['scenario_id'] == scenario.id and i['solution_name'] == request.data['SelectSolution2']:
                        path_testdata = i['test_file']
                        path_module = i['model_file']
                        # print("ScenarioSolution data:", i.SolutionName)
            print("Performance_Metrics reslt:", get_performance_metrics2(
                path_module, path_testdata, 'Target'))

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
        result = ''

        try:
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
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'confidence_score'):
                    if (solutionType == 'supervised'):
                        result = get_confidence_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'carliwagnerwttack_score'):
                    if (solutionType == 'supervised'):
                        result = get_carliwagnerwttack_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'loss_sensitivity_score'):
                    if (solutionType == 'supervised'):
                        result = get_loss_sensitivity_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'deepfoolattack_score'):
                    if (solutionType == 'supervised'):
                        result = get_deepfoolattack_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
                
                elif (detailType == 'fast_gradient_attack_score'):
                    if (solutionType == 'supervised'):
                        result = get_fast_gradient_attack_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)
                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
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
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

                elif (detailType == 'feature_relevance_score'):
                    if (solutionType == 'supervised'):
                        result = get_feature_relevance_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
                elif (detailType == 'permutation_feature_importance_score'):
                    if (solutionType == 'supervised'):
                        status = 404
                        result = "The metric function isn't applicable for supervised ML/DL solutions"
                    else:
                        if (outliers_data.find('.') < 0):
                            status = 404
                            result = "The outlier data file is missing"
                        else:
                            result = get_permutation_feature_importance_score_unsupervised(
                                model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, outliers_data)

            elif (type == 'fairness'):
                if (detailType == 'disparate_impact_score'):
                    if (solutionType == 'supervised'):
                        result = get_disparate_impact_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        print('before get:')
                        result = get_disparate_impact_score_unsupervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                        print('get result:', result)

                elif (detailType == 'class_balance_score'):
                    if (solutionType == 'supervised'):
                        result = get_class_balance_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

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
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"
                elif (detailType == 'average_odds_difference_score'):
                    if (solutionType == 'supervised'):
                        result = get_average_odds_difference_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)

                    else:
                        status = 404
                        result = "The metric function isn't applicable for unsupervised ML/DL solutions"

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
                    print("outdata:", )
                    if (solutionType == 'supervised'):
                        print('super called')
                        result = get_explainability_score_supervised(
                            model_path, training_dataset_path, test_dataset_path, factsheet_path, mappings_path, target_column, outliers_data)
                    else:
                        if (outliers_data.find('.') < 0):
                            status = 404
                            result = "The outlier data file is missing"
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
                print('test:', type, detailType)
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
        except Exception as e:
            print('error:', e)
            return Response("Error while reading corrupted model file", status=409)

    else:
        return Response("Please login / User doesn't exist", status=409)


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
    print('data:', request.data, request.user.id)
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


class SolutionList(APIView):
    """
    List all solutions or create a new solution.
    """

    def get(self, request):
        user = request.user
        print("GET SOLUTIONLIST USER: ", user)
        if not user.is_authenticated:
            return Response({'error': 'Authentication failed'})

        solutions = ScenarioSolution.objects.all()
        serializer = SolutionSerializer(solutions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = SolutionSerializer(data=request.data)
        if serializer.is_valid():
            print("REQUEST DATA: ",request.data)
            print("USER ID: ",request.user.id)
            user = request.user
            scenario_name = request.data.get('scenario_name', '')
            if not scenario_name:
                # Generate new scenario name
                base_name = f'Scenario_{slugify(user.username)}_'
                count = Scenario.objects.filter(name__startswith=base_name).count()
                scenario_name = f'{base_name}{count + 1}'

            # Check if scenario exists
            scenario = Scenario.objects.filter(scenario_name=scenario_name, user_id=user.id).first()
            if not scenario:
                # Create new scenario
                scenario = Scenario.objects.create(scenario_name=scenario_name, user_id=user.id)

            serializer.save(user=user, scenario=scenario)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk):
        solution = get_object_or_404(ScenarioSolution, pk=pk)
        serializer = SolutionSerializer(solution, data=request.data)
        if serializer.is_valid():
            user = request.user
            scenario_name = serializer.validated_data['scenario_name']
            scenario = get_object_or_404(Scenario, scenario_name=scenario_name, user=user)
            serializer.save(user=user, scenario=scenario)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user = request.user
        solution_name = request.data.get('solution_name')
        if not solution_name:
            return Response({'error': 'solution_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        solution = ScenarioSolution.objects.filter(user=user, solution_name=solution_name).first()
        if not solution:
            return Response({'error': 'Solution not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        solution.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


from django.http import Http404, HttpResponse
class ScenarioList(APIView):
    """
    List all scenarios or create a new scenario.
    """
    def get_object(self, scenario_name):
        try:
            return Scenario.objects.get(user=self.request.user, scenario_name=scenario_name)
        except Scenario.DoesNotExist:
            raise Http404

    def get(self, request):
        scenarios = Scenario.objects.all()
        serializer = ScenarioSerializer(scenarios, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = ScenarioSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            serializer.save(user=user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request):
        user = request.user
        scenario_name = request.data.get('scenario_name')
        if not scenario_name:            
            return Response({'error': 'scenario_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        scenario = Scenario.objects.filter(user=user, scenario_name=scenario_name).first()
        if not scenario:
            return Response({'error': 'Scenario not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        serializer = ScenarioSerializer(scenario, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user = request.user
        scenario_name = request.data.get('scenario_name')
        if not scenario_name:
            return Response({'error': 'scenario_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        scenario = Scenario.objects.filter(user=user, scenario_name=scenario_name).first()
        if not scenario:
            return Response({'error': 'Scenario not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        scenario.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes


class IsAdminOrReadOnly(permissions.BasePermission):

    def has_permission(self, request, view):
        # allow GET requests for all users
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # allow admins to perform unsafe operations
        return request.user.is_staff
    

class IsOwnerOrAdminOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS or request.user.is_superuser:
            return True
        return obj.user == request.user

class AllUserOrAdminOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        print("REQUEST METHOD: ",request.method)
        print("USER REQUEST: ", request.data)
        if request.method in permissions.SAFE_METHODS or request.user.is_superuser:
            return True
        return obj == request.user


@api_view(['GET', 'POST'])
@authentication_classes([CustomUserAuthentication])
@permission_classes([AllUserOrAdminOrReadOnly, permissions.IsAuthenticated])
def all_users(request):
    if request.method == 'GET':
        queryset = CustomUser.objects.all()
        serializer = AllUserSerializer(queryset, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        data = request.data.copy()
        user = request.user
        data['user'] = user.id
        serializer = ScenarioSerializer(data=data)
        if serializer.is_valid():
            serializer.save(user=user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


from rest_framework.viewsets import ModelViewSet
class SolutionViewSet(ModelViewSet):
    serializer_class = ScenarioSolution
    queryset = Scenario.objects.all()
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrAdminOrReadOnly]

class CustomAutoSchema(SwaggerAutoSchema):
    def get_request_serializer(self):
        if self.method.lower() == 'post':
            return None
        return super().get_request_serializer()

    def get_query_parameters(self, args, *kwargs):
        params = super().get_query_parameters(*args, **kwargs)
        if self.method.lower() == 'post':
            params.append(openapi.Parameter('model_file', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter('training_dataset', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter('test_dataset', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter('factsheet', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter('mappings', in_=openapi.IN_FORM, type=openapi.TYPE_FILE, required=True))
            params.append(openapi.Parameter('solution_type', in_=openapi.IN_FORM, type=openapi.TYPE_STRING, required=True))
            params.append(openapi.Parameter('scenario_name', in_=openapi.IN_FORM, type=openapi.TYPE_STRING, required=True))
            params.append(openapi.Parameter('solution_name', in_=openapi.IN_FORM, type=openapi.TYPE_STRING, required=True))


        return params