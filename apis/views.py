from .apiback.public import save_files_return_paths
from rest_framework import permissions
from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import api_view, permission_classes
from django.http import Http404
from algorithms.unsupervised.Functions.Accountability.NormalizationScore import normalization_score as get_normalization_score_unsupervised
from algorithms.supervised.Functions.Robustness.CleverScore_supervised import get_clever_score_supervised
from algorithms.unsupervised.Functions.Robustness.CleverScore import clever_score as get_clever_score_unsupervised
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
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import pandas as pd
import json
from .models import CustomUser, Scenario, ScenarioSolution
from .serilizers import SolutionSerializer, ScenarioSerializer
# import stripe
from rest_framework import status
from rest_framework.decorators import api_view
from pathlib import Path
import os
import collections

from .authentication import CustomUserAuthentication
from .apiback.public import analyse_explainability, analyse_fairness, analyse_methodology, analyse_robustness

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
print("Base dir is:", BASE_DIR)

path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
path_mapping_fairness = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/robustness/default.json')
print("Robustness reslt:", analyse_robustness(path_module,
      path_traindata, path_testdata, path_mapping_fairness, path_factsheet))
print("############################################################################")


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

path_testdata = os.path.join(BASE_DIR, 'apis/TestValues/test.csv')
path_traindata = os.path.join(BASE_DIR, 'apis/TestValues/train.csv')
path_module = os.path.join(BASE_DIR, 'apis/TestValues/model.pkl')
path_factsheet = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')
path_mapping_fairness = os.path.join(
    BASE_DIR, 'apis/MappingsWeightsMetrics/Mappings/explainability/default.json')
print("Explainability reslt:", analyse_explainability(path_module,
      path_traindata, path_testdata, path_mapping_fairness, path_factsheet))
print("############################################################################")

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
print("Final Score unsupervised:", get_final_score_unsupervised(path_module, path_traindata,
      path_testdata, outliers_data, config_weights, mappings_config, path_factsheet, solution_set_path))


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
                        result = get_equal_opportunity_difference_score_supervised(
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
            user = request.user
            scenario_name = request.data.get('scenario_name', '')
            if not scenario_name:
                # Generate new scenario name
                base_name = f'Scenario_{slugify(user.username)}_'
                count = Scenario.objects.filter(
                    name__startswith=base_name).count()
                scenario_name = f'{base_name}{count + 1}'

            # Check if scenario exists
            scenario = Scenario.objects.filter(
                scenario_name=scenario_name, user_id=user.id).first()
            if not scenario:
                # Create new scenario
                scenario = Scenario.objects.create(
                    scenario_name=scenario_name, user_id=user.id)

            serializer.save(user=user, scenario=scenario)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk):
        solution = get_object_or_404(ScenarioSolution, pk=pk)
        serializer = SolutionSerializer(solution, data=request.data)
        if serializer.is_valid():
            user = request.user
            scenario_name = serializer.validated_data['scenario_name']
            scenario = get_object_or_404(
                Scenario, scenario_name=scenario_name, user=user)
            serializer.save(user=user, scenario=scenario)
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user = request.user
        solution_name = request.data.get('solution_name')
        if not solution_name:
            return Response({'error': 'solution_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        solution = ScenarioSolution.objects.filter(
            user=user, solution_name=solution_name).first()
        if not solution:
            return Response({'error': 'Solution not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        solution.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


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
        scenario = Scenario.objects.filter(
            user=user, scenario_name=scenario_name).first()
        if not scenario:
            return Response({'error': 'Scenario not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        serializer = ScenarioSerializer(
            scenario, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        user = request.user
        scenario_name = request.data.get('scenario_name')
        if not scenario_name:
            return Response({'error': 'scenario_name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        scenario = Scenario.objects.filter(
            user=user, scenario_name=scenario_name).first()
        if not scenario:
            return Response({'error': 'Scenario not found for the given user.'}, status=status.HTTP_404_NOT_FOUND)
        scenario.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
