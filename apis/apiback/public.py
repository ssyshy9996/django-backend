from algorithms.supervised.Functions.Fairness.ClassBalanceScore_supervised import get_class_balance_score_supervised
import sys
import django
from ..models import Score, MetricProperty, PerformanceMetrics

sys.path.extend([r"Backend", r"Backend/apis"])


def save_files_return_paths(*args):
    fs = django.core.files.storage.FileSystemStorage()
    paths = []
    for arg in args:
        if arg is None or arg == 'undefined':
            paths.append(None)
        else:
            paths.append(fs.path(arg))
    return tuple(paths)


def save_scores(solution_id, data):
    try:
        isExist = Score.objects.get(solution_id=solution_id)
        if isExist is not None:
            return False
    except:

        newScoreObject = Score.objects.create(
            solution_id=solution_id,

            fairness=data['fairness_score'],
            overfitting=data['overfitting'],
            underfitting=data['underfitting'],
            statistical=data['statistical_parity_difference'],
            disparate=data['disparate_impact'],
            equal=data['equal_opportunity_difference'],
            average=data['average_odds_difference'],
            class_balance=data['class_balance'],

            explain=data['explainability_score'],
            correlated=data['correlated_features'],
            model_size=data['model_size'],
            permutation=data['permutation_feature_importance'],
            feature=data['feature_relevance'],
            algorithm=data['algorithm_class'],

            robust=data['robustness_score'],
            clever=data['clever_score'],
            confidence=data['confidence_score'],
            clique=data['clique_method'],
            er_fast=data['er_fast_gradient_attack'],
            er_carlini=data['er_carlini_wagner_attack'],
            er_deep=data['er_deepfool_attack'],
            loss_sensitivity=data['loss_sensitivity'],

            account=data['methodology_score'],
            train_test_split=data['train_test_split'],
            missing_data=data['missing_data'],
            normalization=data['normalization'],
            regularization=data['regularization'],
            factsheet=data['factsheet_completeness'],

            trust=data['trust_score']
        )

        newScoreObject.save()
    return True


def get_score(solution_id):
    try:
        isExist = Score.objects.get(solution_id=solution_id)
        uploaddic = {}
        uploaddic['fairness_score'] = isExist.fairness
        uploaddic['overfitting'] = isExist.overfitting
        uploaddic['underfitting'] = isExist.underfitting
        uploaddic['statistical_parity_difference'] = isExist.statistical
        uploaddic['disparate_impact'] = isExist.disparate
        uploaddic['equal_opportunity_difference'] = isExist.equal
        uploaddic['average_odds_difference'] = isExist.average
        uploaddic['class_balance'] = isExist.class_balance
        uploaddic['explainability_score'] = isExist.explain
        uploaddic['correlated_features'] = isExist.correlated
        uploaddic['model_size'] = isExist.model_size
        uploaddic['permutation_feature_importance'] = isExist.permutation
        uploaddic['feature_relevance'] = isExist.feature
        uploaddic['algorithm_class'] = isExist.algorithm
        uploaddic['robustness_score'] = isExist.robust
        uploaddic['clever_score'] = isExist.clever
        uploaddic['confidence_score'] = isExist.confidence
        uploaddic['clique_method'] = isExist.clique
        uploaddic['er_fast_gradient_attack'] = isExist.er_fast
        uploaddic['er_carlini_wagner_attack'] = isExist.er_carlini
        uploaddic['er_deepfool_attack'] = isExist.er_deep
        uploaddic['loss_sensitivity'] = isExist.loss_sensitivity
        uploaddic['methodology_score'] = isExist.account
        uploaddic['train_test_split'] = isExist.train_test_split
        uploaddic['missing_data'] = isExist.missing_data
        uploaddic['normalization'] = isExist.normalization
        uploaddic['regularization'] = isExist.regularization
        uploaddic['factsheet_completeness'] = isExist.factsheet
        uploaddic['trust_score'] = isExist.trust

        print('exactly returend')
        return uploaddic
    except Exception as e:
        print('get score exception:', e)
        raise Exception("score not exist")


def save_property(solution_id, data):
    print('save property called')
    try:
        isExist = MetricProperty.objects.get(solution_id=solution_id)
        if isExist is not None:
            return False
    except:

        newScoreObject = MetricProperty.objects.create(
            solution_id=solution_id,

            underfitting_property=data['underfitting_property'],
            overfitting_property=data['overfitting_property'],
            statistical_parity_difference_property=data['statistical_parity_difference_property'],
            equal_opportunity_difference_property=data['equal_opportunity_difference_property'],
            average_odds_difference_property=data['average_odds_difference_property'],
            disparate_impact_property=data['disparate_impact_property'],
            class_balance_property=data['class_balance_property'],
            algorithm_class_property=data['algorithm_class_property'],
            correlated_features_property=data['correlated_features_property'],
            model_size_property=data['model_size_property'],
            feature_relevance_property=data['feature_relevance_property'],
            permutation_feature_importance_property=data['permutation_feature_importance_property'],
            normalization_property=data['normalization_property'],
            missing_data_property=data['missing_data_property'],
            regularization_property=data['regularization_property'],
            train_test_split_property=data['train_test_split_property'],
            factsheet_completeness_property=data['factsheet_completeness_property'],
            confidence_score_property=data['confidence_score_property'],
            clique_method_property=data['clique_method_property'],
            clever_score_property=data['clever_score_property'],
            er_fast_gradient_attack_property=data['er_fast_gradient_attack_property'],
            er_carlini_wagner_attack_property=data['er_carlini_wagner_attack_property'],
            er_deepfool_attack_property=data['er_deepfool_attack_property'],
            loss_sensitivity_property=data['loss_sensitivity_property'],
        )

        newScoreObject.save()
    return True


def get_property(solution_id):
    try:
        isExist = MetricProperty.objects.get(solution_id=solution_id)
        uploaddic = {}
        uploaddic['underfitting_property'] = isExist.underfitting_property
        uploaddic['overfitting_property'] = isExist.overfitting_property
        uploaddic['statistical_parity_difference_property'] = isExist.statistical_parity_difference_property
        uploaddic['equal_opportunity_difference_property'] = isExist.equal_opportunity_difference_property
        uploaddic['average_odds_difference_property'] = isExist.average_odds_difference_property
        uploaddic['disparate_impact_property'] = isExist.disparate_impact_property
        uploaddic['class_balance_property'] = isExist.class_balance_property
        uploaddic['algorithm_class_property'] = isExist.algorithm_class_property
        uploaddic['correlated_features_property'] = isExist.correlated_features_property
        uploaddic['model_size_property'] = isExist.model_size_property
        uploaddic['feature_relevance_property'] = isExist.feature_relevance_property
        uploaddic['permutation_feature_importance_property'] = isExist.permutation_feature_importance_property
        uploaddic['normalization_property'] = isExist.normalization_property
        uploaddic['missing_data_property'] = isExist.missing_data_property
        uploaddic['regularization_property'] = isExist.regularization_property
        uploaddic['train_test_split_property'] = isExist.train_test_split_property
        uploaddic['factsheet_completeness_property'] = isExist.factsheet_completeness_property
        uploaddic['confidence_score_property'] = isExist.confidence_score_property
        uploaddic['clique_method_property'] = isExist.clique_method_property
        uploaddic['clever_score_property'] = isExist.clever_score_property
        uploaddic['er_fast_gradient_attack_property'] = isExist.er_fast_gradient_attack_property
        uploaddic['er_carlini_wagner_attack_property'] = isExist.er_carlini_wagner_attack_property
        uploaddic['er_deepfool_attack_property'] = isExist.er_deepfool_attack_property
        uploaddic['loss_sensitivity_property'] = isExist.loss_sensitivity_property

        return uploaddic

    except Exception as e:
        print('get property exception:', e)
        raise Exception("property not exist")


def save_metric(solution_id, data):
    print('save metric called', )
    try:
        isExist = PerformanceMetrics.objects.get(solution_id=solution_id)
        if isExist is not None:
            return False
    except:

        newScoreObject = PerformanceMetrics.objects.create(
            solution_id=solution_id,

            accuracy=data['accuracy'],
            globalrecall=data['globalrecall'],
            classweightedrecall=data['classweightedrecall'],
            globalprecision=data['globalprecision'],
            classweightedprecision=data['classweightedprecision'],
            globalf1score=data['globalf1score'],
            classweightedf1score=data['classweightedf1score'],
        )

        newScoreObject.save()
    return True


def get_metric(solution_id):
    try:
        isExist = PerformanceMetrics.objects.get(solution_id=solution_id)
        uploaddic = {}
        uploaddic['accuracy'] = isExist.accuracy
        uploaddic['globalrecall'] = isExist.globalrecall
        uploaddic['classweightedrecall'] = isExist.classweightedrecall
        uploaddic['globalprecision'] = isExist.globalprecision
        uploaddic['classweightedprecision'] = isExist.classweightedprecision
        uploaddic['globalf1score'] = isExist.globalf1score
        uploaddic['classweightedf1score'] = isExist.classweightedf1score

        return uploaddic

    except Exception as e:
        print('metric exception:', e)
        raise Exception("metric not exist")
