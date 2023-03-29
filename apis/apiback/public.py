from algorithms.supervised.Functions.Fairness.ClassBalanceScore_supervised import get_class_balance_score_supervised
import sys
import django
from ..models import Score

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
