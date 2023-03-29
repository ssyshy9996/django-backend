from rest_framework.views import APIView
from rest_framework.response import Response
from ...models import CustomUser, ScenarioSolution
from ..public import get_score, save_files_return_paths
from .analyze import get_properties_section, get_properties_section_unsupervised
import pandas as pd


class compare(APIView):
    def post(self, request):
        uploaddic = {}

        if request.data is not None:
            userexist = CustomUser.objects.get(
                email=request.data['emailid'])
            solution1 = ScenarioSolution.objects.get(
                user_id=userexist.id,
                solution_name=request.data['SelectSolution1']
            )
            solution2 = ScenarioSolution.objects.get(
                user_id=userexist.id,
                solution_name=request.data['SelectSolution2']
            )

            # # for solution 1
            # model_data = solution1.model_file
            # test_data = solution1.test_file
            # train_data = solution1.training_file
            # factsheet = solution1.factsheet_file
            # test_data, train_data, factsheet, model_data = save_files_return_paths(
            #     f"{test_data}", f"{train_data}", f"{factsheet}", f"{model_data}")
            # test_data = pd.read_csv(test_data)
            # train_data = pd.read_csv(train_data)
            # factsheet = pd.read_json(factsheet)
            # uploaddic['ScenarioName'] = solution1.solution_name
            # uploaddic['Description'] = solution1.description
            # if (solution1.solution_type == 'supervised'):
            #     data = get_properties_section(
            #         train_data, test_data, factsheet)
            #     uploaddic['ModelType'] = data[data.columns[1]][0]
            #     uploaddic['TrainTestSplit'] = data[data.columns[1]][1]
            #     uploaddic['DataSize'] = data[data.columns[1]][2]
            #     uploaddic['NormalizationTechnique'] = data[data.columns[1]][3]
            #     uploaddic['NumberofFeatures'] = data[data.columns[1]][4]
            # else:
            #     data = get_properties_section_unsupervised(
            #         train_data, test_data, factsheet)
            #     # uploaddic['ModelType'] = data[data.columns[1]][0]
            #     uploaddic['TrainTestSplit'] = data[data.columns[1]][0]
            #     uploaddic['DataSize'] = data[data.columns[1]][1]
            #     uploaddic['NormalizationTechnique'] = data[data.columns[1]][2]
            #     uploaddic['NumberofFeatures'] = data[data.columns[1]][3]

            # # for solution 2

            try:
                result1 = get_score(solution1.id)
                result2 = get_score(solution2.id)

                uploaddic['fairness_score1'] = result1['fairness_score']
                uploaddic['overfitting'] = result1['overfitting']
                uploaddic['underfitting'] = result1['underfitting']
                uploaddic['disparate_impact'] = result1['disparate_impact']
                uploaddic['statistical_parity_difference'] = result1['statistical_parity_difference']
                uploaddic['normalization'] = result1['normalization']
                uploaddic['missing_data'] = result1['missing_data']
                uploaddic['regularization'] = result1['regularization']
                uploaddic['train_test_split'] = result1['train_test_split']
                uploaddic['factsheet_completeness'] = result1['factsheet_completeness']
                uploaddic['correlated_features'] = result1['correlated_features']
                uploaddic['permutation_feature_importance'] = result1['permutation_feature_importance']
                uploaddic['model_size'] = result1['model_size']
                uploaddic['algorithm_class'] = result1['algorithm_class']
                uploaddic['feature_relevance'] = result1['feature_relevance']
                uploaddic['confidence_score'] = result1['confidence_score']
                uploaddic['clique_method'] = result1['clique_method']
                uploaddic['loss_sensitivity'] = result1['loss_sensitivity']
                uploaddic['clever_score'] = result1['clever_score']
                uploaddic['er_fast_gradient_attack'] = result1['er_fast_gradient_attack']
                uploaddic['er_carlini_wagner_attack'] = result1['er_carlini_wagner_attack']
                uploaddic['er_deepfool_attack'] = result1['er_deepfool_attack']
                uploaddic['explainability_score1'] = result1['explainability_score']
                uploaddic['robustness_score1'] = result1['robustness_score']
                uploaddic['methodology_score1'] = result1['methodology_score']
                uploaddic['trust_score1'] = result1['trust_score']

                uploaddic['disparate_impact2'] = result2['disparate_impact']
                uploaddic['underfitting2'] = result2['underfitting']
                uploaddic['overfitting2'] = result2['overfitting']
                uploaddic['statistical_parity_difference2'] = result2['statistical_parity_difference']
                uploaddic['fairness_score2'] = result2['fairness_score']
                uploaddic['correlated_features2'] = result2['correlated_features']
                uploaddic['permutation_feature_importance2'] = result2['permutation_feature_importance']
                uploaddic['model_size2'] = result2['model_size']
                uploaddic['explainability_score2'] = result2['explainability_score']
                uploaddic['confidence_score2'] = result2['confidence_score']
                uploaddic['algorithm_class2'] = result2['algorithm_class']
                uploaddic['clique_method2'] = result2['clique_method']
                uploaddic['feature_relevance2'] = result2['feature_relevance']
                uploaddic['loss_sensitivity2'] = result2['loss_sensitivity']
                uploaddic['clever_score2'] = result2['clever_score']
                uploaddic['er_fast_gradient_attack2'] = result2['er_fast_gradient_attack']
                uploaddic['er_carlini_wagner_attack2'] = result2['er_carlini_wagner_attack']
                uploaddic['er_deepfool_attack2'] = result2['er_deepfool_attack']
                uploaddic['robustness_score2'] = result2['robustness_score']
                uploaddic['normalization2'] = result2['normalization']
                uploaddic['missing_data2'] = result2['missing_data']
                uploaddic['regularization2'] = result2['regularization']
                uploaddic['train_test_split2'] = result2['train_test_split']
                uploaddic['factsheet_completeness2'] = result2['factsheet_completeness']
                uploaddic['methodology_score2'] = result2['methodology_score']
                uploaddic['trust_score2'] = result2['trust_score']

                return Response(uploaddic, status=200)
            except Exception as e:
                print('asdfasdfsdaf:', e)
                return Response("You must analyze the solution before compare", status=400)
