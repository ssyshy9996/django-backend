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

            weights_metrics1 = solution1.weights_metrics
            weights_pillars1 = solution1.weights_pillars
            model_data1 = solution1.model_file
            test_data1 = solution1.test_file
            train_data1 = solution1.training_file
            factsheet1 = solution1.factsheet_file
            test_data1, train_data1, factsheet1, model_data1, weights_pillars1, weights_metrics1 = save_files_return_paths(
                f"{test_data1}", f"{train_data1}", f"{factsheet1}", f"{model_data1}", f"{weights_pillars1}", f"{weights_metrics1}")
            test_data1 = pd.read_csv(test_data1)
            train_data1 = pd.read_csv(train_data1)
            factsheet1 = pd.read_json(factsheet1)
            weights_metrics1 = pd.read_json(weights_metrics1)
            weights_pillars1 = pd.read_json(weights_pillars1)
            print('metrics:', weights_metrics1)

            weights_metrics2 = solution2.weights_metrics
            weights_pillars2 = solution2.weights_pillars
            model_data2 = solution2.model_file
            test_data2 = solution2.test_file
            train_data2 = solution2.training_file
            factsheet2 = solution2.factsheet_file
            test_data2, train_data2, factsheet2, model_data2, weights_pillars2, weights_metrics2 = save_files_return_paths(
                f"{test_data2}", f"{train_data2}", f"{factsheet2}", f"{model_data2}", f"{weights_pillars2}", f"{weights_metrics2}")
            test_data2 = pd.read_csv(test_data2)
            train_data2 = pd.read_csv(train_data2)
            factsheet2 = pd.read_json(factsheet2)
            weights_metrics2 = pd.read_json(weights_metrics2)
            weights_pillars2 = pd.read_json(weights_pillars2)
            print('metrics:', weights_metrics2)

            metricData = {
                'fairness1': {},
                'explainability1': {},
                'robustness1': {},
                'methodology1': {},

                'fairness2': {},
                'explainability2': {},
                'robustness2': {},
                'methodology2': {},
            }

            if (solution1.solution_type == 'supervised'):
                metricData['fairness1']['main'] = weights_pillars1['pillars']['fairness']
                metricData['fairness1']['underfitting'] = weights_metrics1['fairness']['underfitting']
                metricData['fairness1']['overfitting'] = weights_metrics1['fairness']['overfitting']
                metricData['fairness1']['statistical_parity_difference'] = weights_metrics1['fairness']['statistical_parity_difference']
                metricData['fairness1']['equal_opportunity_difference'] = weights_metrics1['fairness']['equal_opportunity_difference']
                metricData['fairness1']['average_odds_difference'] = weights_metrics1['fairness']['average_odds_difference']
                metricData['fairness1']['disparate_impact'] = weights_metrics1['fairness']['disparate_impact']
                metricData['fairness1']['class_balance'] = weights_metrics1['fairness']['class_balance']

                metricData['explainability1']['main'] = weights_pillars1['pillars']['explainability']
                metricData['explainability1']['algorithm_class'] = weights_metrics1['explainability']['algorithm_class']
                metricData['explainability1']['correlated_features'] = weights_metrics1['explainability']['correlated_features']
                metricData['explainability1']['model_size'] = weights_metrics1['explainability']['model_size']
                metricData['explainability1']['feature_relevance'] = weights_metrics1['explainability']['feature_relevance']

                metricData['robustness1']['main'] = weights_pillars1['pillars']['robustness']
                metricData['robustness1']['confidence_score'] = weights_metrics1['robustness']['confidence_score']
                metricData['robustness1']['clique_method'] = weights_metrics1['robustness']['clique_method']
                metricData['robustness1']['loss_sensitivity'] = weights_metrics1['robustness']['loss_sensitivity']
                metricData['robustness1']['clever_score'] = weights_metrics1['robustness']['clever_score']
                metricData['robustness1']['er_fast_gradient_attack'] = weights_metrics1['robustness']['er_fast_gradient_attack']
                metricData['robustness1']['er_carlini_wagner_attack'] = weights_metrics1['robustness']['er_carlini_wagner_attack']
                metricData['robustness1']['er_deepfool_attack'] = weights_metrics1['robustness']['er_deepfool_attack']

                metricData['methodology1']['main'] = weights_pillars1['pillars']['accountability']
                try:
                    metricData['methodology1']['normalization'] = weights_metrics1['methodology']['normalization']
                except:
                    metricData['methodology1']['normalization'] = weights_metrics1['accountability']['normalization']
                try:
                    metricData['methodology1']['missing_data'] = weights_metrics1['methodology']['missing_data']
                except:
                    metricData['methodology1']['missing_data'] = weights_metrics1['accountability']['missing_data']
                try:
                    metricData['methodology1']['regularization'] = weights_metrics1['methodology']['regularization']
                except:
                    metricData['methodology1']['regularization'] = weights_metrics1['accountability']['regularization']
                try:
                    metricData['methodology1']['train_test_split'] = weights_metrics1['methodology']['train_test_split']
                except:
                    metricData['methodology1']['train_test_split'] = weights_metrics1['accountability']['train_test_split']
                try:
                    metricData['methodology1']['factsheet_completeness'] = weights_metrics1['methodology']['factsheet_completeness']
                except:
                    metricData['methodology1']['factsheet_completeness'] = weights_metrics1['accountability']['factsheet_completeness']
            else:
                metricData['fairness1']['main'] = weights_pillars1['pillars']['fairness']
                metricData['fairness1']['underfitting'] = weights_metrics1['fairness']['underfitting']
                metricData['fairness1']['overfitting'] = weights_metrics1['fairness']['overfitting']
                metricData['fairness1']['statistical_parity_difference'] = weights_metrics1['fairness']['statistical_parity_difference']
                metricData['fairness1']['disparate_impact'] = weights_metrics1['fairness']['disparate_impact']

                metricData['explainability1']['main'] = weights_pillars1['pillars']['explainability']
                metricData['explainability1']['correlated_features'] = weights_metrics1['explainability']['correlated_features']
                metricData['explainability1']['model_size'] = weights_metrics1['explainability']['model_size']
                metricData['explainability1']['permutation_feature_importance'] = weights_metrics1['explainability']['permutation_feature_importance']

                metricData['robustness1']['main'] = weights_pillars1['pillars']['robustness']
                metricData['robustness1']['clever_score'] = weights_metrics1['robustness']['clever_score']

                metricData['methodology1']['main'] = weights_pillars1['pillars']['accountability']
                metricData['methodology1']['regularization'] = weights_metrics1['accountability']['regularization']
                metricData['methodology1']['normalization'] = weights_metrics1['accountability']['normalization']
                metricData['methodology1']['missing_data'] = weights_metrics1['accountability']['missing_data']
                metricData['methodology1']['train_test_split'] = weights_metrics1['accountability']['train_test_split']
                metricData['methodology1']['factsheet_completeness'] = weights_metrics1['accountability']['factsheet_completeness']

            if (solution2.solution_type == 'supervised'):
                metricData['fairness2']['main'] = weights_pillars2['pillars']['fairness']
                metricData['fairness2']['underfitting'] = weights_metrics2['fairness']['underfitting']
                metricData['fairness2']['overfitting'] = weights_metrics2['fairness']['overfitting']
                metricData['fairness2']['statistical_parity_difference'] = weights_metrics2['fairness']['statistical_parity_difference']
                metricData['fairness2']['equal_opportunity_difference'] = weights_metrics2['fairness']['equal_opportunity_difference']
                metricData['fairness2']['average_odds_difference'] = weights_metrics2['fairness']['average_odds_difference']
                metricData['fairness2']['disparate_impact'] = weights_metrics2['fairness']['disparate_impact']
                metricData['fairness2']['class_balance'] = weights_metrics2['fairness']['class_balance']

                metricData['explainability2']['main'] = weights_pillars2['pillars']['explainability']
                metricData['explainability2']['algorithm_class'] = weights_metrics2['explainability']['algorithm_class']
                metricData['explainability2']['correlated_features'] = weights_metrics2['explainability']['correlated_features']
                metricData['explainability2']['model_size'] = weights_metrics2['explainability']['model_size']
                metricData['explainability2']['feature_relevance'] = weights_metrics2['explainability']['feature_relevance']

                metricData['robustness2']['main'] = weights_pillars2['pillars']['robustness']
                metricData['robustness2']['confidence_score'] = weights_metrics2['robustness']['confidence_score']
                metricData['robustness2']['clique_method'] = weights_metrics2['robustness']['clique_method']
                metricData['robustness2']['loss_sensitivity'] = weights_metrics2['robustness']['loss_sensitivity']
                metricData['robustness2']['clever_score'] = weights_metrics2['robustness']['clever_score']
                metricData['robustness2']['er_fast_gradient_attack'] = weights_metrics2['robustness']['er_fast_gradient_attack']
                metricData['robustness2']['er_carlini_wagner_attack'] = weights_metrics2['robustness']['er_carlini_wagner_attack']
                metricData['robustness2']['er_deepfool_attack'] = weights_metrics2['robustness']['er_deepfool_attack']

                metricData['methodology2']['main'] = weights_pillars2['pillars']['accountability']
                try:
                    metricData['methodology2']['normalization'] = weights_metrics2['methodology']['normalization']
                except:
                    metricData['methodology2']['normalization'] = weights_metrics2['accountability']['normalization']
                try:
                    metricData['methodology2']['missing_data'] = weights_metrics2['methodology']['missing_data']
                except:
                    metricData['methodology2']['missing_data'] = weights_metrics2['accountability']['missing_data']
                try:
                    metricData['methodology2']['regularization'] = weights_metrics2['methodology']['regularization']
                except:
                    metricData['methodology2']['regularization'] = weights_metrics2['accountability']['regularization']
                try:
                    metricData['methodology2']['train_test_split'] = weights_metrics2['methodology']['train_test_split']
                except:
                    metricData['methodology2']['train_test_split'] = weights_metrics2['accountability']['train_test_split']
                try:
                    metricData['methodology2']['factsheet_completeness'] = weights_metrics2['methodology']['factsheet_completeness']
                except:
                    metricData['methodology2']['factsheet_completeness'] = weights_metrics2['accountability']['factsheet_completeness']
            else:
                metricData['fairness2']['main'] = weights_pillars2['pillars']['fairness']
                metricData['fairness2']['underfitting'] = weights_metrics2['fairness']['underfitting']
                metricData['fairness2']['overfitting'] = weights_metrics2['fairness']['overfitting']
                metricData['fairness2']['statistical_parity_difference'] = weights_metrics2['fairness']['statistical_parity_difference']
                metricData['fairness2']['disparate_impact'] = weights_metrics2['fairness']['disparate_impact']

                metricData['explainability2']['main'] = weights_pillars2['pillars']['explainability']
                metricData['explainability2']['correlated_features'] = weights_metrics2['explainability']['correlated_features']
                metricData['explainability2']['model_size'] = weights_metrics2['explainability']['model_size']
                metricData['explainability2']['permutation_feature_importance'] = weights_metrics2['explainability']['permutation_feature_importance']

                metricData['robustness2']['main'] = weights_pillars2['pillars']['robustness']
                metricData['robustness2']['clever_score'] = weights_metrics2['robustness']['clever_score']

                metricData['methodology2']['main'] = weights_pillars2['pillars']['accountability']
                metricData['methodology2']['regularization'] = weights_metrics2['accountability']['regularization']
                metricData['methodology2']['normalization'] = weights_metrics2['accountability']['normalization']
                metricData['methodology2']['missing_data'] = weights_metrics2['accountability']['missing_data']
                metricData['methodology2']['train_test_split'] = weights_metrics2['accountability']['train_test_split']
                metricData['methodology2']['factsheet_completeness'] = weights_metrics2['accountability']['factsheet_completeness']

            uploaddic['weight'] = metricData

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
                uploaddic['class_balance'] = result1['class_balance']
                uploaddic['equal_opportunity_difference'] = result1['equal_opportunity_difference']
                uploaddic['average_odds_difference'] = result1['average_odds_difference']
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

                uploaddic['class_balance2'] = result1['class_balance']
                uploaddic['equal_opportunity_difference2'] = result1['equal_opportunity_difference']
                uploaddic['average_odds_difference2'] = result1['average_odds_difference']
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
