from django.urls import path
from .import views
from django.conf import settings
from django.conf.urls.static import static

# urlpatterns = [
#     path('',views.dataList1.as_view()),
#     path('datalist1/<str:id>',views.dataList1.as_view()),
#     path('marketcomparison',views.marketComparison),
#     path('datalist2/<str:id>',views.dataList2.as_view()),
#     path('datalist2',views.dataList2.as_view()),
#     path('create-checkout-session',views.create_checkout_session,name="create_checkout_session"),
#     path('freetrial/',views.FreeTrial,name="FreeTrial"),
#     path('registerUser',views.registerUser,name="registerUser"),
#     #path('dataDetail/<int:pk>/', views.dataDetail.as_view()),

# ]

urlpatterns = [
    # path('',views.dataList1.as_view()),
    # path('datalist1/<str:id>',views.dataList1.as_view()),

    # path('datalist2/<str:id>',views.dataList2.as_view()),
    # path('datalist2',views.dataList2.as_view()),

     path('api/scenario/', views.ScenarioList.as_view()),
     path('api/solution/', views.SolutionList.as_view()),
    # 1)Accountability Scores
    path('api/factsheet_completness_score/',
         views.get_factsheet_completness_score),
    path('api/missing_data_score/', views.get_missing_data_score),
    path('api/normalization_score/', views.get_normalization_score),
    path('api/regularization_score/', views.get_regularization_score),
    path('api/train_test_split_score/', views.get_train_test_split_score),

    # 2)Robustness Scores
    path('api/clever_score/', views.get_clever_score),
    path('api/clique_method_score/', views.get_clique_method_score),
    path('api/confidence_score/', views.get_confidence_score),
    path('api/carliwagnerwttack_score/', views.get_carliwagnerwttack_score),
    path('api/deepfoolattack_score/', views.get_deepfoolattack_score),
    path('api/fast_gradient_attack_score/',
         views.get_fast_gradient_attack_score),
    path('api/loss_sensitivity_score/', views.get_loss_sensitivity_score),

    # 3)Explainability Scores
    path('api/modelsize_score/', views.get_modelsize_score),
    path('api/correlated_features_score/', views.get_correlated_features_score),
    path('api/algorithm_class_score/', views.get_algorithm_class_score),
    path('api/feature_relevance_score/', views.get_feature_relevance_score),
    path('api/permutation_feature_importance_score/',
         views.get_permutation_feature_importance_score),


    # 4)Fairness Scores
    path('api/disparate_impact_score/', views.get_disparate_impact_score),
    path('api/class_balance_score/', views.get_class_balance_score),
    path('api/overfitting_score/', views.get_overfitting_score),
    path('api/underfitting_score/', views.get_underfitting_score),
    path('api/statistical_parity_difference_score/',
         views.get_statistical_parity_difference_score),
    path('api/equal_opportunity_difference_score/',
         views.get_equal_opportunity_difference_score),
    path('api/average_odds_difference_score/',
         views.get_average_odds_difference_score),

    # 5)Pillar Scores
    path('api/accountability_score/', views.get_accountability_score),
    path('api/robustnesss_score/', views.get_robustness_score),
    path('api/explainability_score/', views.get_explainability_score),
    path('api/fairness_score/', views.get_fairness_score),

    # 6)Trust Scores
    path('api/trustscore/', views.get_trust_score),
    path('api/trusting_AI_scores_supervised/',
         views.get_trusting_AI_scores_supervised),
    path('api/trusting_AI_scores_unsupervised/',
         views.get_trusting_AI_scores_unsupervised),

    path('create-checkout-session', views.create_checkout_session,
         name="create_checkout_session"),
    # path('freetrial/',views.FreeTrial,name="FreeTrial"),
    # path('registerUser',views.registerUser,name="registerUser"),
    path('api/registerUser', views.registerUser.as_view(), name="registerUser"),
    path('api/user', views.user.as_view(), name="user"),
    path('api/user/<str:email>/', views.user.as_view(), name="user"),
    path('api/scenario/<str:scenarioName>/<int:user_id>/',
         views.scenario_detail, name='scenario_detail'),

    path('api/dashboard/<str:email>', views.dashboard.as_view(), name="dashboard"),

    path('api/solution', views.solution.as_view()),
    path('api/solution/<str:email>', views.solution.as_view()),

    path('api/userpage/<str:email>', views.userpage.as_view()),
    path('api/userpage', views.userpage.as_view()),

    path('api/analyze', views.analyze.as_view()),

    path('api/compare', views.compare.as_view()),
    path('api/setuser/<str:email>', views.userset.as_view(), name="userset"),
    path('api/setuser', views.userset.as_view(), name="userset"),

    path('api/scenario/<int:scenarioId>',
         views.scenario.as_view(), name="scenario"),
    path('api/scenario', views.scenario.as_view(), name="scenario"),

    path('api/solution_detail/<int:id>',
         views.solutiondetail.as_view(), name="solutiondetail"),
    path('api/solution_detail', views.solutiondetail.as_view(),
         name="solutiondetail"),
    path('api/auth', views.auth.as_view(), name="auth")
    # path('api/user/<str:id>',views.user.as_view()),
    # path('dataDetail/<int:pk>/', views.dataDetail.as_view()),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

'''{
    "stockNames":["GME","AAPL","AMC"],
    "purchaseDates":["19/02/2002","17/12/1980","23/12/2013"],
    "quantitiesPurchased":[15,2,10]
}
{
    "stockNames":["GME","AAPL","AMC"],
    "purchaseDates":["19/02/2020","17/12/2021","12/01/2022"],
    "quantitiesPurchased":[15,2,10]
}
GME
13/02/2002
2002-02-13->14-15-19-2020
2021-12-07->8-9-10-13
AAPl
12/12/1980
1980-12-12->15-16-17-18
2021-12-07->8-9-10-13
AMC
18/12/2013
2013-12-18->19-20-23-24
2021-12-07->8-9-10-13
{
    "stockNames":["ABC","INMD","RDSB","BA"],
    "interval":"d",
    "quantitiesPurchased":15
}'''
