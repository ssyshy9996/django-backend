def get_accountability_score_supervised(model=not None, training_dataset=not None, test_dataset=not None, factsheet=not None, mappings=not None,target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None,print_details=None):
    
    
    import sys
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/supervised", r"Backend/algorithms/supervised/Functions",r"Backend/algorithms/supervised/Functions/Accountability"])
    from Functions.helpers_supervised import import_functions_from_folder
    info,result,accountability_functions=import_functions_from_folder(['Accountability'])
    get_normalization_score,get_missing_data_score, get_regularization_score,get_train_test_split_score,get_factsheetcomplettness_score=accountability_functions['normalizationscoresupervised'],accountability_functions['missingdatascoresupervised'],accountability_functions['regularizationscoresupervised'],accountability_functions['traintestsplitscoresupervised'],accountability_functions['factsheetcompletnessscoresupervised']
    output = dict(  
        normalization  = get_normalization_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        missing_data  = get_missing_data_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        regularization  = get_regularization_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        train_test_split  = get_train_test_split_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
        factsheet_completeness  = get_factsheetcomplettness_score(model=model, training_dataset=training_dataset, test_dataset=test_dataset, outliers_data=outliers_data, factsheet=factsheet, mappings=mappings, thresholds=thresholds, outlier_thresholds=outlier_thresholds, outlier_percentage=outlier_percentage, print_details=print_details),
    )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

"""########################################TEST VALUES#############################################
model,train,test,factsheet,mappigns=r"Backend/algorithms/supervised/TestValues/model.pkl", r"Backend/algorithms/supervised/TestValues/train.csv", r"Backend/algorithms/supervised/TestValues/test.csv", r"Backend/algorithms/supervised/TestValues/factsheet.json", r"Backend/algorithms/supervised/Mapping&Weights/mapping_metrics_default.json"
print(get_accountability_score_supervised(model=model,training_dataset=train,test_dataset=test,factsheet=factsheet,mappings=mappigns))"""
