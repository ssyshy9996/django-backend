def underfitting_score(model=not None, training_dataset=not None, test_dataset=None, factsheet= None, mappings=not None,target_column=None, outliers_data=not None, thresholds= not None, outlier_thresholds= not None,penalty_outlier=  None, outlier_percentage= None, high_cor=None,print_details=None):
    import collections, numpy,sys, pandas as pd
    sys.path.extend([r"Backend",r"Backend/algorithms",r"Backend/algorithms/unsupervised", r"Backend/algorithms/unsupervised/Functions", r"Backend/algorithms/unsupervised/Functions/Accountability",r"Backend/algorithms/unsupervised/Functions/Fairness",r"Backend/algorithms/unsupervised/Functions/Explainability",r"Backend/algorithms/unsupervised/Functions/Robustness"])
    try:
        from algorithms.unsupervised.Functions.Fairness.helpers_fairness_unsupervised import read_model
        from algorithms.unsupervised.Functions.Fairness.helpers_fairness_unsupervised import compute_outlier_ratio, get_threshold_mse_iqr, isKerasAutoencoder,isIsolationForest
    except:
        from unsupervised.Functions.Fairness.helpers_fairness_unsupervised import read_model
        from unsupervised.Functions.Fairness.helpers_fairness_unsupervised import compute_outlier_ratio, get_threshold_mse_iqr, isKerasAutoencoder,isIsolationForest

    info,result = collections.namedtuple('info', 'description value'), collections.namedtuple('result', 'score properties')
    
    training_dataset=pd.read_csv(training_dataset)
    outliers_data=pd.read_csv(outliers_data)
    model=read_model(model)
    mappings=pd.read_json(mappings)

    if not thresholds:
        thresholds= mappings["fairness"]["score_underfitting"]["thresholds"]["value"]
    if isKerasAutoencoder(model):
        outlier_thresh = get_threshold_mse_iqr(model, training_dataset)

    try:
        properties = {}
        properties['Metric Description'] = "Computes the difference of outlier detection ratio in the training and test data."
        properties['Depends on'] = 'Model, Train Data, Test Data'
        score = 0

        detection_ratio_train = compute_outlier_ratio(model=model, data=outliers_data, outlier_thresh=outlier_thresholds)
        detection_ratio_test = compute_outlier_ratio(model=model, data=outliers_data, outlier_thresh=outlier_thresholds)

        perc_diff = abs(detection_ratio_train - detection_ratio_test)
        score = numpy.digitize(perc_diff, thresholds, right=False) + 1
        print("SCORE: ",score)
        

        if print_details:
            print("\t   UNDERFITTING DETAILS")
            print("\t model is AutoEncoder: ", isKerasAutoencoder(model))
            print("\t model is IsolationForest: ", isIsolationForest(model))
            print("\t detected outlier ratio in training data: %.4f" % detection_ratio_train)
            print("\t detected outlier ratio in validation data: %.4f" % detection_ratio_test)
            print("\t absolute difference: %.4f" % perc_diff)

        properties["Train Data Outlier Detection Ratio"] = "{:.2f}%".format(detection_ratio_train*100)
        properties["Test Data Outlier Detection Ratio"] = "{:.2f}%".format(detection_ratio_test*100)
        properties["Absolute Difference"] = "{:.2f}%".format(perc_diff*100)

        if score == 5:
            properties["Conclusion"] = "Model is not underfitting"
        elif score == 4:
            properties["Conclusion"] = "Model mildly underfitting"
        elif score == 3:
            properties["Conclusion"] = "Model is slighly underfitting"
        elif score == 2:
            properties["Conclusion"] = "Model is underfitting"
        else:
            properties["Conclusion"] = "Model is strongly underfitting"

        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)
    
    except Exception as e:
        print("ERROR in underfitting_score(): {}".format(e))
        return result(score=numpy.nan, properties={"Non computable because": str(e)}) 

"""########################################TEST VALUES#############################################
train=r"Backend/algorithms/unsupervised/TestValues/train.csv"
test=r"Backend/algorithms/unsupervised/TestValues/test.csv"
outliers=r"Backend/algorithms/unsupervised/TestValues/outliers.csv"
model=r"Backend/algorithms/unsupervised/TestValues/model.joblib"
mapping_metrics_default=r"Backend/algorithms/unsupervised/Mapping&Weights/mapping_metrics_default.json"
a= underfitting_score(model=model, training_dataset=train, test_dataset=test, outliers_data=outliers, mappings=mapping_metrics_default,outlier_percentage=0.1, outlier_thresholds=0, print_details = False,thresholds=None)
print(a)"""