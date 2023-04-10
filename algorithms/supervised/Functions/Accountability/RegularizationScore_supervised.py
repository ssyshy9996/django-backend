def get_regularization_score_supervised(model=None, training_dataset=None, test_dataset=None, factsheet=not None, mappings=None, target_column=None, outliers_data=None, thresholds=None, outlier_thresholds=None, outlier_percentage=None, high_cor=None, print_details=None):
    import sys
    import inspect
    sys.path.append(r"Functions_Trust")
    sys.path.append(r"Functions_Trust\Backend")
    sys.path.append(r"Functions_Trust\Backend\algorithms")
    sys.path.append(r"Functions_Trust\Backend\algorithms\supervised")
    sys.path.append(r"Functions_Trust\Backend\algorithms\supervised\Functions")
    sys.path.append(
        r"Functions_Trust\Backend\algorithms\supervised\Functions\Accountability")
    try:
        from algorithms.supervised.Functions.Accountability.helpers_supervised_accountability import accountabiltiy_parameter_file_loader
    except:
        from helpers_supervised_accountability import accountabiltiy_parameter_file_loader

    metric_fname, NOT_SPECIFIED = inspect.currentframe().f_code.co_name, "not specified"
    foo = accountabiltiy_parameter_file_loader(
        metric_function_name=metric_fname, factsheet=factsheet)

    # print('foo dataA:', foo['data']['methodology'])
    np, info, result, factsheet, factsheet2 = foo['np'], foo['info'], foo[
        'result'], foo['data'], foo["data"]["methodology"]

    def regularization_metric(factsheet):
        if "methodology" in factsheet and "regularization" in factsheet["methodology"]:
            return factsheet["methodology"]["regularization"]
        else:
            return NOT_SPECIFIED
    score = 1
    regularization = regularization_metric(factsheet)
    properties = {"dep": info('Depends on', 'Factsheet'),
                  "regularization_technique": info("Regularization technique", regularization)}

    if regularization == "elasticnet_regression":
        score = 5
    elif regularization == "lasso_regression" or regularization == "lasso_regression":
        score = 4
    elif regularization == "Other":
        score = 3
    elif regularization == NOT_SPECIFIED:
        score = 1
    else:
        score = 1
    return result(score=score, properties=properties)


"""########################################TEST VALUES#############################################
factsheet=r"Validation\supervised\Jans Sgd Classifier\factsheet.json"
print(get_regularization_score_supervised(factsheet=factsheet),"\n")"""
