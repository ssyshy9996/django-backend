def get_statistical_parity_difference_score_unsupervised(model=not None, training_dataset=not None, test_dataset=None, factsheet=not None, mappings=not None, target_column=None, outliers_data=not None, thresholds=not None, outlier_thresholds=not None, penalty_outlier=None, outlier_percentage=None, high_cor=None, print_details=None):
    import collections
    import numpy
    import sys
    import pandas as pd
    sys.path.extend([r"Backend", r"Backend/algorithms", r"Backend/algorithms/unsupervised", r"Backend/algorithms/unsupervised/Functions", r"Backend/algorithms/unsupervised/Functions/Accountability",
                    r"Backend/algorithms/unsupervised/Functions/Fairness", r"Backend/algorithms/unsupervised/Functions/Explainability", r"Backend/algorithms/unsupervised/Functions/Robustness"])
    try:
        from algorithms.unsupervised.Functions.Fairness.helpers_fairness_unsupervised import read_model
        from algorithms.unsupervised.Functions.Fairness.helpers_fairness_unsupervised import compute_outlier_ratio, get_threshold_mse_iqr, isKerasAutoencoder, load_fairness_config, detect_outliers, isIsolationForest
    except:
        from unsupervised.Functions.Fairness.helpers_fairness_unsupervised import read_model
        from unsupervised.Functions.Fairness.helpers_fairness_unsupervised import compute_outlier_ratio, get_threshold_mse_iqr, isKerasAutoencoder, load_fairness_config, detect_outliers, isIsolationForest

    info, result = collections.namedtuple(
        'info', 'description value'), collections.namedtuple('result', 'score properties')

    factsheet = pd.read_json(factsheet)
    training_dataset = pd.read_csv(training_dataset)
    outliers_data = pd.read_csv(outliers_data)
    model = read_model(model)
    mappings = pd.read_json(mappings)

    if not thresholds:
        thresholds = mappings["fairness"]["score_overfitting"]["thresholds"]["value"]

    try:
        protected_feature, protected_values = load_fairness_config(factsheet)

        minority = training_dataset[training_dataset[protected_feature].isin(
            protected_values)]
        minority_size = len(minority)
        majority = training_dataset[~training_dataset[protected_feature].isin(
            protected_values)]
        majority_size = len(majority)

        if isKerasAutoencoder(model):
            thresh = get_threshold_mse_iqr(model, training_dataset)
            mad_outliers = detect_outliers(model, training_dataset, thresh)
            outlier_indices = [i for i, el in enumerate(
                mad_outliers[0].tolist()) if el == False]

        elif isIsolationForest(model):
            mad_outliers = model.predict(training_dataset)
            outlier_indices = [i for i, el in enumerate(
                mad_outliers.tolist()) if el == -1]

        else:
            mad_outliers = model.predict(training_dataset)
            outlier_indices = [i for i, el in enumerate(
                mad_outliers.tolist()) if el == 1]

        minority_indices = minority.index.tolist()
        majority_indices = majority.index.tolist()

        # measure num of outliers in majority group by intersection of indices
        num_outliers_minority = len(
            list(set(minority_indices) & set(outlier_indices)))
        num_outliers_majority = len(
            list(set(majority_indices) & set(outlier_indices)))

        favored_minority_ratio = num_outliers_minority / minority_size
        favored_majority_ratio = num_outliers_majority / majority_size

        statistical_parity_difference = abs(
            favored_minority_ratio - favored_majority_ratio)

        if print_details:
            print("\t protected feature: ", protected_feature)
            print("\t protected values: ", protected_values)
            print("\t group size: ", len(majority_indices), len(minority_indices))
            print("\t num outlier: ", num_outliers_majority, num_outliers_minority)
            print("\t outlier ratios: %.4f " %
                  favored_majority_ratio, "%.4f " % favored_minority_ratio)
            print("\t statistical_parity_difference: %.4f" %
                  statistical_parity_difference)

        properties = {}
        properties["Metric Description"] = "The spread between the percentage of detected outliers in the majority group compared to the protected group. The closer this spread is to zero the better."
        properties["Depends on"] = "Training Data, Factsheet (Definition of Protected Group and Favorable Outcome)"

        properties['----------'] = ''
        properties["protected feature: "] = protected_feature
        properties["protected values: "] = str(protected_values)
        properties['-----------'] = ''
        properties["Formula"] = "Statistical Parity Difference = |Favored Protected Group Ratio - Favored Unprotected Group Ratio|"
        properties["Statistical Parity Difference"] = "{:.2f}%".format(
            statistical_parity_difference*100)

        properties["|{x|x is protected, y_true is favorable}|"] = num_outliers_minority
        properties["|{x|x is protected}|"] = minority_size
        properties["Favored Protected Group Ratio"] = "P(y_true is favorable|protected=True) = {:.2f}%".format(
            num_outliers_minority/minority_size * 100)
        properties["|{x|x is not protected, y_true is favorable}|"] = num_outliers_majority
        properties["|{x|x is not protected}|"] = majority_size
        properties["Favored Unprotected Group Ratio"] = "P(y_true is favorable|protected=False) = {:.2f}%".format(
            num_outliers_majority/majority_size * 100)

        score = numpy.digitize(
            statistical_parity_difference, thresholds, right=False) + 1

        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)

    except Exception as e:
        print("ERROR in statistical_parity_difference_score(): {}".format(e))
        return result(score=numpy.nan, properties={"Non computable because": str(e)})
