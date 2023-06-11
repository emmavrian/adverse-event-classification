from ml_classification_selection import supervised_ml_classification, semi_supervised_ml_classification, ml_cross_validation, hyperparameter_tuning
import pandas as pd
import numpy as np
from ml_classification_base_methods import create_supervised_ml_pipeline, create_semi_supervised_ml_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, f1_score

def supervised_classification_all_models_10_random(X, y):
    random_states = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dt_supervised_stats = []
    gb_supervised_stats = []
    lg_supervised_stats = []
    mnb_supervised_stats = []
    cnb_supervised_stats = []
    svm_supervised_stats = []
    rf_supervised_stats = []


    # logistic regression
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'logistic_regression', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        lg_supervised_stats.append(stats)

    lg_supervised_stats_df = pd.DataFrame(lg_supervised_stats)
    lg_supervised_scores = lg_supervised_stats_df.round(2)

    # multinomial naive bayes
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'multinomial_naive_bayes', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        mnb_supervised_stats.append(stats)

    mnb_supervised_stats_df = pd.DataFrame(mnb_supervised_stats)
    mnb_supervised_scores = mnb_supervised_stats_df.round(2)


    # complement naive bayes
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'complement_naive_bayes', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        cnb_supervised_stats.append(stats)

    cnb_supervised_stats_df = pd.DataFrame(cnb_supervised_stats)
    cnb_supervised_scores = cnb_supervised_stats_df.round(2)
    

    # linear svm
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'linear_svm', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        svm_supervised_stats.append(stats)

    svm_supervised_stats_df = pd.DataFrame(svm_supervised_stats)
    svm_supervised_scores = svm_supervised_stats_df.round(2)

    # decision tree
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'decision_tree', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        dt_supervised_stats.append(stats)

    dt_supervised_stats_df = pd.DataFrame(dt_supervised_stats)
    dt_supervised_scores = dt_supervised_stats_df.round(2)

    # random forest
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'random_forest', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        rf_supervised_stats.append(stats)

    rf_supervised_stats_df = pd.DataFrame(rf_supervised_stats)
    rf_supervised_scores = rf_supervised_stats_df.round(2)


    # gradient boosting
    for random_state in random_states:
        stats = supervised_ml_classification(X, y, 'gradient_boosting', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        gb_supervised_stats.append(stats)

    gb_supervised_stats_df = pd.DataFrame(gb_supervised_stats)
    gb_supervised_scores = gb_supervised_stats_df.round(2)

    list_of_all_supervised_results = [lg_supervised_stats_df, mnb_supervised_stats_df, cnb_supervised_stats_df, svm_supervised_stats_df, dt_supervised_stats_df, rf_supervised_stats_df, gb_supervised_stats_df]
    supervised_results_df = pd.concat(list_of_all_supervised_results)

    # save all results to pickle for easy load
    supervised_results_df.to_pickle("filename")
    supervised_results_df.to_csv("filename")

    
    return supervised_results_df


def cross_validation_all_models_10_random(X, y):
    random_states = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dt_cv_stats = []
    gb_cv_stats = []
    lg_cv_stats = []
    mnb_cv_stats = []
    cnb_cv_stats = []
    svm_cv_stats = []
    rf_cv_stats = []

    # decision tree
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'decision_tree', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        dt_cv_stats.append(stats)

    dt_cv_stats_df = pd.DataFrame(dt_cv_stats)

    # gradient boosting
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'gradient_boosting', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        gb_cv_stats.append(stats)

    gb_cv_stats_df = pd.DataFrame(gb_cv_stats)

    # logistic regression
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'logistic_regression', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        lg_cv_stats.append(stats)

    lg_cv_stats_df = pd.DataFrame(lg_cv_stats)

    # multinomial naive bayes
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'multinomial_naive_bayes', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        mnb_cv_stats.append(stats)

    mnb_cv_stats_df = pd.DataFrame(mnb_cv_stats)


    # complement naive bayes
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'complement_naive_bayes', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        cnb_cv_stats.append(stats)

    cnb_cv_stats_df = pd.DataFrame(cnb_cv_stats)
    
    # linear svm
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'linear_svm', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        svm_cv_stats.append(stats)

    svm_cv_stats_df = pd.DataFrame(svm_cv_stats)

    # random forest
    for random_state in random_states:
        stats = ml_cross_validation(X, y, 'random_forest', 'count_vectorizer', 'classifier_chain', random_state=random_state, print_info=False)
        rf_cv_stats.append(stats)

    rf_cv_stats_df = pd.DataFrame(rf_cv_stats)

    list_of_all_cv_results = [lg_cv_stats_df, mnb_cv_stats_df, cnb_cv_stats_df, svm_cv_stats_df, dt_cv_stats_df, rf_cv_stats_df, gb_cv_stats_df]
    cv_results_df = pd.concat(list_of_all_cv_results)

    # save all results to pickle for easy load
    cv_results_df.to_pickle("filename")
    cv_results_df.to_csv("filename")

    return cv_results_df


def semi_supervised_classification_gb_varying_data_threshold(X, y, X_train_unlab):

    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    amount_of_unlabeled_samples = [100, 200, 300, 400, 500]

    semi_supervised_stats = []

    for sample_amount in amount_of_unlabeled_samples:
        for threshold in thresholds:
            stats = semi_supervised_ml_classification(X, y, X_train_unlab[:sample_amount], "gradient_boosting", 'count_vectorizer', 'classifier_chain', threshold=threshold, random_state=42, print_info=False)
            semi_supervised_stats.append(stats)
    

    stats_df = pd.DataFrame(semi_supervised_stats)
    
    stats_df.to_pickle("filename")
    stats_df.to_csv("filename")

    return semi_supervised_stats


### HYPERPARAMETER TUNING FUNCTION NOT USED, JUST TESTED ###

def supervised_classification_parameter_tuning(X, y):
    # NOTE: best parameters are very sensitive to choice of random state. Because of this, doing parameter tuning will likely lead to overfitting and should be avoided.
    lg_parameter_tuning = hyperparameter_tuning(X, y, 'logistic_regression', 'count_vectorizer', 'classifier_chain', random_state=42)
    mnb_parameter_tuning = hyperparameter_tuning(X, y, 'multinomial_naive_bayes', 'count_vectorizer', 'classifier_chain', random_state=42)
    cnb_parameter_tuning = hyperparameter_tuning(X, y, 'logistic_regression', 'count_vectorizer', 'classifier_chain', random_state=42)

    print("\n####\n")

    lg_parameter_tuning = hyperparameter_tuning(X, y, 'logistic_regression', 'count_vectorizer', 'classifier_chain', random_state=89)
    mnb_parameter_tuning = hyperparameter_tuning(X, y, 'multinomial_naive_bayes', 'count_vectorizer', 'classifier_chain', random_state=89)
    cnb_parameter_tuning = hyperparameter_tuning(X, y, 'logistic_regression', 'count_vectorizer', 'classifier_chain', random_state=89)

    return None

