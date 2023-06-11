from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.pipeline import Pipeline


from method_param_dicts import fetch_method_dict


import pandas as pd
import numpy as np


def create_supervised_ml_pipeline(base_clf_name, vectorizer_name, multilabel_name):
    
    base_clf_dict, vectorizer_dict = fetch_method_dict()
    
    try:
        base_clf = base_clf_dict[base_clf_name]
    except:
        raise ValueError("Base classifier name not recognized. Must be either logistic_regression, multinomial_naive_bayes, complement_naive_bayes, linear_svm, random_forest, gradient_boosting or decision_tree.")
    
    try:
        vectorizer = vectorizer_dict[vectorizer_name]
    except:
        raise ValueError("Vectorizer name not recognized. Must be either count_vectorizer or tfidf_vectorizer.")


    # create a supervised multi label pipeline for the labeled data
    if(multilabel_name=='classifier_chain'):
        pipe = Pipeline([('vectorizer', vectorizer), ('classifier', ClassifierChain(base_estimator=base_clf))])
    elif(multilabel_name=='multioutput_classifier'):
        pipe = Pipeline([('vectorizer', vectorizer), ('classifier', MultiOutputClassifier(base_clf))])
    else:
        raise ValueError("Multilabel method not recognized. Must be either classifier_chain or multioutput_classifier.")

    return pipe



def create_semi_supervised_ml_pipeline(base_clf_name, vectorizer_name, multilabel_name, threshold):
    
    base_clf_dict, vectorizer_dict = fetch_method_dict()
    
    try:
        base_clf = base_clf_dict[base_clf_name]
    except:
        raise ValueError("Base classifier name not recognized. Must be either logistic_regression, multinomial_naive_bayes, complement_naive_bayes, linear_svm, random_forest or gradient_boosting.")
    
    try:
        vectorizer = vectorizer_dict[vectorizer_name]
    except:
        raise ValueError("Vectorizer name not recognized. Must be either count_vectorizer or tfidf_vectorizer.")

    # create a semi-supervised multi label pipeline for the labeled+unlabeled data using SelfTrainingClassifier
    if(multilabel_name=='classifier_chain'):
        pipe = Pipeline([('vectorizer', vectorizer), ('classifier', ClassifierChain(base_estimator=SelfTrainingClassifier(base_estimator=base_clf, threshold=threshold, verbose=True), verbose=True))])
    elif(multilabel_name=='multioutput_classifier'):
        pipe = Pipeline([('vectorizer', vectorizer), ('classifier', MultiOutputClassifier(SelfTrainingClassifier(base_estimator=base_clf, threshold=threshold, verbose=True)))])
    else:
        raise ValueError("Multilabel method not recognized. Must be either classifier_chain or multioutput_classifier.")

    return pipe