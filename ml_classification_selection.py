from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.pipeline import Pipeline

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import hamming_loss, f1_score, jaccard_score, classification_report

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from method_param_dicts import fetch_method_dict, fetch_params_dict
from ml_classification_base_methods import create_supervised_ml_pipeline, create_semi_supervised_ml_pipeline

from sklearn.metrics import hamming_loss, f1_score

import pandas as pd
import numpy as np



def ml_cross_validation(X, y, base_clf_name, vectorizer_name, multilabel_name, random_state=42, print_info=False):

    # create supervised pipeline
    pipe = create_supervised_ml_pipeline(base_clf_name=base_clf_name, vectorizer_name=vectorizer_name, multilabel_name=multilabel_name)

    # define k_folds
    stratified_k_fold = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)


    scoring = {'ham_loss': make_scorer(hamming_loss), 
               'micro_f1': make_scorer(f1_score, average='micro'),
               'macro_f1': make_scorer(f1_score, average='macro')}
    
    stratified_cv_results = cross_validate(pipe, X.processed_text, y, cv=stratified_k_fold, scoring=scoring)

    stats = {'model': base_clf_name,
            'vectorizer': vectorizer_name,
            'multilabel_method': multilabel_name,
            'random_state': random_state,
            'method': "3-fold-cross-validation",
            '3_fold_test_ham_loss': stratified_cv_results['test_ham_loss'], 
            '3_fold_test_micro_f1': stratified_cv_results['test_micro_f1'], 
            '3_fold_test_macro_f1': stratified_cv_results['test_macro_f1'],
            'avg_3_fold_ham_loss': np.mean(stratified_cv_results['test_ham_loss']),
            'std_3_fold_ham_loss': np.std(stratified_cv_results['test_ham_loss']),
            'avg_3_fold_micro_f1': np.mean(stratified_cv_results['test_micro_f1']),
            'std_3_fold_micro_f1': np.std(stratified_cv_results['test_micro_f1']),
            'avg_3_fold_macro_f1': np.mean(stratified_cv_results['test_macro_f1']),
            'std_3_fold_weighted_jaccard': np.std(stratified_cv_results['test_macro_f1'])
        }
    
    if(print_info):
        print("----------------------------------------")
        print("CROSS VALIDATION FOR MODEL SELECTION USING FULL LABELED DATASET (", len(X) ,"NOTES )")
        print("BASE CLASSIFIER: ", base_clf_name)
        print("MULTILABEL METHOD: ", multilabel_name)
        print("VECTORIZER: ", vectorizer_name)
        print()
        print("----------------------------------------")
        print("Results for 3-fold CV of ", base_clf_name)
        print("Hamming loss per fold: ", stats['3_fold_test_ham_loss'])
        print("Average hamming loss with std: ", stats['avg_3_fold_ham_loss'], "+-", stats['std_3_fold_ham_loss'])
        print("Micro F1-score per fold: ", stats['3_fold_test_micro_f1'])
        print("Average micro F1-score with std: ", stats['avg_3_fold_micro_f1'], "+-", stats['std_3_fold_micro_f1'])
        print("Macro F1-score per fold: ", stats['3_fold_test_macro_f1'])
        print("Average weighted macro F1-score with std: ", stats['avg_3_fold_macro_f1'], "+-", stats['std_3_fold_macro_f1'])
        print("----------------------------------------")
        print()

    return stats



def supervised_ml_classification(X, y, base_clf_name, vectorizer_name, multilabel_name, random_state=42, print_info=False):

    # split labeled data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state)

    # create supervised pipeline
    pipe = create_supervised_ml_pipeline(base_clf_name=base_clf_name, vectorizer_name=vectorizer_name, multilabel_name=multilabel_name)

    # train/fit the pipeline on the labeled data X_train and y_train
    pipe.fit(X_train.processed_text, y_train)

    # use the trained pipeline to predict labels for X_test and calculate score based on y_test
    y_pred = pipe.predict(X_test.processed_text)

    #calculate metrics
    ham_loss = hamming_loss(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    f1_per_label = f1_score(y_test, y_pred, average=None)

    stats = {'model': base_clf_name,
             'vectorizer': vectorizer_name,
             'multilabel_method': multilabel_name,
             'random_state': random_state,
             'method': "supervised",
             'number_of_train_samples': len(X_train),
             'number_of_test_samples': len(X_test),
             'test_ham_loss': ham_loss, 
             'test_micro_f1': micro_f1,
             'test_macro_f1': macro_f1, 
             'test_f1_venous_catheter': f1_per_label[0], 
             'test_f1_infection': f1_per_label[1], 
            }

    if(print_info):
        print("----------------------------------------")
        print("METHOD: SUPERVISED, FULL TRAINING DATA (", len(X_train) ,"NOTES ) TEST DATA: ", len(X_test), "NOTES.")
        print("BASE CLASSIFIER: ", base_clf_name)
        print("MULTILABEL METHOD: ", multilabel_name)
        print("VECTORIZER: ", vectorizer_name)
        print()
        print("Results for prediction of X_test: ")
        print("Hamming loss: ", stats['test_ham_loss'])
        print("Micro F1-score: ", stats['test_micro_f1'])
        print("Macro F1-score: ", stats['test_macro_f1'])
        print("F1-score venous catheter: ", stats['test_f1_venous_catheter'])
        print("F1-score infection: ", stats['test_f1_infection'])
        #print(classification_report(y_test, y_pred, target_names=["Venous Catheter", "Infection"]))
        print("----------------------------------------")
        print()

    return stats


def semi_supervised_ml_classification(X, y, X_train_unlab, base_clf_name, vectorizer_name, multilabel_name, threshold=0.7, random_state=42, print_info=False):

    # split labeled data into train and test
    X_train_lab, X_test, y_train_lab, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state)

    # create semi-supervised pipeline
    pipe = create_semi_supervised_ml_pipeline(base_clf_name=base_clf_name, vectorizer_name=vectorizer_name, multilabel_name=multilabel_name, threshold=threshold)
    
    # create the training dataset for semi supervised (both labeled and unlabeled)
    X_train_mixed = pd.concat([X_train_lab, X_train_unlab])
    y_train_nolabel = pd.DataFrame(index=X_train_unlab.index, columns=['venous_catheter', 'infection'])
    y_train_nolabel['venous_catheter'] = -1
    y_train_nolabel['infection'] = -1
    # recombine training dataset labels
    y_train_mixed = pd.concat([y_train_lab, y_train_nolabel])
    
    print("\n\nThreshold = ", threshold)
    print("Amount of unlabeled data = ", len(X_train_unlab))
    print("Processing ...\n")

    # fit the semi supervised pipeline on labeled+unlabeled training data, the SelfTrainingClassifier will add pseudo-labels to the unlabeled
    pipe.fit(X_train_mixed.processed_text, y_train_mixed)

    # use the trained semi-supervised pipeline to predict labels for X_test and calculate score based on y_test
    y_pred = pipe.predict(X_test.processed_text)

    #calculate metrics
    ham_loss = hamming_loss(y_test, y_pred)
    #weighted_jaccard = jaccard_score(y_test, y_pred, average='weighted')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    f1_per_label = f1_score(y_test, y_pred, average=None)

    stats = {'model': base_clf_name,
             'vectorizer': vectorizer_name,
             'multilabel_method': multilabel_name,
             'random_state': random_state,
             'method': "semi-supervised",
             'threshold': threshold,
             'number_of_labeled_train_samples': len(X_train_lab),
             'number_of_unlabeled_train_samples': len(X_train_unlab),
             'number_of_test_samples': len(X_test),
             'test_ham_loss': ham_loss, 
             'test_micro_f1': micro_f1,
             'test_macro_f1': macro_f1, 
             'test_f1_venous_catheter': f1_per_label[0], 
             'test_f1_infection': f1_per_label[1], 
            }

    if(print_info):
        print("----------------------------------------")
        print("METHOD: SEMI-SUPERVISED, FULL TRAINING DATA (", len(X_train_lab) ,"NOTES ) AND UNLABELED DATA (", len(X_train_unlab), "NOTES ). TEST DATA: ", len(X_test), "NOTES.")
        print("BASE CLASSIFIER: ", base_clf_name)
        print("MULTILABEL METHOD: ", multilabel_name)
        print("VECTORIZER: ", vectorizer_name)
        print()
        print("Results for prediction of X_test: ")
        print("Hamming loss: ", stats['test_ham_loss'])
        print("Micro F1-score: ", stats['test_micro_f1'])
        print("Macro F1-score: ", stats['test_macro_f1'])
        print("F1-score venous catheter: ", stats['test_f1_venous_catheter'])
        print("F1-score infection: ", stats['test_f1_infection'])
        print("----------------------------------------")
        print()

    return stats


### NOT USED, JUST FOR TESTING
def hyperparameter_tuning(X, y, base_clf_name, vectorizer_name, multilabel_name, random_state=42):

    # split labeled data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state)
    # NOTE: further split into validation?

    # create supervised pipeline
    pipe = create_supervised_ml_pipeline(base_clf_name=base_clf_name, vectorizer_name=vectorizer_name, multilabel_name=multilabel_name)

    # fetch params
    param_dict = fetch_params_dict()
    param_grid = param_dict[base_clf_name]

    # define k_folds for grid search
    stratified_k_fold = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # Perform grid search with 3-fold stratified cross-validation
    grid_search = GridSearchCV(pipe, param_grid, cv=stratified_k_fold, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(X_train.processed_text, y_train)

    # Print the best set of parameters
    print("Best parameters: ", grid_search.best_params_)
    
    # Fit the model with the best parameters on the training set
    best_pipe = grid_search.best_estimator_
    best_pipe.fit(X_train.processed_text, y_train)

    # Evaluate performance on test set
    y_pred = best_pipe.predict(X_test.processed_text)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print("F1_weighted score on test set: ", f1_weighted)
    print("F1_micro score on test set: ", f1_micro)
    print("F1_macro score on test set: ", f1_macro)

    results = {"model": base_clf_name, "params": grid_search.best_params_}

    return results

    