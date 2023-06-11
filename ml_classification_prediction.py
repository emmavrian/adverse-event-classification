from ml_classification_base_methods import create_supervised_ml_pipeline, create_semi_supervised_ml_pipeline
import pandas as pd



def supervised_ml_classification_label_prediction(X_train, y_train, X_predict, base_clf_name, vectorizer_name, multilabel_name):
    
    # create supervised pipeline
    pipe = create_supervised_ml_pipeline(base_clf_name=base_clf_name, vectorizer_name=vectorizer_name, multilabel_name=multilabel_name)

    # train/fit the pipeline on the labeled data X_train and y_train (full labeled dataset, no test set)
    pipe.fit(X_train.processed_text, y_train)

    # create predicted labels for X_predict based on trained pipeline, to be further validated by a nurse
    y_pred = pipe.predict(X_predict.processed_text)

    # create a new dataframe with the text and the predicted labels
    text_with_predicted_labels = X_predict.copy()
    text_with_predicted_labels['venous_catheter'] = y_pred[:, 0]  # add venous catheter labels
    text_with_predicted_labels['infection'] = y_pred[:, 1]  # add infection labels

    # add a new column with textual representation of the labels
    text_with_predicted_labels['labels'] = text_with_predicted_labels.apply(lambda row: ', '.join([col for col in ['venous_catheter', 'infection'] if row[col] == 1]), axis=1)

    return text_with_predicted_labels


def semi_supervised_ml_classification_label_prediction(X_train, y_train, X_train_unlab, X_predict, base_clf_name, vectorizer_name, multilabel_name, threshold=0.7):
    
    # create semi-supervised pipeline
    pipe = create_semi_supervised_ml_pipeline(base_clf_name=base_clf_name, vectorizer_name=vectorizer_name, multilabel_name=multilabel_name, threshold=threshold)

    # create the training dataset for semi supervised (both labeled and unlabeled)
    X_train_mixed = pd.concat([X_train, X_train_unlab])
    y_train_nolabel = pd.DataFrame(index=X_train_unlab.index, columns=['venous_catheter', 'infection'])
    y_train_nolabel['venous_catheter'] = -1
    y_train_nolabel['infection'] = -1
    # recombine training dataset labels
    y_train_mixed = pd.concat([y_train, y_train_nolabel])

    print("\nThreshold: ", threshold)

    # train/fit the pipeline on the labeled data X_train_mixed and y_train_mixed (full labeled dataset+unlabeled train set, no test set)
    pipe.fit(X_train_mixed.processed_text, y_train_mixed)

    # create predicted labels for X_predict based on trained pipeline, to be further validated by a nurse
    y_pred = pipe.predict(X_predict.processed_text)

    # create a new dataframe with the text and the predicted labels
    text_with_predicted_labels = X_predict.copy()
    text_with_predicted_labels['venous_catheter'] = y_pred[:, 0]  # add venous catheter labels
    text_with_predicted_labels['infection'] = y_pred[:, 1]  # add infection labels

    # add a new column with textual representation of the labels
    text_with_predicted_labels['labels'] = text_with_predicted_labels.apply(lambda row: ', '.join([col for col in ['venous_catheter', 'infection'] if row[col] == 1]), axis=1)

    return text_with_predicted_labels
