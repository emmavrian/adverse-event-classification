from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def fetch_method_dict():

    # NOTE: svm does not work for semi-supervised since it does not implement predict_proba()
    # Multinomial and complement naive bayes also have issues in semi supervised since it can't handle the negative unlabeled values(?)

    base_clf_dict = {'logistic_regression': LogisticRegression(random_state=1), 
                     'multinomial_naive_bayes': MultinomialNB(), 
                     'complement_naive_bayes': ComplementNB(), 
                     'linear_svm': LinearSVC(random_state=1),
                     'random_forest': RandomForestClassifier(random_state=1),
                     'gradient_boosting': GradientBoostingClassifier(random_state=1),
                     'decision_tree': DecisionTreeClassifier(random_state=1)
                    }
    
    
    vectorizer_dict = {'count_vectorizer': CountVectorizer(ngram_range=(1,1)),
                       'tfidf_vectorizer': TfidfVectorizer()
                       }
    
    # can also add another dict for tuned classifiers if this is relevant
    
    return base_clf_dict, vectorizer_dict


def fetch_params_dict():
    params_dict = {

        'logistic_regression': [{'classifier__base_estimator__C': [0.1, 1.0, 10.0, 100.0],
                                'classifier__base_estimator__class_weight': [None, 'balanced'],
                                'classifier__base_estimator__max_iter': [100, 1000, 10000],
                                'classifier__base_estimator__penalty': ['l1', 'l2'],
                                #might have issues with these because of penalty
                                'classifier__base_estimator__solver': ['liblinear']
                                },
                                {'classifier__base_estimator__C': [0.1, 1.0, 10.0, 100.0],
                                'classifier__base_estimator__class_weight': [None, 'balanced'],
                                'classifier__base_estimator__max_iter': [100, 1000, 10000],
                                'classifier__base_estimator__penalty': ['l2'],
                                #might have issues with these because of penalty
                                'classifier__base_estimator__solver': ['newton-cg', 'lbfgs']
                                },
                                ],
        
        'multinomial_naive_bayes': {'classifier__base_estimator__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                                    'classifier__base_estimator__fit_prior': [True, False],
                                    'classifier__base_estimator__class_prior': [None, [0.2, 0.8], [0.3, 0.7]]
                                    },
        
        'complement_naive_bayes': {'classifier__base_estimator__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                                    'classifier__base_estimator__fit_prior': [True, False],
                                    'classifier__base_estimator__class_prior': [None, [0.2, 0.8], [0.3, 0.7]],
                                    'classifier__base_estimator__norm': [True, False]
                                    },

        'linear_svm': [{'classifier__base_estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0],
                       'classifier__base_estimator__loss': ['squared_hinge'],
                       'classifier__base_estimator__penalty': ['l1'],
                       'classifier__base_estimator__dual': [False],
                       'classifier__base_estimator__class_weight': [None, 'balanced'],
                       'classifier__base_estimator__max_iter': [10000]
                       },
                       {'classifier__base_estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0],
                       'classifier__base_estimator__loss': ['hinge', 'squared_hinge'],
                       'classifier__base_estimator__penalty': ['l2'],
                       'classifier__base_estimator__dual': [True],
                       'classifier__base_estimator__class_weight': [None, 'balanced'],
                       'classifier__base_estimator__max_iter': [10000]
                       }
                       ],

        'decision_tree': {'classifier__base_estimator__max_depth': [3, 5, 7, None],
                          'classifier__base_estimator__min_samples_split': [2, 5, 10],
                          'classifier__base_estimator__min_samples_leaf': [1, 2, 4],
                          'classifier__base_estimator__max_features': [None, 'sqrt', 'log2'],
                          'classifier__base_estimator__criterion': ['gini', 'entropy']
                          },

        'random_forest': {'classifier__base_estimator__n_estimators': [50, 100, 200, 300],
                          'classifier__base_estimator__max_depth': [5, 10, 15, 20, None],
                          'classifier__base_estimator__min_samples_split': [2, 5, 10],
                          'classifier__base_estimator__min_samples_leaf': [1, 2, 4],
                          'classifier__base_estimator__max_features': [None, 'sqrt', 'log2'],
                          'classifier__base_estimator__class_weight': [None, 'balanced']},

        'gradient_boosting': {'classifier__base_estimator__n_estimators': [50, 100, 200, 300],
                             'classifier__base_estimator__learning_rate': [0.05, 0.1, 0.2],
                             'classifier__base_estimator__max_depth': [5, 10, 15, 20, None],
                             'classifier__base_estimator__subsample': [0.5, 0.75, 1.0],
                             'classifier__base_estimator__max_features': [None, 'sqrt', 'log2']
                            },
    }

    return params_dict