from decimal import Decimal
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# %%
def change_cols_types(df):
    cols = df.select_dtypes(include=np.number).columns.values
    for col in cols:
        df[col] = df[col].apply(Decimal).astype(str)
        if any('.' in s for s in df[col]):
            df[col] = df[col].astype('float')
        else:
            df[col] = df[col].astype('int')
    return df


# %% Functions to modeling
def prepare_data(df):
    # deal with categorical attributes
    tgt = df.iloc[:, -1]
    df = df[df.columns[:-1]]
    # too many levels
    uniques_per = df.select_dtypes(exclude=np.number).apply(lambda col: col.nunique() / len(df))
    uniques_max_per = uniques_per[uniques_per > 0.7]
    cols = df.columns[df.columns.isin(uniques_max_per.index)].values
    df_dummie = df.copy()
    if len(uniques_max_per) != 0:
        for col in cols:
            df_dummie[col] = OrdinalEncoder().fit_transform(df_dummie[[col]])
            # df[col] = df[col].astype('category').cat.codes

    # remove continuous attributes with many uniques
    df_aux = df_dummie.copy()
    df_aux = change_cols_types(df_aux)
    uniques_per_cont = df_aux.select_dtypes(incldude=np.int).apply(lambda col: col.nunique() / len(df))
    uniques_max_per_cont = uniques_per[uniques_per_cont > 0.9]
    cols_cont = df_aux.columns[df_aux.columns.isin(uniques_max_per_cont.index)].values
    print(cols_cont)
    if len(uniques_max_per) != 0:
        del df_dummie[cols_cont[0]]

    # one-hot encode the data using pandas get_dummies
    df_dummie = pd.get_dummies(df_dummie)
    # put target variable at the end
    df_dummie[tgt.name] = tgt.values

    # split into input and output elements
    df_val = df_dummie.values
    X, y = df_val[:, :-1], df_val[:, -1]
    # label encode the target variable to have the classes 0, 1 and 2
    y = LabelEncoder().fit_transform(y)

    return X, y


# evaluate a model
def evaluate_model(X, y, res):
    # split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)

    seed = np.random.seed(1234)
    # rfc = RandomForestClassifier(random_state=seed)
    rfc = RandomForestClassifier(random_state=seed)
    bc = BaggingClassifier(random_state=seed)
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=seed)
    lreg = LogisticRegression(random_state=seed)
    nnet = MLPClassifier(random_state=seed)

    n_feat = X_train.shape[1]

    # set parameters
    param_grid_rf = {
        'n_estimators': [100, 250, 500],
        'max_depth': [4, 6, 8, 10]
    }
    param_grid_bc = {
        'n_estimators': [100, 250, 500]
    }
    param_grid_xgb = {
        'n_estimators': [100, 250, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.1, 0.01, 0.001]
    }
    param_grid_lreg = {'C': np.logspace(-4, 4, 3),
                       'max_iter': [10000, 100000]
                       }
    param_grid_nnet = {'hidden_layer_sizes': [[n_feat], [n_feat // 2], [int(n_feat * (2 / 3))], [n_feat, n_feat // 2],
                                              [n_feat, int(n_feat * (2 / 3))], [n_feat // 2, int(n_feat * (2 / 3))],
                                              [n_feat, n_feat // 2, int(n_feat * (2 / 3))]
                                              ],
                       'alpha': [5e-3, 1e-3, 1e-4],
                       'max_iter': [10000, 100000]
                       }

    # define metric functions
    gmean = make_scorer(geometric_mean_score)
    scoring = {'gmean': gmean, 'acc': 'accuracy', 'bal_acc': 'balanced_accuracy',
               'f1': 'f1', 'f1_weighted': 'f1_weighted'}

    # create the parameter grid
    gs_rf = GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=5, scoring=scoring, refit='bal_acc',
                         return_train_score=True)
    gs_bc = GridSearchCV(estimator=bc, param_grid=param_grid_bc, cv=5, scoring=scoring, refit='bal_acc',
                         return_train_score=True)
    gs_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring=scoring, refit='bal_acc',
                          return_train_score=True)
    gs_lreg = GridSearchCV(estimator=lreg, param_grid=param_grid_lreg, cv=5, scoring=scoring, refit='bal_acc',
                           return_train_score=True)
    gs_nnet = GridSearchCV(estimator=nnet, param_grid=param_grid_nnet, cv=5, scoring=scoring, refit='bal_acc',
                           return_train_score=True)

    # List of pipelines for ease of iteration
    grids = [gs_rf, gs_bc, gs_xgb, gs_lreg, gs_nnet]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'Random Forest', 1: 'Bagging', 2: 'Boosting', 3: 'Logistic Regression', 4: 'Neural Network'}

    # Fit the grid search objects
    # print('Performing model optimizations...')

    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        # Performing cross validation to tune parameters for best model fit
        gs.fit(X_train, y_train)
        # Best params
        # print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Store results from grid search
        # res['cv_results_' + str(grid_dict[idx])] = pd.DataFrame.from_dict(scores.cv_results_)
        res['cv_results_' + str(grid_dict[idx])] = gs.cv_results_
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # print(confusion_matrix(y_test, y_pred))
        # Test data accuracy of model with best params
        print('Test set accuracy score for best params: %.3f ' % balanced_accuracy_score(y_test, y_pred))

    return res
