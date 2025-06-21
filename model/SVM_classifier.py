from sklearn.svm import SVC
import optuna
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.results import ModelEvaluator

class SVMModel:
    def __init__(self, C=1.0, kernel='rbf', random_state=None):
        self.model = SVC(C=C, kernel=kernel, random_state=random_state, probability=True)

    def fit(self, X_train, y_train, X_val=None, y_val=None, use_optuna=False, n_trials=100):
        if use_optuna and X_val is not None and y_val is not None:
            # Compute class weights for imbalanced data
            classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))

            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True),
                    'degree': trial.suggest_int('degree', 2, 5),  # Only used for poly kernel
                    'coef0': trial.suggest_float('coef0', 0.0, 1.0),
                    'class_weight': class_weight_dict,
                    'random_state': 42,
                    'probability': True  # Needed for ROC-AUC
                }
                
                # Remove degree if not using poly kernel to avoid warnings
                if params['kernel'] != 'poly':
                    params.pop('degree')
                
                model = SVC(**params)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
                return scores.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_f1_score = study.best_value
            best_params['class_weight'] = class_weight_dict
            best_params['random_state'] = 42
            best_params['probability'] = True  # Ensure this is set

            self.model = SVC(**best_params)
            self.model.fit(X_train, y_train)
            return best_f1_score
            
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def score(self, X_test, y_test):
        evaluator = ModelEvaluator(self.model, "SVM")
        evaluator.evaluate(X_test, y_test)