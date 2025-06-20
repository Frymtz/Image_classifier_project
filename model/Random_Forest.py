from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.results import ModelEvaluator

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None, use_optuna=False, n_trials=100):
        if use_optuna and X_val is not None and y_val is not None:

            # Compute class weights for imbalanced data
            classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 2, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
                    'n_jobs': -1,
                    'class_weight': class_weight_dict,
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
                return scores.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_f1_score = study.best_value
            best_params['class_weight'] = class_weight_dict
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1 
            self.model = RandomForestClassifier(**best_params)
            self.model.fit(X_train, y_train)
            return best_f1_score
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def score(self, X_test, y_test):
        evaluator = ModelEvaluator(self.model, "RandomForest")
        evaluator.evaluate(X_test, y_test)