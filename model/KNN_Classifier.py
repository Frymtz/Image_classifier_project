from sklearn.neighbors import KNeighborsClassifier
import optuna
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train, X_val=None, y_val=None, use_optuna=False, n_trials=50):
        if use_optuna and X_val is not None and y_val is not None:

            def objective(trial):
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                    'n_jobs': -1,
                }
                model = KNeighborsClassifier(**params)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_f1_score = study.best_value
            best_params['n_jobs'] = -1 
            self.model = KNeighborsClassifier(**best_params)
            self.model.fit(X_train, y_train)
            print(f"Best F1 SCORE: ", study.best_value)
            return best_f1_score
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        # ROC-AUC: only for binary or multilabel-indicator, handle multiclass with 'ovr'
        try:
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
            else:
                roc_auc = None
        except ValueError:
            roc_auc = None
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }