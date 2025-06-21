from sklearn.neighbors import KNeighborsClassifier
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.results import ModelEvaluator

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train, X_val=None, y_val=None, use_optuna=False, n_trials=100):
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
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_f1_score = study.best_value
            best_params['n_jobs'] = -1 
            
            self.model = KNeighborsClassifier(**best_params)
            self.model.fit(X_train, y_train)
            return best_f1_score
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def score(self, X_test, y_test):
       evaluator = ModelEvaluator(self.model, "KNN")
       evaluator.evaluate(X_test, y_test)