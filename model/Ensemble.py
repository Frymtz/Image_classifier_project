import numpy as np
from scipy.stats import mode
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score

class HardVotingEnsemble:
    def __init__(self, models, mode):
        """
        models: list of already trained models, each with a .predict() method
        """
        self.models = models
        self.mode = mode

    def predict_function(self, X):
        """
        Makes predictions using hard voting (majority).
        """
        if self.mode == 'mult_processing':
            preds = []
            for i, model in enumerate(self.models):
                preds.append(model.predict(X[i]))
            preds = np.stack(preds, axis=1)
            majority_vote, _ = mode(preds, axis=1)
            return majority_vote.ravel()
        else:
            preds = [model.predict(X) for model in self.models]
            preds = np.stack(preds, axis=1)
            majority_vote, _ = mode(preds, axis=1)
            return majority_vote.ravel()

    def score(self, X, y_true, average='weighted'):
        """
        Calculates the F1-score of the ensemble.
        """
        y_pred = self.predict_function(X)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average)
        cm = confusion_matrix(y_true, y_pred)
        try:
            roc_auc = roc_auc_score(y_true, y_pred, average=average, multi_class='ovr')
        except Exception:
            roc_auc = None

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }