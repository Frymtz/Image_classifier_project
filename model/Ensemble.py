import numpy as np
from scipy.stats import mode
from sklearn.metrics import f1_score

class HardVotingEnsemble:
    def __init__(self, models):
        """
        models: list of already trained models, each with a .predict() method
        """
        self.models = models

    def predict(self, X):
        """
        Makes predictions using hard voting (majority).
        """
        preds = [model.predict(X) for model in self.models]
        preds = np.stack(preds, axis=1)
        majority_vote, _ = mode(preds, axis=1)
        return majority_vote.ravel()

    def score(self, X, y_true, average='weighted'):
        """
        Calculates the F1-score of the ensemble.
        """
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred, average=average)