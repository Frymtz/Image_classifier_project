import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)
from sklearn.base import BaseEstimator


class SHVotingEnsemble(BaseEstimator):
    def __init__(self, models, mode):
        """
        models: list of already trained models, each with a .predict() method
        """
        self.models = models
        self.mode = mode

        results_dir = os.path.join(os.getcwd(), 'Results', 'Ensemble')
        os.makedirs(results_dir, exist_ok=True)
        self.cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        self.roc_path = os.path.join(results_dir, 'roc_curve.png')
        self.metrics_path = os.path.join(results_dir, 'metrics.txt')

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.predict_function(X)

    def predict_function(self, X):
        """
        Makes predictions using soft voting.
        """
        if self.mode == 'multi_inputs':
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
    
    def predict_proba(self, X):
        probas = []
        for i, model in enumerate(self.models):
            if hasattr(model, "predict_proba"):
                if self.mode == 'multi_inputs':
                    probas.append(model.predict_proba(X[i]))
                else:
                    probas.append(model.predict_proba(X))
            else:
                raise AttributeError(f"Model {type(model)} does not have predict_proba.")
        probas = np.stack(probas, axis=0)
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def score(self, X, y_true, average='macro'):
        """
        Calculates ensemble metrics and plots confusion matrix and ROC curve (if binary).
        """
        y_pred = self.predict_function(X)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        try:
            if hasattr(self, "predict_proba"):
                y_proba = self.predict_proba(X)
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
            else:
                roc_auc = None
        except Exception as e:
            roc_auc = None

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = range(len(set(y_true)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(self.cm_path)
        plt.close()

        # Save ROC curve (only for binary classification)
        if roc_auc is not None and len(set(y_true)) == 2:
            if y_proba.shape[1] == 2:
                RocCurveDisplay.from_predictions(y_true, y_proba[:, 1])
                plt.title('ROC Curve')
                plt.tight_layout()
                roc_path = os.path.join(self.roc_path)
                plt.savefig(roc_path)
                plt.close()

        # Write metrics to a txt file
        with open(self.metrics_path, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write('Confusion Matrix:\n')
            for row in cm:
                f.write(' '.join(map(str, row)) + '\n')