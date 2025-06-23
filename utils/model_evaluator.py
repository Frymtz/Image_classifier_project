import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, results_dir="Results/Inference"):
        self.model = model
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate(self, X_test, y_test, average='macro', prefix=''):
        y_pred = self.model.predict(X_test)
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)
        else:
            y_proba = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = None
        if y_proba is not None and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])

        # Save metrics
        metrics_path = os.path.join(self.results_dir, f"{prefix}metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write('Confusion Matrix:\n')
            for row in cm:
                f.write(' '.join(map(str, row)) + '\n')

        # Save confusion matrix plot
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_test)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{prefix}confusion_matrix.png"))
        plt.close()

        # Save ROC curve (binary only)
        if y_proba is not None and len(np.unique(y_test)) == 2:
            RocCurveDisplay.from_predictions(y_test, y_proba[:, 1])
            plt.title('ROC Curve')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{prefix}roc_curve.png"))
            plt.close()

        print(f"Evaluation complete. Metrics saved to {metrics_path}")
