import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)

class ModelEvaluator:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

        results_dir = os.path.join(os.getcwd(), 'Results', model_name)
        os.makedirs(results_dir, exist_ok=True)
        self.cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        self.roc_path = os.path.join(results_dir, 'roc_curve.png')
        self.metrics_path = os.path.join(results_dir, 'metrics.txt')

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        try:
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(X_test)
                if y_score.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_score[:, 1], average='macro')
                else:
                    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
            else:
                roc_auc = None
        except Exception:
            roc_auc = None

        # Salvar matriz de confusão
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = range(len(set(y_test)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(self.cm_path)
        plt.close()

        # Salvar curva ROC (apenas para classificação binária)
        if roc_auc is not None and len(set(y_test)) == 2:
            RocCurveDisplay.from_estimator(self.model, X_test, y_test)
            plt.title('ROC Curve')
            plt.tight_layout()
            plt.savefig(self.roc_path)
            plt.close()

        # Escrever as métricas em um arquivo txt
        with open(self.metrics_path, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')
            f.write(f'F1 Score: {f1}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write('Confusion Matrix:\n')
            for row in cm:
                f.write(' '.join(map(str, row)) + '\n')
