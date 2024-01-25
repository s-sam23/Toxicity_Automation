from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import glob

class ModelEvaluator:
    def __init__(self,x,y,parser):
    
        self.x = x #Input Testing data
        self.y = y # Testing Data
        self.parser = parser #Argument parser
        self.data = pd.DataFrame(columns=['Model_Name',"Accuracy", "Precision",'Recall', 'F1_Score', 'AUC_ROC'])

    def evaluate_binary_classification(self, y_true, y_pred, threshold=0.5):
        """
        Evaluate binary classification metrics.

        Parameters:
        - y_true: true labels
        - y_pred: predicted labels (probabilities or raw scores)
        - threshold: decision threshold for converting probabilities to binary labels

        Returns:
        - Dictionary containing various binary classification metrics
        """
        #print(y_pred)
        y_pred_binary = (y_pred >= threshold).astype(int)
       # print(y_pred_binary)

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred_binary),
            'Precision': precision_score(y_true, y_pred_binary),
            'Recall': recall_score(y_true, y_pred_binary),
            'F1_Score': f1_score(y_true, y_pred_binary),
            'AUC_ROC': roc_auc_score(y_true, y_pred)
        }

        return metrics

    # def plot_roc_curve(self, y_true, y_pred, label='Model'):
    #     """
    #     Plot the Receiver Operating Characteristic (ROC) curve.

    #     Parameters:
    #     - y_true: true labels
    #     - y_pred: predicted labels (probabilities or raw scores)
    #     - label: label for the ROC curve in the plot
    #     """
    #     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    #     auc = roc_auc_score(y_true, y_pred)

    #     plt.figure(figsize=(8, 8))
    #     plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})', linewidth=2)
    #     plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic (ROC) Curve')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()


    def Eval(self):
    
        results = self.data
        #print(type(results))
        paths = glob.glob(f'{self.parser.Models}/*')

        for v,i in enumerate(paths):
            if 'scaler' not in i and 'rdkit' not in i:
                model = joblib.load(i)

                y_true = self.y
                y_pred = model.predict(self.x)

                score = self.evaluate_binary_classification(y_true,y_pred)
                #print(i)

                score['Model_Name'] = i.split(f'{self.parser.Models}')[-1].split('.')[0]
                
                # new_row = {"Model Name": i.split('/')[-1][:-4], "AUC Score": score}
                #results = results.append(new_row, ignore_index=True)
                #results.concat([results, new_row], ignore_index=True)
                results = pd.concat([results, pd.DataFrame([score], index=[v])])       

             # results = results.sort_values(by="AUC Score", ascending=False).reset_index(drop=True)
            results.to_csv(f'{self.parser.Results}/Results.csv',index = False)
    
        return results
 



# Example usage:
# evaluator = ModelEvaluator()
# metrics = evaluator.evaluate_binary_classification(true_labels, predicted_probabilities)
# print(metrics)

# Example ROC curve plot:
# evaluator.plot_roc_curve(true_labels, predicted_probabilities, label='My Model')