import joblib 
import glob 
import argparse 
from utils import Utils
from Parser import BaseOptions
from Standardizer import Standardizer_OP
 

top_feat = joblib.load('Best_Model/rdkit_best_feat.pkl')
top_model = joblib.load('Best_Model/etc.pkl')
# scale_model = joblib.load('Best_Model/scale_model.pkl')


parser = BaseOptions().initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args()


# class validation_and_predictions:
#     def __init__(self,data,parser,top_feat,top_model,scaler_model):
#         self.scaler_model = scaler_model
#         self.top_model = top_model
#         self.top_feat = top_feat
#         self.data = data
#         self.parser = parser
    

#     def valid_fit_converter(self):
        
#         df = self.data
#         # print(df)
#         util = Utils(df)
        
#         m1 = util.FINGERPRINTS_MACCS() # Converting fingerprint of smiles from maccs 
#         self.m1 = m1
#         e1 = util.FINGERPRINTS_ECFP4() # Converting fingerprint of smiles from ecpf4
#         self.e1 = e1
#         r1 = util.RDKIT_FEATURES()     # Converting fingerprint of smiles from rdkitfeatures
#         self.r1 = r1
#         m1.to_csv(f'{self.parser.Validation_location}/maccs.csv',index=False)
#         e1.to_csv(f'{self.parser.Validation_location}/ecfp4.csv',index=False)
#         r1.to_csv(f'{self.parser.Validation_location}/rdkit.csv',index=False)

#         return m1 , r1 , e1 
    
    
#     def rdkit_final_transform(self,r1):
#          self.r1 = r1
          
#          x = r1.columns[:-2]
#          y = r1[r1.columns[-1]]
#          y_true = self.y
         
#          X = x[[top_feat]]
         
#          paths = glob.glob(f'{self.parser.Models}/*')

#          for v,i in enumerate(paths):
#             if 'scaler'  in i:
#                 scaler_model = joblib.load(i)
#                 X = scaler_model.transform(X)
                
#          y_pred = top_model.predict(self.x)
#          evaluation = ModelEvaluator()

#          score = evaluation.evaluate_binary_classification(y_true,y_pred)
#         #print(i) 
    
#          return score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import glob



class ModelValidator:
    def __init__(self,data,parser):
        self.data = data
        self.parser = parser #Argument parser

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
    
    def valid_fit_converter(self):
        
        df = self.data
        # print(df)
        util = Utils(df)
        
        m1 = util.FINGERPRINTS_MACCS() # Converting fingerprint of smiles from maccs 
        self.m1 = m1
        e1 = util.FINGERPRINTS_ECFP4() # Converting fingerprint of smiles from ecpf4
        self.e1 = e1
        r1 = util.RDKIT_FEATURES()     # Converting fingerprint of smiles from rdkitfeatures
        self.r1 = r1
        m1.to_csv(f'{self.parser.Validation_location}/v1_maccs.csv',index=False)
        e1.to_csv(f'{self.parser.Validation_location}/v1_ecfp4.csv',index=False)
        r1.to_csv(f'{self.parser.Validation_location}/v1_rdkit.csv',index=False)

        return m1 , r1 , e1 
    


    def rdkit_final_transform(self,r1):
         self.r1 = r1
         r1.dropna(inplace=True)

         x = r1[r1.columns[:-2]]
         y = r1[r1.columns[-1]]
         y_true = y
         
         X = x[top_feat]
         
         paths = glob.glob(f'{self.parser.Models}/*')

         for v,i in enumerate(paths):
            if 'scaler'  in i:
                scaler_model = joblib.load(i)
                X = scaler_model.transform(X)
                
         y_pred = top_model.predict(X)
          

         score = self.evaluate_binary_classification(y_true,y_pred)
         print(score) 
    
         return score
 


if __name__ == '__main__':

        ################## VALIDATION #####################


    vData = pd.read_csv(parser.Validation)
    Standard = Standardizer_OP(vData,'SMILES')
    vclean_data = Standard.standardize_dataframe()
    vclean_data.to_csv(f'{parser.Validation_location}/validation_1.csv',index=False)

    validation  = ModelValidator(vclean_data,parser)
    mv , rv , ev  = validation.valid_fit_converter()

    score = validation.rdkit_final_transform(rv)
    print(score)




    

    

     
 



 