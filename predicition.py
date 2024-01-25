import pandas as pd
import argparse
import glob
import joblib
from utils import Utils
from Standardizer import Standardizer_OP
from Parser import BaseOptions


parser = BaseOptions().initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args()

top_feat = joblib.load('Best_Model/rdkit_best_feat.pkl')
top_model = joblib.load('Best_Model/etc.pkl')
# scale_model = joblib.load('Best_Model/scale_model.pkl')

class ModelPredictor:
    def __init__(self,data,parser):
        self.data = data
        self.parser = parser #Argument parser
        self.dataframe = pd.DataFrame(columns=['SMILES','hERG_Prediction'])


    
    def rdkit_transfomer(self):
        
        df = self.data
        # print(df)
        util = Utils(df)
        rv = util.rdkit_prediction_features()
        return rv
        # m1 = util.FINGERPRINTS_MACCS() # Converting fingerprint of smiles from maccs 
        # self.m1 = m1
        # e1 = util.FINGERPRINTS_ECFP4() # Converting fingerprint of smiles from ecpf4
        # self.e1 = e1
        # r1 = util.RDKIT_FEATURES()     # Converting fingerprint of smiles from rdkitfeatures
        # self.r1 = r1
        # m1.to_csv(f'{self.parser.Validation_location}/v1_maccs.csv',index=False)
        # e1.to_csv(f'{self.parser.Validation_location}/v1_ecfp4.csv',index=False)
        # r1.to_csv(f'{self.parser.Validation_location}/v1_rdkit.csv',index=False)

        # return m1 , r1 , e1

    def rdkit_final_transform(self,rv):
         pred_results = self.dataframe

         self.rv = rv
         rv.dropna(inplace=True)
         pred_results['SMILES'] = rv['SMILES']

         x = rv[rv.columns[:-1]]

         
         X = x[top_feat]
        
         paths = glob.glob(f'{self.parser.Models}/*')

         for v,i in enumerate(paths):
            if 'scaler'  in i :
                scaler_model = joblib.load(i)
                X = scaler_model.transform(X)
                
         y_pred = top_model.predict(X)
         
         pred_results['hERG_Prediction'] = y_pred
         #pred_results.to_csv(f'{self.parser.Prediction_folder}/pred_results.csv',index=False)
         
         return pred_results

if __name__ == '__main__':

        ################## VALIDATION #####################


    pData = pd.read_csv(parser.Prediction)
    #Standard = Standardizer_OP(pData,'SMILES')
    #pclean_data = Standard.standardize_dataframe()
    #pclean_data.to_csv(f'{parser.Prediction_location}/predictiontion_1.csv',index=False)

    prediction  = ModelPredictor(pData,parser)
    rv  = prediction.rdkit_transfomer()

    res = prediction.rdkit_final_transform(rv)
    print(res)



 