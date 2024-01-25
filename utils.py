from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

class Utils:
    def __init__(self,df):
        self.df = df
    # Function to calculate MACCS fingerprints, handling potential errors
    def calculate_maccs(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                fp = AllChem.GetMACCSKeysFingerprint(mol)  # Use AllChem function
            except:
                fp = None  # Handle potential errors
            return fp  # Return the fingerprint as a bit vector
        else:
            return None  # Return None for invalid SMILES
        
    def FINGERPRINTS_MACCS(self):
        
        df = self.df 
        # Apply the function to the SMILES column and create a new DataFrame
        fingerprints_df = df.apply(lambda row: self.calculate_maccs(row["SMILES"]), axis=1)
        fingerprints_df = pd.DataFrame.from_records(fingerprints_df.tolist())
        
        c_list=['M_'+str(i) for i in fingerprints_df.columns]
        fingerprints_df.columns = c_list
    
        # Combine the original DataFrame with the fingerprints
        final_df = pd.concat([ fingerprints_df,df], axis=1)

        final_df.drop_duplicates(subset=final_df.columns[:-2],keep=False,inplace=True)
        final_df.reset_index(drop=True,inplace=True)
    
        # Print the final DataFrame
        return final_df

      # Function to calculate ECFP fingerprints, handling potential errors
      
    def calculate_ecfp(self,smiles, radius=3):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=True)
            return fp  # Return the fingerprint as a bit vector
        else:
            return None  # Return None for invalid SMILE
        
    def FINGERPRINTS_ECFP4(self):
        df = self.df
        # Apply the function to the SMILES column and create a new DataFrame
        fingerprints_df = df.apply(lambda row: self.calculate_ecfp(row["SMILES"]), axis=1)
        fingerprints_df = pd.DataFrame.from_records(fingerprints_df.tolist())
        
        c_list=['E_'+str(i) for i in fingerprints_df.columns]
        fingerprints_df.columns = c_list
    
        # Combine the original DataFrame with the fingerprints
        final_df = pd.concat([fingerprints_df,df], axis=1)
        final_df.drop_duplicates(subset=final_df.columns[:-2],keep=False,inplace=True)
        final_df.reset_index(drop=True,inplace=True)
    
        # Print the final DataFrame
        return final_df        
    
        # Function to calculate descriptors for a SMILES string
    def calculate_descriptors(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = Descriptors.CalcMolDescriptors(mol)
            return descriptors
        else:
            return None  # Handle invalid SMILES
        
    def RDKIT_FEATURES(self):
        
        df = self.df
        # Apply the function to the SMILES column and create a new DataFrame
        descriptors_df = df.apply(lambda row: self.calculate_descriptors(row["SMILES"]),axis=1)
        descriptors_df1 = pd.DataFrame.from_records(descriptors_df.tolist())
    
        # Combine the original DataFrame with the descriptors
        final_df = pd.concat([descriptors_df1,df], axis=1)

        final_df.drop_duplicates(subset=final_df.columns[:-2],keep=False,inplace=True)
        final_df.reset_index(drop=True,inplace=True)
    
        # Print the final DataFrame
        return final_df
    

    def rdkit_prediction_features(self):
        df = self.df
        # Apply the function to the SMILES column and create a new DataFrame
        descriptors_df = df.apply(lambda row: self.calculate_descriptors(row["SMILES"]),axis=1)
        descriptors_df1 = pd.DataFrame.from_records(descriptors_df.tolist())
    
        # Combine the original DataFrame with the descriptors
        final_df = pd.concat([descriptors_df1,df], axis=1)

        # final_df.drop_duplicates(subset=final_df.columns[:-2],keep=False,inplace=True)
        # final_df.reset_index(drop=True,inplace=True)
    
        # Print the final DataFrame
        return final_df

    
    def OP_RDKIT(self,rd):
        
        rd.drop('SMILES',axis=1,inplace=True)
        rd.drop_duplicates(inplace=True)
        rd.dropna(inplace=True)
        
        cf = rd.columns[rd.nunique()==1]
        rd.drop(cf,axis=1,inplace=True)
        
        train1 = rd.iloc[:,:-1]
        var_thr = VarianceThreshold(threshold = 0.25) #Removing both constant and quasi-constant
        var_thr.fit(train1)
        concol = [column for column in train1.columns if column not in train1.columns[var_thr.get_support()]]
        rd.drop(concol,axis=1,inplace=True)
        
        corr = rd.corr()
        upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
        rd.drop(to_drop,axis=1,inplace=True)
        
        rd.drop('Ipc',axis=1,inplace=True)

        return rd 
    
    


    
    