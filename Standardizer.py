from rdkit import Chem
import pandas as pd
import os

class Standardizer_OP:

    def __init__(self, input_df, smiles_column):
        self.input_df = input_df
        self.smiles_column = smiles_column
        self.output_df = None

    def convert_to_canonical_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return canonical_smiles
        else:
            return None

    def standardize_dataframe(self):
        self.output_df = self.input_df.copy()
        self.output_df['Standardized_SMILES'] = self.output_df[self.smiles_column].apply(self.convert_to_canonical_smiles)
        self.output_df.drop('SMILES',axis=1,inplace=True)
        self.output_df.drop_duplicates(subset=['Standardized_SMILES'],keep=False,inplace=True)
        self.output_df.rename(columns={'Standardized_SMILES':'SMILES'},inplace=True)
        self.output_df = self.output_df[['SMILES','ACT']]
        self.output_df.reset_index(drop=True,inplace=True)

        return self.output_df  # Return the standardized DataFrame

    def save_to_csv(self, output_directory, output_filename):
        if self.output_df is None:
            print("Please call standardize_dataframe first.")
            return

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_path = os.path.join(output_directory, output_filename)
        self.output_df.to_csv(output_path, index=False)
        print(f"Dataframe saved to {output_path}")