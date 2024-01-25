import os
import argparse
import shutil
from Parser import BaseOptions
import pandas as pd
from train import Train
from test import ModelEvaluator
#from validation import ModelValidator
from Standardizer import Standardizer_OP
#from predicition import ModelPredictor
 
from utils import Utils

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' already exists.")


def main_f(Data):
    # Parser for Parsing Arguments
    parser = BaseOptions().initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args()
    
    create_folder_if_not_exists(f'{parser.Models}')
    create_folder_if_not_exists(f'{parser.Results}')
    create_folder_if_not_exists(f'{parser.temp}')
    
    
    # Data = pd.read_csv(parser.data)

    #Data = pd.read_csv(parser.data)
    Standard = Standardizer_OP(Data,'SMILES')
    clean_data = Standard.standardize_dataframe()
    clean_data.to_csv(f'{parser.temp}/temp_1.csv',index=False)

    # print(clean_data,"\n================================")
    Training = Train(clean_data,parser)
    
    m1 , r1 , e1  = Training.fit_converter()

    X_val,y_val =Training.rdkit_trainer(r1)

    # Eva = Evaluation(X_val,y_val,parser)

    evaluator = ModelEvaluator(X_val,y_val,parser)

    metrics = evaluator.Eval()
    return (metrics)



if  __name__ == "__main__":
    
    # Parser for Parsing Arguments
    parser = BaseOptions().initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args()
    
    create_folder_if_not_exists(f'{parser.Models}')
    create_folder_if_not_exists(f'{parser.Results}')
    create_folder_if_not_exists(f'{parser.temp}')
    create_folder_if_not_exists(f'{parser.Validation_location}')
    create_folder_if_not_exists(f'{parser.Prediction_location}')

    #############  TRAINING #############

    Data = pd.read_csv(parser.data)
    Standard = Standardizer_OP(Data,'SMILES')
    clean_data = Standard.standardize_dataframe()
    clean_data.to_csv(f'{parser.temp}/temp_1.csv',index=False)


    Training = Train(clean_data,parser)

    m1 , r1 , e1  = Training.fit_converter()

    X_val,y_val =Training.rdkit_trainer(r1)



    ################ TESTING #######################


    evaluator = ModelEvaluator(X_val,y_val,parser)

    metrics = evaluator.Eval()



    ################## VALIDATION #####################


    # vData = pd.read_csv(parser.Validation)
    # Standard = Standardizer_OP(vData,'SMILES')
    # vclean_data = Standardizer_OP.standardize_dataframe()
    # vclean_data.to_csv(f'{parser.Validation_location}/validation_1.csv',index=False)

    # validation  = ModelValidator(vclean_data,parser)
    # mv , rv , ev  = validation.fit_converter()

    # score = validation.rdkit_final_transform(rv)



    # ################### PREDICTION #####################


    # pred_data = pd.read_csv(parser.Prediction)

    # pred = ModelPredictor(pred_data,parser)

    # rv  = pred.rdkit_transformer()
    # pred_results = pred.rdkit_final_transform(rv)
    # print(pred_results)

    ###########################################################################################################################
