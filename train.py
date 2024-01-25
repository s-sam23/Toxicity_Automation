import argparse
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from imblearn.under_sampling import RandomUnderSampler
from utils import Utils
import joblib


class Train:
    
    def __init__(self,data,parser):
        self.data = data
        self.parser = parser
       

    
    def train_logistic_regression(self,X_train, y_train):
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        return classifier

    def train_random_forest(self,X_train, y_train):
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_gradient_boosting(self,X_train, y_train):
        classifier = GradientBoostingClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_adaboost(self,X_train, y_train):
        classifier = AdaBoostClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_decision_tree(self,X_train, y_train):
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_svc(self,X_train, y_train):
        classifier = SVC(probability=True)
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_svc_lr(self,X_train, y_train):
        classifier = SVC(probability=True,kernel='linear')
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_knn(self,X_train, y_train):
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_bernoulli_nb(self,X_train, y_train):
        classifier =BernoulliNB()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_mlp(self,X_train, y_train):
        classifier = MLPClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_bagging(self,X_train, y_train):
        classifier = BaggingClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_extra_trees(self,X_train, y_train):
        classifier = ExtraTreesClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_lgbmc(self,X_train, y_train):
        classifier = LGBMClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def train_xgbc(self,X_train, y_train):
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def fit_converter(self):
        
        df = self.data
        # print(df)
        
        util = Utils(df)
        
        m1 = util.FINGERPRINTS_MACCS() # Converting fingerprint of smiles from maccs 
        self.m1 = m1
        e1 = util.FINGERPRINTS_ECFP4() # Converting fingerprint of smiles from ecpf4
        self.e1 = e1
        r1 = util.RDKIT_FEATURES()     # Converting fingerprint of smiles from rdkitfeatures
        self.r1 = r1
        m1.to_csv(f'{self.parser.temp}/maccs.csv',index=False)
        e1.to_csv(f'{self.parser.temp}/ecfp4.csv',index=False)
        r1.to_csv(f'{self.parser.temp}/rdkit.csv',index=False)
        return m1 , r1 , e1 





    def maccs_trainer(self,m1):
        m1.drop(columns='SMILES',inplace=True)
        X = m1.iloc[:, :-1]  # Features (shape: 11000, 2216)
        y = m1.iloc[:, -1]  # Target
        

        rus  = RandomUnderSampler(replacement=False,random_state=42)
        x_rus,y_rus = rus.fit_resample(X,y)
    
        X_train, X_val, y_train, y_val = train_test_split(x_rus,y_rus,stratify=y_rus,random_state=97,test_size=.2)
        # Splitting Training and Testing Dataset
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        joblib.dump(self.train_logistic_regression(X_train,y_train),f'{self.parser.Models}/maccs_lr.pkl')
        joblib.dump(self.train_random_forest(X_train,y_train),f'{self.parser.Models}/maccs_rf.pkl')
        joblib.dump(self.train_gradient_boosting(X_train,y_train),f'{self.parser.Models}/maccs_gb.pkl')
        joblib.dump(self.train_adaboost(X_train,y_train),f'{self.parser.Models}/maccs_adb.pkl')
        joblib.dump(self.train_decision_tree(X_train,y_train),f'{self.parser.Models}/maccs_dt.pkl')
        joblib.dump(self.train_svc(X_train,y_train),f'{self.parser.Models}/maccs_svc.pkl')
        joblib.dump(self.train_svc_lr(X_train,y_train),f'{self.parser.Models}/maccs_svc_lr.pkl')
        joblib.dump(self.train_knn(X_train,y_train),f'{self.parser.Models}/maccs_knn.pkl')
        joblib.dump(self.train_mlp(X_train,y_train),f'{self.parser.Models}/maccs_lr.pkl')
        joblib.dump(self.train_bagging(X_train,y_train),f'{self.parser.Models}/maccs_bagging.pkl')
        joblib.dump(self.train_extra_trees(X_train,y_train),f'{self.parser.Models}/maccs_etc.pkl')
       
        joblib.dump(self.train_bernoulli_nb(X_train,y_train),f'{self.parser.Models}/maccs_bnb.pkl')
        joblib.dump(self.train_xgbc(X_train,y_train),f'{self.parser.Models}/maccs_xgbc.pkl')
        joblib.dump(self.train_lgbmc(X_train,y_train),f'{self.parser.Models}/maccs_lgbmc.pkl')
        return X_val,y_val



    def ecfp4_trainer(self,e1):
        e1.drop(columns='SMILES',inplace=True)
        X = e1.iloc[:, :-1]  # Features (shape: 11000, 2216)
        y = e1.iloc[:, -1]  # Target
        

        rus  = RandomUnderSampler(replacement=False,random_state=42)
        x_rus,y_rus = rus.fit_resample(X,y)
    
        X_train, X_val, y_train, y_val = train_test_split(x_rus,y_rus,stratify=y_rus,random_state=97,test_size=.2)
        # Splitting Training and Testing Dataset
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        joblib.dump(self.train_logistic_regression(X_train,y_train),f'{self.parser.Models}/ecfp4_lr.pkl')
        joblib.dump(self.train_random_forest(X_train,y_train),f'{self.parser.Models}/ecfp4_rf.pkl')
        joblib.dump(self.train_gradient_boosting(X_train,y_train),f'{self.parser.Models}/ecfp4_gb.pkl')
        joblib.dump(self.train_adaboost(X_train,y_train),f'{self.parser.Models}/ecfp4_adb.pkl')
        joblib.dump(self.train_decision_tree(X_train,y_train),f'{self.parser.Models}/ecfp4_dt.pkl')
        joblib.dump(self.train_svc(X_train,y_train),f'{self.parser.Models}/ecfp4_svc.pkl')
        joblib.dump(self.train_knn(X_train,y_train),f'{self.parser.Models}/ecfp4_knn.pkl')
        joblib.dump(self.train_mlp(X_train,y_train),f'{self.parser.Models}/ecfp4_lr.pkl')
        joblib.dump(self.train_bagging(X_train,y_train),f'{self.parser.Models}/ecfp4_bagging.pkl')
        joblib.dump(self.train_extra_trees(X_train,y_train),f'{self.parser.Models}/ecfp4_etc.pkl')
        joblib.dump(self.train_bernoulli_nb(X_train,y_train),f'{self.parser.Models}/ecfp4_bnb.pkl')
        joblib.dump(self.train_xgbc(X_train,y_train),f'{self.parser.Models}/ecfp4_xgbc.pkl')
        joblib.dump(self.train_lgbmc(X_train,y_train),f'{self.parser.Models}/ecfp4_lgbmc.pkl')
        
        return X_val,y_val
        
        
    def rdkit_trainer(self,r1):
        util = Utils(self.data)

        top_features = util.OP_RDKIT(r1) # Finding Best Features
        data = top_features.copy()
                # Separate features and target
        X = data.iloc[:, :-1]  # Features (shape: 11000, 2216)
        y = data.iloc[:, -1]  # Target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Assuming you have identified top_features using your model
        top_features_list = data.columns[:-1]



        rus  = RandomUnderSampler(replacement=False,random_state=42)
        x_rus,y_rus = rus.fit_resample(X,y)
    
        X_train, X_val, y_train, y_val = train_test_split(x_rus,y_rus,stratify=y_rus,random_state=97,test_size=.2)
        # Splitting Training and Testing Dataset
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
   
        joblib.dump(scaler,f'{self.parser.Models}/scaler.pkl')
        joblib.dump(top_features_list,f'{self.parser.Models}/rdkit_best_feat.pkl')
        
        joblib.dump(self.train_logistic_regression(X_train,y_train),f'{self.parser.Models}/logistic_regression.pkl')
        joblib.dump(self.train_random_forest(X_train,y_train),f'{self.parser.Models}/random_forest.pkl')
        joblib.dump(self.train_gradient_boosting(X_train,y_train),f'{self.parser.Models}/gradient_boosting.pkl')
        joblib.dump(self.train_adaboost(X_train,y_train),f'{self.parser.Models}/adaboost.pkl')
        joblib.dump(self.train_decision_tree(X_train,y_train),f'{self.parser.Models}/decision_tree.pkl')
        joblib.dump(self.train_svc(X_train,y_train),f'{self.parser.Models}/svc.pkl')
        joblib.dump(self.train_knn(X_train,y_train),f'{self.parser.Models}/knn.pkl')
        joblib.dump(self.train_mlp(X_train,y_train),f'{self.parser.Models}/mlp.pkl')
        joblib.dump(self.train_bagging(X_train,y_train),f'{self.parser.Models}/bagging.pkl')
        joblib.dump(self.train_extra_trees(X_train,y_train),f'{self.parser.Models}/etc.pkl')
        joblib.dump(self.train_bernoulli_nb(X_train,y_train),f'{self.parser.Models}/bnb.pkl')
        joblib.dump(self.train_xgbc(X_train,y_train),f'{self.parser.Models}/xgbc.pkl')
        joblib.dump(self.train_lgbmc(X_train,y_train),f'{self.parser.Models}/lgbmc.pkl')
      

        
        return X_val,y_val