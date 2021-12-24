import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os


BASEDIR = "C:/JupyterNotebook/PROJECT/dm-api/"
os.chdir(BASEDIR)
def data2df(data):
    '''
    input data type [list]
    '''


    AM_cols = ['T_TAKAM', 'T_RICEAM', 'T_WINEAM', 'T_SOJUAM', 'T_BEERAM', 'T_HLIQAM']
    FQ_cols = ['T_TAKFQ', 'T_RICEFQ', 'T_WINEFQ', 'T_SOJUFQ', 'T_BEERFQ', 'T_HLIQFQ']
    col_names = ['T_AGE', 'T_INCOME', 'T_MARRY', 'T_HEIGHT', 'T_WEIGHT', 'T_BMI', 'T_DRINK', 'T_DRDU', 'T_TAKFQ', 'T_TAKAM', 'T_RICEFQ',
       'T_RICEAM', 'T_WINEFQ', 'T_WINEAM', 'T_SOJUFQ', 'T_SOJUAM', 'T_BEERFQ',
       'T_BEERAM', 'T_HLIQFQ', 'T_HLIQAM', 'T_SMOKE', 'T_SMDUYR', 'T_SMDUMO',
       'T_SMAM', 'T_PSM', 'T_EXER']
    input_df = pd.DataFrame(np.array(data).reshape(1,-1),columns=col_names)

    input_df["AM_sum"] = input_df[AM_cols].apply(lambda x : sum(x), axis = 1)
    input_df["FQ_sum"] = input_df[FQ_cols].apply(lambda x : sum(x), axis = 1)
    input_df["AM_mean"] = input_df[AM_cols].apply(lambda x : np.mean(x),axis=1)
    input_df["FQ_mean"] = input_df[FQ_cols].apply(lambda x : np.mean(x),axis=1)
    
    return input_df
    
def preprocessing_OHE(df_imputed):

    ohe_list = ['T_MARRY', 'T_PSM','T_EXER']
    LOADDIR = "models/"

    for df_col in df_imputed.columns: 
        for ohe_col in ohe_list:
            if df_col == ohe_col:
                ohe_scaler = pickle.load(open(BASEDIR+LOADDIR+str(df_col) + '_OneHotEncoder.pickle', 'rb'))
                ohe_values = ohe_scaler.transform(df_imputed[df_col].values.reshape(-1,1))
                ohe_df = pd.DataFrame(ohe_values,columns=ohe_scaler.get_feature_names(["_"+df_col]))
                df_imputed.drop(columns=df_col,inplace=True)
                df_imputed = pd.concat([df_imputed,ohe_df],axis=1)

    AM_cols = ['T_TAKAM', 'T_RICEAM', 'T_WINEAM', 'T_SOJUAM', 'T_BEERAM', 'T_HLIQAM']
    FQ_cols = ['T_TAKFQ', 'T_RICEFQ', 'T_WINEFQ', 'T_SOJUFQ', 'T_BEERFQ', 'T_HLIQFQ']
    df_imputed.drop(columns = AM_cols, inplace=True)
    df_imputed.drop(columns = FQ_cols,inplace= True)

    return df_imputed

def preprocessing_Robust(df_imputed):

    LOADDIR = "models/"
    with open(BASEDIR+LOADDIR+'RobustScaler.pickle', 'rb') as fr:
        robust_scaler = pickle.load(fr)
        robust_df = robust_scaler.transform(df_imputed)

    return robust_df

def preprocessing(data):
    df = data2df(data)
    ohe_df = preprocessing_OHE(df)
    preprocessed_df = preprocessing_Robust(ohe_df)
    return preprocessed_df