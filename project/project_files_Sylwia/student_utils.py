import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        reduce_dim_df: pandas dataframe, output dataframe with joined generic drug name
    '''
    reduce_dim_df = df.merge(ndc_df[['NDC_Code','Non-proprietary Name']], how='left', left_on='ndc_code', right_on='NDC_Code')
    reduce_dim_df.drop(columns=['NDC_Code'], inplace=True)
    reduce_dim_df.rename(columns={'Non-proprietary Name':'generic_drug_name'}, inplace=True)
    
    return reduce_dim_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.sort_values(['encounter_id'], ascending=True).groupby('patient_nbr').head(1)
     
    return first_encounter_df



#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    train_inds, validation_test_inds = next(GroupShuffleSplit(test_size=0.4, 
                                                     n_splits=1, 
                                                     random_state = 16).split(df, groups=df[patient_key]))
    
    train = df.loc[train_inds,:]
    validation_test = df.loc[validation_test_inds,:]
    validation_test.reset_index(inplace=True)
    
    validation_inds, test_inds = next(GroupShuffleSplit(test_size=0.5, 
                                                     n_splits=1, 
                                                     random_state = 16).split(validation_test, groups=validation_test[patient_key]))
    
    validation = validation_test.loc[validation_inds,:]
    test = validation_test.loc[test_inds,:]
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key=c, 
                                                                                                  vocabulary_file=vocab_file_path,
                                                                                                  num_oov_buckets=0)
        
        if c == 'primary_diagnosis_code':
            categorical_column = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=10)
            
        else:
            categorical_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
          
        output_tf_list.append(categorical_column)
    
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=default_value, dtype=tf.float64, 
                                                          normalizer_fn=normalizer)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    treshold = 5.0
    df['binary_pred'] = df[col] >= treshold
    student_binary_prediction = df['binary_pred'].replace(True, 1).replace(False, 0).to_numpy()
    
    return student_binary_prediction
