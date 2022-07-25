import numpy as np
import pandas as pd
from processing_functions import grouping, feature_extract
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge



VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

# number of sample_per_patient is 12
SMPL_PER_PATIENT = 12

df_train_features = pd.read_csv('./train_features.csv')
df_train_labels = pd.read_csv('./train_labels.csv')
df_test_features = pd.read_csv('./test_features.csv')

def extract_features(data, sample_per_patient):
    out_data = []
    #define the features
    features = [np.nanmean, np.nanstd, np.isnan]
    for index in tqdm(range(int(data.shape[0] // sample_per_patient))):

        data_pid = data[sample_per_patient * index:sample_per_patient * (index + 1),:]
        descriptor = np.empty((len(features), data.shape[1]))

        for idx, feature in enumerate(features):

            #for mean and std features
            try:
                descriptor[idx] = feature(data_pid, axis=0)

            #include the number of nan of every measurement per patient over the 12 hours period as a feature
            except TypeError:
                descriptor[idx] = feature(data_pid).sum(axis = 0)

        out_data.append(descriptor.flatten())
    return np.array(out_data)



def main():

    #define the labels for the 3 subtasks
    Y_train_label = df_train_labels[TESTS].to_numpy()
    Y_train_stepsis = df_train_labels['LABEL_Sepsis'].to_numpy()
    Y_train_regression = df_train_labels[VITALS].to_numpy()


    #define the predictions destination dataframes
    predictions_task1 = pd.DataFrame(np.zeros((12664,10)),columns=TESTS)
    predictions_task2 = pd.DataFrame(np.zeros((12664,1)),columns=['LABEL_Sepsis'])
    predictions_task3 = pd.DataFrame(np.zeros((12664,4)),columns=VITALS)


    #extract the features from training data
    print(f'EXTRACTING TRAINING FEATURES...')
    X_features = extract_features(df_train_features.drop(['Time','pid'], axis=1).to_numpy(), SMPL_PER_PATIENT)
    #extract the features from test data
    print(f'EXTRACTING TESTING FEATURES...')
    X_features_test = extract_features(df_test_features.drop(['Time','pid'], axis=1).to_numpy(), SMPL_PER_PATIENT)

    #classification and regression model declaration
    model_clf = RandomForestClassifier()
    model_sub2 = RandomForestClassifier()
    model_regression = Ridge(alpha=10)

    #imputer declaration for missing values, both on training and testing data, using mean strategy
    model_imptr = SimpleImputer(missing_values=np.nan, strategy='mean')
    model_imptr.fit(X_features)
    X_T = model_imptr.transform(X_features)
    X_T_test = model_imptr.transform(X_features_test)

    print(f'testing features shape : {X_T_test.shape}')

    #fitting respecting models for all 3 substasks
    model_clf.fit(X_T, Y_train_label)
    model_sub2.fit(X_T, Y_train_stepsis)
    model_regression.fit(X_T, Y_train_regression)


    #make the predictions

    #TASK1
    predict_1_list = model_clf.predict_proba(X_T_test)

    for predict, label in zip(predict_1_list, TESTS):
        predictions_task1[label] = np.take(predict, 1, axis = 1)

    print(f'PREDICTION FOR TASK 1: {predictions_task1.head(5)}')


    #TASK2
    predict_2_list = model_sub2.predict_proba(X_T_test)

    predictions_task2['LABEL_Sepsis'] = np.take(predict_2_list, 1, axis = 1)

    print(f' PREDICTION FOR TASK 2: {predictions_task2.head(5)}')


    #TASK3
    predict_3_list = model_regression.predict(X_T_test)
    for idx, label in enumerate(VITALS): 
        predictions_task3[label] = predict_3_list[:, idx]

    print(f' PREDICTION FOR TASK 3: {predictions_task3.head(5)}')


    #join the predictions and output the predict dataframe in prediction_final.zip file
    print('JOINING THE PREDICTIONS')
    prediction_FINAL = pd.read_csv('sample.csv')
    for label in TESTS:
        prediction_FINAL[label] = predictions_task1[label]
    
    prediction_FINAL['LABEL_Sepsis'] = predictions_task2['LABEL_Sepsis']
    for label in VITALS:
        prediction_FINAL[label] = predictions_task3[label]


    prediction_FINAL.to_csv('prediction_Hugo.zip', index=False, float_format='%.3f', compression='zip')

    print('PREDICTION DONE, OUTPUT FILE READY')


    #cross validation for model validation
    # scores_sub1 = cross_val_score(model_clf, X_T, Y_train_label, cv=7, scoring='roc_auc',verbose=True)
    # scores_sub2 = cross_val_score(model_sub2, X_T, Y_train_stepsis, cv=7, scoring='roc_auc', verbose = True)
    # scores_sub3 = cross_val_score(model_regression, X_T, Y_train_stepsis, cv= 7, scoring= 'r2', verbose = True)

    # # print(f'scores for subtask1 cross-validation (7 splits): {scores_sub1}')
    # # print(f'scores for subtask2 cross-validation (7 splits): {scores_sub2}')
    # print(f'scores for subtask3 cross-validation (7 splits): {scores_sub3}')


if __name__ == '__main__':
    main()


