# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import time
import inspect

class ParkinsonDiseaseKaggle:
    # Define a list with symptoms
    updrs_cols = ['updrs_1','updrs_2','updrs_3','updrs_4']

    def __init__(self):
        print('\nParkinson Disease Progression Predicition: Initialized\n')

    def load_training_datasets(self):
        print('\tStart loading training dataframes')
        # Get the name of the current function
        function_name = inspect.currentframe().f_code.co_name
        # Get the start time
        start_time = time.time()
                
        # Load train datasets
        self.train_proteins = pd.read_csv("./input/train_proteins.csv")
        self.train_clinical = pd.read_csv("./input/train_clinical_data.csv")
        self.train_peptides = pd.read_csv("./input/train_peptides.csv")
        self.supplemental_clinical = pd.read_csv("./input/supplemental_clinical_data.csv")
        
        # Get the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        # Print the function name and elapsed time
        print(f"\t{function_name} took {elapsed_time :.2f} seconds to run.")
        print('\tEnd loading training dataframes.\n')


    def prepare_data(self):
        print('\tStart preparing data might take a while!')
        # Get the name of the current function
        function_name = inspect.currentframe().f_code.co_name
        # Get the start time
        start_time = time.time()

        # Pivot the proteins and peptides tables so each protein and peptide is a feature
        train_proteins = self.train_proteins.pivot_table(values="NPX", index="visit_id", columns="UniProt")
        train_peptides = self.train_peptides.pivot_table(values="PeptideAbundance", index="visit_id", columns="Peptide")

        # Merge the three tables
        train = self.train_clinical.merge(train_proteins, on="visit_id", how="left")  \
                            .merge(train_peptides, on="visit_id", how="left")
                                    
        # Create a list of common columns
        train.drop('upd23b_clinical_state_on_medication', axis=1, inplace=True)
        train_cols = train.columns.values[6:]
        cols = list(set(train_cols))
                            
        # Set the patient id as index
        train = train.set_index(["patient_id"])

        # Drop the visit id column
        train = train.drop("visit_id", axis=1)
        
        # Fill updrs and state of medication values
        train.updrs_4 = train.updrs_4.fillna(0).round()

        # Create a list of the ids of the patients
        self.patient_id = list(train.index.unique())

        # Interpolate the missing data of every patient
        for patient in self.patient_id:
            train.loc[patient] = train.loc[patient].interpolate(method="linear").fillna(method="bfill")
            
        # Fill the remaining na values with the mean of the columns
        train = train.fillna(train.mean())

        # Normalize all the columns values from the train and the test dataset
        for col in cols:
            values_list = np.array(list(train[col]))
            train[col] = (train[col] - values_list.min()) / (values_list.max() - values_list.min())
        
        # Lets store the results in the class
        self.train = train

        # Get the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        # Print the function name and elapsed time
        print(f"\t{function_name} took {elapsed_time :.2f} seconds to run.")
        print('\tEnd preparing data!\n')


    def model_creation_and_fit(self):
        print('\tCreates models!')
        
        # Get the name of the current function
        function_name = inspect.currentframe().f_code.co_name
        # Get the start time
        start_time = time.time()
        
        # Empty dict for slopes
        updrs_slopes = {}

        # Slope and intercept for every patient and updrs
        for patient in self.patient_id:
            
            # Create empty dicts of lists
            updrs_slopes[patient] = []
            
            for updrs in self.updrs_cols:
                
                X = self.train.loc[patient]['visit_month'].values.reshape(-1, 1)
                y = self.train.loc[patient][updrs].values.reshape(-1, 1)
                lr = LinearRegression()
                lr.fit(X, y)
                updrs_slopes[patient].append(float(lr.coef_))

        # Create df for slopes
        slopes_df = pd.DataFrame.from_dict(updrs_slopes, orient="index", columns=[s + "_slope" for s in self.updrs_cols])

        # Merge all the data in the train data frame
        train = self.train.merge(slopes_df, how="left", left_index=True, right_index=True)

        # Create dictionaries to store the models
        self.updrs_predictors = {}
        self.updrs_slopes_predictors = {}

        # Define X
        X = train.iloc[:, 5:-4]

        print('\tCreates XGBoostRegressor models and train!')

        # Train a model per each symptom
        for updrs in self.updrs_cols:
            # self.updrs_predictors[updrs] = XGBRegressor(objective = self.smape)
            self.updrs_predictors[updrs] = XGBRegressor()
            self.updrs_predictors[updrs].fit(X, train.loc[:,updrs])
            
        # Train a model for each slope
        for updrs in self.updrs_cols:
            # self.updrs_slopes_predictors[updrs] = XGBRegressor(objective = self.smape)
            self.updrs_slopes_predictors[updrs] = XGBRegressor()
            self.updrs_slopes_predictors[updrs].fit(X, train.loc[:,(updrs + "_slope")])

        # Get the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        # Print the function name and elapsed time
        print(f"\t{function_name} took {elapsed_time :.2f} seconds to run.")
        print('\tEnds model creatrion and training! Models are ready :)\n')
        

    def generate_output(self):
        print('\tStarts generating output!')
        # Get the name of the current function
        function_name = inspect.currentframe().f_code.co_name
        # Get the start time
        start_time = time.time()

        # Load sample submission fie
        output_doc = pd.read_csv("./example_test_files/sample_submission.csv")
        output_doc = output_doc.drop(index=output_doc.index)

        for idx, patient in enumerate(self.patient_id_test):
            
            visit_id  = self.test_clinical.loc[patient].visit_id.values[idx+3]   # In order to match group_id of patient 50423 
            group_key = self.test_clinical.loc[patient].group_key.values[idx+3] # In order to match group_id of patient 50423 

            for updrs in self.updrs_cols:
                
                # Updrs values predicted with protein and peptides
                updrs_prediction = self.updrs_predictors[updrs].predict(self.test.loc[patient].values.reshape(1,-1))
                
                prediction_id = f'{visit_id}_{updrs}_plus_0_months'
                new_row = {'prediction_id': prediction_id, 'rating': int(updrs_prediction.round()), 'group_key': group_key}
                output_doc = output_doc.append(new_row, ignore_index=True)
                
                # To get the slope
                slope = self.updrs_slopes_predictors[updrs].predict(self.test.loc[patient].values.reshape(1,-1))
                        
                for month in range(6, 25, 6):
                    
                    #Compute the evolution at a given month
                    updrs_prediction =  (updrs_prediction + slope * month)
                    prediction_id = f'{visit_id}_{updrs}_plus_{month}_months'
                    new_row = {'prediction_id': prediction_id, 'rating': int(updrs_prediction.round()), 'group_key': group_key}
                    output_doc = output_doc.append(new_row, ignore_index=True)

            output_doc.to_csv('submission_2.csv',  index = False)
        
        # Get the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        # Print the function name and elapsed time
        print(f"\t{function_name} took {elapsed_time :.2f} seconds to run.")
        print('\tEnd of program! See output in submission.csv file.\n')


    def kaggle_data(self, test_proteins, test_peptides, test_clinical):
        print('\tStart preparing kaggle test data!')
        # Get the name of the current function
        function_name = inspect.currentframe().f_code.co_name
        
        # Get the start time
        start_time = time.time()
        
        # Pivot the proteins and peptides tables so each protein and peptide is a feature
        test_proteins = test_proteins.pivot_table(values="NPX", index="patient_id", columns="UniProt")
        test_peptides = test_peptides.pivot_table(values="PeptideAbundance", index="patient_id", columns="Peptide")

        # Set the patient id as index
        test_clinical = test_clinical.set_index(["patient_id"])

        # Merge the three tables
        test = test_proteins.merge(test_peptides, on="patient_id", how="left")
        
        # Create a list of common columns
        test_cols = test.columns.values
        cols = list(set(test_cols))

        # Use only common columns
        test  = test[cols]

        # Create a list of the ids of the patients
        self.patient_id_test = list(test.index.unique())

        # Interpolate the missing data of every patient
        for patient in self.patient_id_test:
            test.loc[patient] = test.loc[patient].interpolate(method="linear").fillna(method="bfill")

        # Fill the remaining na values with the mean of the columns
        test = test.fillna(test.mean())
        
        # Normalize all the columns values from the train and the test dataset
        for col in cols:
            values_list = np.array(list(test[col]))
            test[col] = (test[col] - values_list.min()) / (values_list.max() - values_list.min())
        
        # Lets store the results in the class
        self.test  = test
        self.test_clinical = test_clinical
        
        # Get the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        # Print the function name and elapsed time
        print(f"\t{function_name} took {elapsed_time :.2f} seconds to run.")
        print('\tEnd preparing kaggle data!\n')

    def make_prediction(self):
        print('\tStarts making prediction!')
        # Get the name of the current function
        function_name = inspect.currentframe().f_code.co_name
        # Get the start time
        start_time = time.time()

        for idx, patient in enumerate(self.patient_id_test):
            
            visit_id  = self.test_clinical.loc[patient].visit_id.values[idx+3]   # In order to match group_id of patient 50423 
            group_key = self.test_clinical.loc[patient].group_key.values[idx+3] # In order to match group_id of patient 50423 

            for updrs in self.updrs_cols:
                
                # Updrs values predicted with protein and peptides
                updrs_prediction = self.updrs_predictors[updrs].predict(self.test.loc[patient].values.reshape(1,-1))
                
                prediction_id = f'{visit_id}_{updrs}_plus_0_months'
                new_row = {'prediction_id': prediction_id, 'rating': int(updrs_prediction.round()), 'group_key': group_key}
                output_doc = output_doc.append(new_row, ignore_index=True)
                
                # To get the slope
                slope = self.updrs_slopes_predictors[updrs].predict(self.test.loc[patient].values.reshape(1,-1))
                        
                for month in range(6, 25, 6):
                    
                    #Compute the evolution at a given month
                    updrs_prediction =  (updrs_prediction + slope * month)
                    prediction_id = f'{visit_id}_{updrs}_plus_{month}_months'
                    new_row = {'prediction_id': prediction_id, 'rating': int(updrs_prediction.round()), 'group_key': group_key}
                    output_doc = output_doc.append(new_row, ignore_index=True)
        
        # Get the end time
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        
        # Print the function name and elapsed time
        print(f"\t{function_name} took {elapsed_time :.2f} seconds to run.")
        print('\tEnd of prediction!.\n')
        return output_doc

    def run_script(self):
        self.load_training_datasets()
        self.prepare_data()
        self.model_creation_and_fit()

    # AUXILIAR FUNCTIONS
    def smape(self, y_true, y_pred):
        """
        Computes the SMAPE between two arrays.
        """
        return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) * 2)


if __name__ == "__main__":
    # Get the start time
    start_time = time.time()
    proj = ParkinsonDisease()
    proj.run_script()

    # Get the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the function name and elapsed time
    print(f"\nThe whole programm took {elapsed_time :.2f} seconds to run.\n\n")

