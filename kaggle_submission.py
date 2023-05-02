import public_timeseries_testing_util
from ParkinsonDiseaseKaggle import ParkinsonDiseaseKaggle 
import numpy as np

env = public_timeseries_testing_util.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
pd = ParkinsonDiseaseKaggle()  # Loads main code class
pd.run_script()                # Train our class with the training data and prepares  

for (test, test_peptides, test_proteins, sample_submission) in iter_test:    
    pd.kaggle_data(test_proteins, test_peptides, test)
    prediction = pd.make_prediction()
    # sample_prediction_df['rating'] = np.arange(len(sample_prediction))  # make your predictions here
    env.predict(prediction)   # register your predictions


    