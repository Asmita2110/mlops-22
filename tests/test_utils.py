import sys
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
sys.path.append('.')
from utils import get_all_h_params_comb, train_save_model, data_viz, preprocess_digits
from plot_digit_classification import h_param
from joblib import dump,load

def test_get_h_param_comb():
    gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list

    h_param_comb = get_all_h_params_comb(params)

    assert len(h_param_comb) == len(gamma_list)*len(c_list)

# train/dev/test split functionality : input 200 images, fraction is 70:15:15, then op should be have 140:30:30 sample in each set




# - some test cases that will validate if models are indeed getting saved or not.
# step1: train on small datset, provide path to save trained model
# step2: assert if a filr exist at the proived path
# step3: assert a file indeed a scikit learn model
# step4: optimally checksome validate the md5






def test_check_model_saving():
    model_path = 'svm_gamma_0.0001_C_5.joblib'
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    smaple_data = data[:500]
    smaple_label = label[:500]
    h_param_comb = h_param()
    actual_modle_path, clf = train_save_model(smaple_data, smaple_label, smaple_data, smaple_label, model_path, h_param_comb)

    assert actual_modle_path == model_path
    assert os.path.exists(model_path)

    loaded_model = load(model_path)

    assert type(loaded_model) == type(clf)



# what more test cases should be there 
# irrespective of the changes to the refactored code.



# preprocessing gives output that computable by model

# accuracy check, if acc(model) < threshold, then must not be pushed.

# what is possible: (model size in execution) < max_memory_you_support

# latency: fix model(input)  tok == linepassed < threshold
# this is dependend on execution environmrnt (as close to the run time environment)

# model variance -- 
# bais vs variance in ML ?
# std([model(train_1), model(train_2), ..., model(train_n)])
# threshold

#  data set verify, if it is as desired
#  dimensionality of the data -- 


#  verify output structure, say if you want output in sertain way
#  assert len(prediction_y) == len(test_y)

#  model persistance?
#  train the model -- check perfomance -- write model to disk
#  if the model load from the disk same as what we had written?

#  assert acc(loaded_model) == excepted_acc
#  assert prediction (loaded_model) == excepted_prediction


