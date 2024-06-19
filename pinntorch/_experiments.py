from pinntorch._dependencies import * 
from datetime import datetime
import h5py
import os

def create_run_folder(run_name):
    today = datetime.now()
    counter = 0
    path = './data/'+str(today.strftime('%Y-%m-%d'))+'/'
    folder = f"{run_name}_{counter:02}"
    while os.path.exists(path+folder):
        counter += 1
        folder = f"{run_name}_{counter:02}"
    os.makedirs(path+folder)
    return path+folder

def is_simple_list(lst):
    return all(isinstance(item, (int, float, str, bool)) for item in lst)

def is_homogeneous_nested_list(lst):
    try:
        np.array(lst)
        return True
    except ValueError:
        return False

def _store_element(f, element_key, element):
    if type(element)==dict:
        dict_group = f.create_group(element_key)
        for key in element:
            _store_element(dict_group, key, element[key])
    elif type(element) == list:
        if is_simple_list(element) or is_homogeneous_nested_list(element):
            f.create_dataset(element_key, data=element)
        else:
            list_group = f.create_group(element_key)
            for i, item in enumerate(element):
                item_key = str(i)
                _store_element(list_group, item_key, item)
    else:
        f.create_dataset(name=element_key, data=element)
        
def save_dictionary(path, filename, dictionary):
    with h5py.File(os.path.join(path,filename+'.h5'), 'w') as f:
        for key in dictionary:
            print(key)
            _store_element(f, key, dictionary[key])

def save_models(path, models):
    for i, model in enumerate(models):
        model_path = os.path.join(path, f"model_{i}.pt")
        torch.save(model.state_dict(), model_path)

def save_list_of_model_states(path, model_states):
    for i, model_state in enumerate(model_states):
        model_path = os.path.join(path, f"model_{i}.pt")
        torch.save(model_state, model_path)