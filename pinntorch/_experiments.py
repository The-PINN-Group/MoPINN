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
        arr = np.array(lst)
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


#def store_list_as_group(f, name, list_data):
#    group = f.create_group(name)
#    for idx, arr in enumerate(list_data):
#        group.create_dataset(name=str(idx), data=arr)
#
#def load_list_from_group(f, name, number):
#    return [np.array(f[name][str(i)]) for i in range(number)]
#
#def create_run_folder(run_name):
#    today = datetime.now()
#    counter = 0
#    path = './data/'+str(today.strftime('%Y-%m-%d'))+'/'
#    folder = f"{run_name}_{counter:02}"
#    while os.path.exists(path+folder):
#        counter += 1
#        folder = f"{run_name}_{counter:02}"
#    os.makedirs(path+folder)
#    return path+folder
#
#def store_dictionary(f, dataset_name, dictionary):
#    f.create_dataset(name=dataset_name, data=dictionary)
#
#def save_results(run_name, settings, noisy_train_data, loss_data, loss_physics, lr_evolution, loss_val, models):
#    folder_path = create_run_folder(run_name)
#    
#    with h5py.File(os.path.join(folder_path, run_name+'.h5'), 'w') as fi:
#        settings_group = fi.create_group('settings')
#        for k, v in settings.items():
#            settings_group[k] = v
#        fi.create_dataset(name = 'noisy_data', data=noisy_train_data.detach())
#        store_list_as_group(fi, 'data_loss', loss_data)
#        store_list_as_group(fi, 'physics_loss', loss_physics)
#        store_list_as_group(fi, 'lr_evolution', lr_evolution)
#        if loss_val != None: 
#            store_list_as_group(fi, 'validation_loss', loss_val)
#    
#    for i, model in enumerate(models):
#        model_path = os.path.join(folder_path, f"model_{i}.pt")
#        torch.save(model.state_dict(), model_path)