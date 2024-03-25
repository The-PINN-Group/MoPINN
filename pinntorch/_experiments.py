from pinntorch._dependencies import * 
from pinntorch._training import train_model
from functools import partial
import jsonpickle
import time
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime
import h5py
import os

class Experiment(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate(self, seed):
        pass

class ExperimentManager:
    def __init__(self, num_repeats):
        self.repeats = num_repeats
        self.data = []
        self.parameters = {}
        
    def save_to_json(self, file_name):
        with open(file_name, 'w') as file:
            file.write(jsonpickle.encode([{"name" : exp_data["experiment"].name, "data" : exp_data["run_callbacks"]} for exp_data in self.data]))
    
    def set_parameter_distribution(self, name, distribution):
        self.parameters[name] = distribution
    
    def set_parameter(self, name, value):
        self.parameters[name] = lambda: value
    
    def add(self, experiment):
        self.data.append({"experiment" : experiment, "run_callbacks" : [], "epoch_times" : None})
        
    def run_experiments(self):
        progress_bar = tqdm(total=self.repeats * len(self.data), desc="Running Experiments", unit="rep", ncols=100)
        
        for _ in range(self.repeats):
            random_seed = random.randint(-2147483648, 2147483647)
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            for param_name, param_distribution in self.parameters.items():
                param_value = param_distribution()
                for dat in self.data:
                    setattr(dat["experiment"], param_name, param_value)
            
            for idx, dictionary in enumerate(self.data):
                model, loss_function, learning_rate, max_epochs, optimizer, callbacks = dictionary["experiment"].generate(random_seed)
                dictionary["run_callbacks"].append(callbacks)
                trained_model, mean_epoch_time, std_epoch_time = _train_model(model = model, loss_fn = loss_function, learning_rate = learning_rate, max_epochs = max_epochs, optimizer_fn = optimizer, epoch_callbacks = callbacks)
                dictionary["epoch_times"] = (mean_epoch_time, std_epoch_time)
                progress_bar.update(1) 
        progress_bar.close()
                
def _train_model(
    model: nn.Module,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
    optimizer_fn = torch.optim.Adam,
    epoch_callbacks : list = []
) -> nn.Module:
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    
    epoch_times = []
    
    for e_callback in epoch_callbacks:
        e_callback.prepare(max_epochs, model, loss_fn, optimizer_fn)
    
    for epoch in range(1, max_epochs+1):
        try:
            start_time = time.time()
            loss: torch.Tensor = loss_fn(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            
            for e_callback in epoch_callbacks:
                e_callback.process(epoch, model, loss_fn, optimizer_fn, loss)
        except KeyboardInterrupt:
            break
    
    epoch_times = np.array(epoch_times)
    mean_epoch_time = np.mean(epoch_times)
    std_epoch_time = np.std(epoch_times)
    
    return model, mean_epoch_time, std_epoch_time

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