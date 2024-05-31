from pinntorch._dependencies import *
from pinntorch._model import *
from pinntorch._moo_weight_methods import *
from abc import ABC, abstractmethod
import warnings
import copy
import time
import math
import wandb
from torch.optim.lr_scheduler import _LRScheduler

class ReduceLROnSpike():
  """
  Custom learning rate scheduler that reduces learning rate on increasing loss.

  Args:
      optimizer: PyTorch optimizer object.
      factor: Factor by which to decrease learning rate on increasing loss (default: 0.1).
  """
  def __init__(self, optimizer, factor=0.1):
    self.optimizer=optimizer
    self.factor = factor
    self.prev_loss = float('inf')  # Initialize with a high value

  def step(self, metrics):
    """
    Updates learning rate based on comparison with previous loss.
    """
    val_loss = float(metrics)

    in_spike = False

    # Update learning rate based on loss comparison
    if val_loss > self.prev_loss:
        if not in_spike:
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = old_lr * self.factor
                param_group['lr'] = new_lr
        in_spike = True
    else:
        in_spike = False
    # Update previous loss for next comparison
    self.prev_loss = val_loss


class EpochCallBack(ABC):
    """
    Abstract base class for epoch callback objects.
    """

    def prepare(self, max_epochs, model, loss_fn, optimizer):
        """
        Perform a specific action for preperation before training
        """
        pass

    def initiated_copy(self):
        """
        Resets this callback_object to let it process with the same initialized values
        """
        pass

    @abstractmethod
    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        """
        Perform a specific action at the end of each epoch during training.
        """
        pass

    def finish(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        """
        Perform a specific action after training.
        """
        pass

#def descent_direction(model):
#    params = model.parameters()
#    return torch.cat([param.data.view(-1) for param in params])

class AllDataMonitor(EpochCallBack):
    """
    Abstract base class for epoch callback objects.
    """
    def __init__(self, data_loss_fn, physics_loss_fn, validation_loss_fn):
        self.data_loss_fn = data_loss_fn
        self.physics_loss_fn = physics_loss_fn
        self.val_loss_fn = validation_loss_fn

    def prepare(self, max_epochs, model, loss_fn, optimizer):
        self.val_history = []
        self.lr_history = []
        self.data_history = []
        self.physics_history = []

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        self.lr_history.append(float(optimizer.param_groups[0]["lr"]))
        self.val_history.append(self.val_loss_fn(model).detach().numpy())
        self.data_history.append(self.data_loss_fn(model).detach().numpy())
        self.physics_history.append(self.physics_loss_fn(model).detach().numpy())


def train_model(
    model: nn.Module,
    loss_fn: Callable,
    mo_method: Callable = None,
    optimizer_fn=torch.optim.Adam,
    max_epochs: int = 1_000,
    live_logging: bool = True,
    log_interval: int = 1_000,
    lr_decay = 0.0,
    parameter_groups: dict = None,
    epoch_callbacks: list = [],
) -> nn.Module:

    # define the trainable parameters
    trainable_parameters = model.parameters()
    if parameter_groups is not None:
        trainable_parameters = parameter_groups

    # declare the optimizer
    optimizer = optimizer_fn(params=trainable_parameters)
    
    # setup learning rate scheduler
    if lr_decay > 0.0:
        lr_scheduler = ReduceLROnSpike(optimizer=optimizer, factor=1-lr_decay)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=1-lr_decay, patience=5, threshold=1e-8, cooldown=5)
    
    # prepare all callbacks
    for e_callback in epoch_callbacks:
        e_callback.prepare(max_epochs, model, loss_fn, optimizer)
    
    # start training
    start_time = time.time()
    for epoch in range(0, max_epochs):
        try:
            extra_logs = dict()
            # calculate all losses, stack them in a tensor ([l_0, l_1, l_2, ...]) if there are multiple losses, ([l_0]) for only one loss
            losses = _calculate_and_stack_loss(loss_fn, model)
            optimizer.zero_grad()

            # loss is undefined yet, variable scope reason
            loss = None
            # BACKWARD
            if mo_method is None:
                # single objective
                if losses.shape[0] > 1 and epoch < 1:
                    warnings.warn("No multi-objective optimization method (mo_method) was set, despite defining multiple losses. Training will continue with the sum of the losses!", RuntimeWarning)
                loss = sum(losses)
                loss.backward(retain_graph=True)
            else:
                # multi-objective
                loss, mo_dict = mo_method.backward(
                    losses=losses, shared_parameters=list(model.parameters())
                )
                # store multi-objective extra informations in an extra log
                if mo_dict is not None:
                    for key, value in mo_dict.items():
                        extra_logs[key] = value
            
            #print(descent_direction(model))

            # learning optimizer step
            optimizer.step()

            # learning rate scheduler step
            if lr_decay > 0.0 and loss is not None:
                lr_scheduler.step(loss)
            
            # log some information live
            if live_logging:
                if epoch % log_interval == 0 and epoch > 0:
                    if loss is None:
                        loss = sum(losses)
                    print(
                        f"epoch: {epoch}\tlog10(loss sum): {math.log10(float(loss)):.4f}\texecution time (past 1k epochs): {(time.time() - start_time):.4f}"
                    )
                    start_time = time.time()
            
            # process all callbacks
            for e_callback in epoch_callbacks:
                e_callback.process(
                    epoch, model, loss_fn, optimizer, losses, extra_logs
                )
        # check if training is interrupted
        except KeyboardInterrupt:
            break

    #  finish all callbacks
    for e_callback in epoch_callbacks:
        e_callback.finish(epoch, model, loss_fn, optimizer, losses, extra_logs)

    return model

def _get_loss_value(loss):
    #print('loss in get loss value ', loss)
    loss_value = [l.detach().cpu().numpy() for l in loss]
    #print('detach in list ', loss_value)
    loss_value = np.array(loss_value)
    #print('finished loss value ', loss_value, type(loss_value))
    return loss_value

def _calculate_and_stack_loss(loss_fun, model):
    #print('loss fun', loss_fun)
    losses = []
    if type(loss_fun) is not list:
        loss_fun = [loss_fun]
    for g in loss_fun:
        result = g(model)
        #print(g, '\tresult:\t', result)
        if isinstance(result, tuple):
            losses.extend(result)
        elif isinstance(result, dict):
            losses.extend(result.values())
        else:
            losses.append(result)
    return torch.stack([torch.squeeze(l) for l in losses])

def _gather_loss_labels(loss_fun, model):
    #print('loss fun', loss_fun)
    loss_labels = []
    if type(loss_fun) is not list:
        loss_fun = [loss_fun]
    for g in loss_fun:
        result = g(model)
        #print(g, '\tresult\t', result)
        #print(type(result))
        if type(result) is dict:
            result = list(result.keys())
        else:
            if type(result) is tuple:
                result_len = len(result)
            elif len(result.shape) == 0:
                result_len = 1
            else:
                result_len = len(result)
            result = [f'loss_{index}' for index in range(len(loss_labels), len(loss_labels)+result_len)]
        for llabel in enumerate(result):
            if llabel in loss_labels:
                warnings.warn('two losses have the same label \"'+str(llabel)+'\"')
        loss_labels.extend(result)
    return loss_labels


class BestModelSaver(EpochCallBack):
    """
    A class saves the best model state dict, according to the train loss.
    """

    def __init__(self):
        self.best_state_dict = None
        self.epoch_of_best = None
        self.lowest_train_loss = None

    def initiated_copy(self):
        return BestModelSaver()

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        loss_value = _get_loss_value(current_loss)
        if self.lowest_train_loss is None or self.lowest_train_loss > loss_value:
            self.lowest_train_loss = loss_value
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.epoch_of_best = epoch


class SolutionModelMonitor(EpochCallBack):
    def __init__(self, save_interval=1):
        self.save_interval = save_interval

    def prepare(self, max_epochs, model, loss_fn, optimizer):
        self.model_states = []

    def initiated_copy(self):
        return SolutionModelMonitor(self.save_interval)
    
    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        if epoch % self.save_interval == 0:
            self.model_states.append(copy.deepcopy(model.state_dict()))
        

class SolutionPlotMonitor(EpochCallBack):
    """
    A class saves the predicted output of the current solution function at specified plot points.
    """

    def __init__(self, plot_points, train_points=None, plot_interval=1):
        self._plot_points = plot_points
        self._train_points = train_points
        self.plot_interval = plot_interval
        self.plot_points = plot_points.detach().cpu().numpy()
        self.train_points = None
        if train_points is not None:
            self.train_points = train_points.detach().cpu().numpy()
        self.solution_plot = []
        self.solution_train = None
        if train_points is not None:
            self.solution_train = []

    def initiated_copy(self):
        return SolutionPlotMonitor(
            self._plot_points, self._train_points, store_every=self.plot_interval
        )

    def prepare(self, max_epochs, model, loss_fn, optimizer):
        self._append_solution(model, max_epochs)

    @staticmethod
    def _detach_and_numpy(list_of_tensor):
        return [tensor.detach().cpu().numpy() for tensor in list_of_tensor]

    def _append_solution(self, model, epoch):
        if self._train_points is not None:
            f_final_training = f(model, self._train_points, of="all")
            if type(f_final_training) is not list:
                f_final_training = [f_final_training]
            self.solution_train.append(
                SolutionPlotMonitor._detach_and_numpy(f_final_training)
            )
        f_final = f(model, self._plot_points, of="all")
        if type(f_final) is not list:
            f_final = [f_final]
        self.solution_plot.append(SolutionPlotMonitor._detach_and_numpy(f_final))

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        if epoch % self.plot_interval == 0:
            self._append_solution(model, epoch)


class TrainLossMonitor(EpochCallBack):
    """
    A class that monitors the model's training loss for every epoch.
    """
    def __init__(self):
        self.loss_history = []
        self.loss_labels = []

    def initiated_copy(self):
        return TrainLossMonitor()

    def prepare(self, max_epochs, model, loss_fn, optimizer):
        self.loss_labels = _gather_loss_labels(loss_fn, model)
        #print('train loss labels', self.loss_labels)
        self.loss_history = np.zeros((max_epochs, len(self.loss_labels)))

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        loss = _get_loss_value(current_loss)
        self.loss_history[epoch] = loss

    def finish(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        self.loss_history = np.transpose(self.loss_history)


class CheckpointMonitor(EpochCallBack):
    def __init__(self, log_period=1000):
        self.log_period = log_period

    def initiated_copy(self):
        return CheckpointMonitor(self.log_period)

    def save_checkpoint(self, epoch, model, loss_fn, optimizer, current_loss):
        loss = _get_loss_value(current_loss)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            "model.pth",
        )

        # Save as artifact for version control.
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file("model.pth")
        wandb.log_artifact(artifact)

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        if epoch % self.log_period == 0:
            self.save_checkpoint(epoch, model, loss_fn, optimizer, current_loss)

    def finish(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        self.save_checkpoint(epoch, model, loss_fn, optimizer, current_loss)


class WandBMonitor(EpochCallBack):
    """
    A class that monitors the model's training loss for every epoch.
    """

    def __init__(self, evaluation_points, true_solution_fn = None, log_interval = 1):
        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project="pinn",
        #     # track hyperparameters and run metadata
        #     config=kwargs,
        # )
        # wandb init is already done in the main script
        self.loss_labels = []
        self.evaluation_points = evaluation_points
        self.true_solution_fn = true_solution_fn
        if true_solution_fn is not None:
            self.true_solution = true_solution_fn(evaluation_points.detach().numpy())
        self.log_interval = log_interval

    def initiated_copy(self):
        return WandBMonitor()

    def prepare(self, max_epochs, model, loss_fn, optimizer):
        wandb.watch(model, log="all", log_freq=self.log_interval)
        self.loss_labels = _gather_loss_labels(loss_fn, model)

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        loss = _get_loss_value(current_loss)

        entry = {"train/epoch": epoch}

        added_loss = 0

        for i, loss_i in enumerate(loss):
            added_loss += loss_i
            entry["train/{}".format(self.loss_labels[i])] = loss_i

        entry["train/added_loss"] = added_loss

        approximated_values = f(model, self.evaluation_points).detach().numpy()
        if self.true_solution_fn is not None:
            entry["validation/loss"] = np.mean(
                np.abs(approximated_values - self.true_solution)
            )

        entry.update(extra_logs)
        wandb.log(entry, step=epoch)


class TrueErrorMonitor(EpochCallBack):
    """
    A class that compares the model's output to the true solution after every epoch.
    """

    def __init__(self, evaluation_points, true_solution_fn: Callable):
        self.true_solution_fn = true_solution_fn
        self.evaluation_points = evaluation_points
        self.mae_history = []

    def initiated_copy(self):
        return TrueErrorMonitor(self.evaluation_points, self.true_solution_fn)

    def process(self, epoch, model, loss_fn, optimizer, current_loss, extra_logs):
        eval_points_np = self.evaluation_points.cpu().detach().numpy()
        predicted_values = f(model, self.evaluation_points)
        predicted_values = predicted_values.cpu().detach().numpy()
        true_values = self.true_solution_fn(eval_points_np)
        self.mae_history.append(np.mean(np.abs(predicted_values - true_values)))


def mean_absolute_error(model, eval_points, true_function):
    """
    Computes the average distance between the output of a neural network and a true solution on a set of evaluation points.
    The distance is computed as the mean absolute difference between the output of the neural network and the solution on the evaluation points.
    
    Parameters
    ----------
    model : torch.nn.Module\\
        The neural network model to evaluate.
    eval_points : torch.Tensor\\
        The evaluation points as a PyTorch tensor of shape (num_points, input_dim).
    true_function : callable\\
        A function that takes a numpy array of shape (num_points, input_dim) as input and returns the true/analytic solution as a numpy array of shape (num_points, output_dim).

    Returns
    -------
    float\\
        The average distance between the output of the neural network and the analytic solution.
    """
    eval_points_np = eval_points.detach().numpy()
    approximated_values = f(model, eval_points)
    approximated_values = approximated_values.detach().numpy()
    true_values = true_function(eval_points_np)
    return np.mean(np.abs(approximated_values - true_values))
