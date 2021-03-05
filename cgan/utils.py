import os
import torch
from torch.autograd import Variable
import shutil
import queue
import logging
import tqdm
import random
import numpy as np
import torch.nn.functional as F


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def var2tensor(x):
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def get_save_dir(base_dir, name, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_one_hot_labels(labels, n_classes):
    """
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    """
    return F.one_hot(labels, num_classes=n_classes)


def combine_vectors(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    combined = torch.cat((x.float(), y.float()), dim=1)
    return combined


# class CheckpointSaver:
#     """Class to save and load model checkpoints.
#
#     Save the best checkpoints as measured by a metric value passed into the
#     `save` method. Overwrite checkpoints with better checkpoints once
#     `max_checkpoints` have been saved.
#
#     Args:
#         save_dir (str): Directory to save checkpoints.
#         max_checkpoints (int): Maximum number of checkpoints to keep before
#             overwriting old ones.
#         metric_name (str): Name of metric used to determine best model.
#         maximize_metric (bool): If true, best checkpoint is that which maximizes
#             the metric value passed in via `save`. Otherwise, best checkpoint
#             minimizes the metric.
#         log (logging.Logger): Optional logger for printing information.
#     """
#     def __init__(self, save_dir, max_checkpoints, metric_name,
#                  maximize_metric=False, log=None):
#         super(CheckpointSaver, self).__init__()
#
#         self.save_dir = save_dir
#         self.max_checkpoints = max_checkpoints
#         self.metric_name = metric_name
#         self.maximize_metric = maximize_metric
#         self.best_val = None
#         self.ckpt_paths = queue.PriorityQueue()
#         self.log = log
#         self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")
#
#     def is_best(self, metric_val):
#         """Check whether `metric_val` is the best seen so far.
#
#         Args:
#             metric_val (float): Metric value to compare to prior checkpoints.
#         """
#         if metric_val is None:
#             # No metric reported
#             return False
#
#         if self.best_val is None:
#             # No checkpoint saved yet
#             return True
#
#         return ((self.maximize_metric and self.best_val < metric_val)
#                 or (not self.maximize_metric and self.best_val > metric_val))
#
#     def _print(self, message):
#         """Print a message if logging is enabled."""
#         if self.log is not None:
#             self.log.info(message)
#
#     def save(self, step, model, metric_val, device):
#         """Save model parameters to disk.
#
#         Args:
#             step (int): Total number of examples seen during training so far.
#             model (torch.nn.DataParallel): Model to save.
#             metric_val (float): Determines whether checkpoint is best so far.
#             device (torch.device): Device where model resides.
#         """
#         ckpt_dict = {
#             'model_name': model.__class__.__name__,
#             'model_state': model.cpu().state_dict(),
#             'step': step
#         }
#         model.to(device)
#
#         checkpoint_path = os.path.join(self.save_dir,
#                                        f'step_{step}.pth.tar')
#         torch.save(ckpt_dict, checkpoint_path)
#         self._print(f'Saved checkpoint: {checkpoint_path}')
#
#         if self.is_best(metric_val):
#             # Save the best model
#             self.best_val = metric_val
#             best_path = os.path.join(self.save_dir, 'best.pth.tar')
#             shutil.copy(checkpoint_path, best_path)
#             self._print(f'New best checkpoint at step {step}...')
#
#         # Add checkpoint path to priority queue (lowest priority removed first)
#         if self.maximize_metric:
#             priority_order = metric_val
#         else:
#             priority_order = -metric_val
#
#         self.ckpt_paths.put((priority_order, checkpoint_path))
#
#         # Remove a checkpoint if more than max_checkpoints have been saved
#         if self.ckpt_paths.qsize() > self.max_checkpoints:
#             _, worst_ckpt = self.ckpt_paths.get()
#             try:
#                 os.remove(worst_ckpt)
#                 self._print(f'Removed checkpoint: {worst_ckpt}')
#             except OSError:
#                 # Avoid crashing if checkpoint has been removed or protected
#                 pass