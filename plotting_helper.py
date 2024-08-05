import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage import measure
from vedo import *
from data_processing.mesh_to_array import *
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
import fnmatch
import torch
import random
from utils.helper import make_cmap
import matplotlib.patches as patches

cmap = make_cmap()

# Set global font size parameters
plt.rcParams.update({
    'font.size': 12,              # Global font size for all text elements
    'axes.titlesize': 14,         # Font size for axis titles
    'axes.labelsize': 12,         # Font size for axis labels
    'xtick.labelsize': 10,        # Font size for x-axis tick labels
    'ytick.labelsize': 10,        # Font size for y-axis tick labels
    'legend.fontsize': 12,        # Font size for legend text
    'figure.titlesize': 16        # Font size for figure titles
})
default_figsize = 3

def set_seed(seed):
    torch.manual_seed(seed)  # Sets the seed for CPU
    torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU
    torch.cuda.manual_seed_all(seed)  # Sets the seed for all GPUs (if you use multi-GPU)
    random.seed(seed)  # Sets the seed for Python's built-in random module
    np.random.seed(seed)  # Sets the seed for NumPy
    
    # Ensure deterministic behavior if required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_all_cases(cfg: DictConfig, base_dir=".."):
    if cfg.data.cases == 'all':
        cases = os.listdir(os.path.join(base_dir,cfg.data.processed_data_folder))
        cases = [case.split('.')[0] for case in cases if fnmatch.fnmatch(case, 'case_TCIA*')]
        cases_number = [int(case.split('_')[-2]) for case in cases]
        # cases = [case for case, case_number in zip(cases, cases_number) if case_number < 290]
        # cases 
    else:
        cases = cfg.data.cases
    return cases

def interpolate_arrays(arr, t):
    arr1, arr2, arr3, arr4 = arr
    # Linear interpolation between arrays based on parameter t (0 <= t <= 1)
    return (1 - t) * (1 - t) * arr1 + 2 * (1 - t) * t * arr2 + t * t * arr3 + (1 - t) * (1 - t) * arr4

def remove_empty_space(img, lung_mask=None):           
    mask = img < 0.001
    masked_data = np.ma.masked_where(mask, img)
    rows_to_keep = ~np.all(mask, axis=1)
    cols_to_keep = ~np.all(mask, axis=0)
    masked_data = masked_data[rows_to_keep][:, cols_to_keep]
    if lung_mask is not None:
        mask = lung_mask < 0.001
        lung_mask = np.ma.masked_where(mask, lung_mask)
        lung_mask = lung_mask[rows_to_keep][:, cols_to_keep]
        return masked_data, lung_mask
    return masked_data

def pred_postprocess(preds, targets, resolution=512, nlevel=4, nres=4, cond_value=0.2):
    targets = targets.reshape(-1, nres, nlevel, resolution, resolution)
    preds = preds.reshape(-1, nres, nlevel, resolution, resolution)
    lung_mask = targets[:,0]==cond_value
    lung_mask = lung_mask.unsqueeze(1).tile(1, nres, 1, 1, 1).reshape(-1, resolution, resolution).numpy()
    lung_mask_shrunken = torch.tensor([binary_erosion(mask, structure=np.ones((30,30))).astype(np.uint8) for mask in lung_mask])
    lung_mask_shrunken = lung_mask_shrunken.reshape(-1, nres, nlevel, resolution, resolution)
    return preds.numpy(), targets.numpy(), lung_mask_shrunken.numpy()

def plot_lung_contour(img, mask, cond_value, ax):
    contours = measure.find_contours(mask, level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    # Get the coordinates of the mask
    center_x, center_y = mask.shape[1] / 2, mask.shape[0]
    # Compute the average pixel value of the rectangle
    average_pixel_value = np.mean(img[mask==1])
    # Annotate the rectangle with the average pixel value
    ax.text(center_x, center_y, f'Error: \n{(np.round(cond_value-average_pixel_value, 4)):.4f}',
            color='white', ha='center', va='center', bbox=dict(facecolor='red', alpha=0.8))
    
def generate_cosine_function(max_val, min_val, num_points):
    """
    Generates cosine function values such that the first argument is the maximum value
    and the second argument is the minimum value.

    Parameters:
    max_val (float): The maximum value of the cosine function.
    min_val (float): The minimum value of the cosine function.
    num_points (int): The number of equidistant points to generate.

    Returns:
    np.ndarray: Array of cosine values at equidistant points.
    """
    # Calculate the amplitude and the vertical shift
    amplitude = (max_val - min_val) / 2
    vertical_shift = (max_val + min_val) / 2
    
    # Generate equidistant points in the range [0, 2*pi)
    x = np.linspace(0, 2 * np.pi, num_points)
    
    # Calculate the cosine values with the calculated amplitude and vertical shift
    y = amplitude * np.cos(x) + vertical_shift
    
    return y

def get_noise(dB_value):
    """
    Adds noise sampled from a corresponding distribution to a standard normally distributed value.

    Parameters:
    normal_value (float): A value from a standard normal distribution.
    dB_value (float): The dB value to determine the power of the added noise.

    Returns:
    float: The resulting value after adding the noise.
    """
    # Calculate signal power (since the signal is standard normal, its variance is 1, so power is also 1)
    signal_power = 1
    
    # Calculate the desired noise power
    snr_linear = 10**(dB_value / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise from a normal distribution with variance equal to the linear power
    noise = np.random.normal(scale=np.sqrt(noise_power))
    return noise

def sample_noise(snr_low, snr_high, nres=4):
    snr_cycle = generate_cosine_function(snr_high, snr_low, 13)
    snr_cycle = snr_cycle.reshape(1, 1, 13)
    snr_cycle = snr_cycle.repeat(16, axis=1)
    snr_cycle = snr_cycle.repeat(nres, axis=0)
    noise = get_noise(snr_cycle)
    return noise

def load_model(path):
    cfg = load_cfg(path)
    cfg.inference_path = path
    model = hydra.utils.instantiate(cfg.learning.model, model_3d=cfg.data.model_3d)
    model.load_state_dict(torch.load(os.path.join(path,'model_lung.pt'), map_location=cfg.learning.training.device)['model_state_dict'], strict=False)
    return model, cfg

def load_cfg(path):
    with initialize(version_base=None, config_path=os.path.join(path, ".hydra"), job_name="test"):
        cfg = compose(config_name="config")
    return cfg
