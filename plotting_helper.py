import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from skimage import measure
from vedo import *
from data_processing.mesh_to_array import *
from data_processing.preprocessing import read_egt
from train.testing import testing
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
import fnmatch
import torch
import random
from utils.helper import make_cmap
import matplotlib.patches as patches
import cv2

cmap = make_cmap()

# Set global font size parameters
plt.rcParams.update(
    {
        "font.size": 12,  # Global font size for all text elements
        "axes.titlesize": 14,  # Font size for axis titles
        "axes.labelsize": 12,  # Font size for axis labels
        "xtick.labelsize": 10,  # Font size for x-axis tick labels
        "ytick.labelsize": 10,  # Font size for y-axis tick labels
        "legend.fontsize": 12,  # Font size for legend text
        "figure.titlesize": 16,  # Font size for figure titles
    }
)
default_figsize = 3


def set_seed(seed):
    torch.manual_seed(seed)  # Sets the seed for CPU
    torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU
    torch.cuda.manual_seed_all(
        seed
    )  # Sets the seed for all GPUs (if you use multi-GPU)
    random.seed(seed)  # Sets the seed for Python's built-in random module
    np.random.seed(seed)  # Sets the seed for NumPy

    # Ensure deterministic behavior if required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# LOADING
def load_model(path, device="cuda"):
    cfg = load_cfg(path)
    cfg.inference_path = path
    model = hydra.utils.instantiate(cfg.learning.model, model_3d=cfg.data.model_3d)
    model.load_state_dict(
        torch.load(os.path.join(path, "model_lung.pt"), map_location=device)[
            "model_state_dict"
        ],
        strict=False,
    )
    return model, cfg


def load_cfg(path):
    with initialize(
        version_base=None, config_path=os.path.join(path, ".hydra"), job_name="test"
    ):
        cfg = compose(config_name="config")
    return cfg


def load_case(case, test_dataset):
    points_case = []
    signals_case = []
    electrodes_case = []
    masks_case = []
    targets_case = []

    # extract case from the dataloader
    for i, data in enumerate(test_dataset):
        if fnmatch.fnmatch(test_dataset.case_files[i], f"{case}*.npz"):
            points, signals, electrodes, masks, targets, _ = data
            points_case.append(points)
            signals_case.append(signals)
            electrodes_case.append(electrodes)
            masks_case.append(masks)
            targets_case.append(targets)
        else:
            continue
    points_case = torch.stack(points_case, dim=0)
    signals_case = torch.stack(signals_case, dim=0)
    electrodes_case = torch.stack(electrodes_case, dim=0)
    masks_case = torch.stack(masks_case, dim=0)
    targets_case = torch.stack(targets_case, dim=0)
    return points_case, signals_case, electrodes_case, masks_case, targets_case


def get_all_cases(cfg: DictConfig, base_dir=".."):
    if cfg.data.cases == "all":
        cases = os.listdir(os.path.join(base_dir, cfg.data.processed_data_folder))
        cases = [
            case.split(".")[0] for case in cases if fnmatch.fnmatch(case, "case_TCIA*")
        ]
        cases_number = [int(case.split("_")[-2]) for case in cases]
        # cases = [case for case, case_number in zip(cases, cases_number) if case_number < 290]
        # cases
    else:
        cases = cfg.data.cases
    return cases


# LITTLE HELPER
def interpolate_arrays(arr, t):
    arr1, arr2, arr3, arr4 = arr
    # Linear interpolation between arrays based on parameter t (0 <= t <= 1)
    return (
        (1 - t) * (1 - t) * arr1
        + 2 * (1 - t) * t * arr2
        + t * t * arr3
        + (1 - t) * (1 - t) * arr4
    )


# PREDICTING
def predict_case(model, dataset, case, device="cuda", electrode_level_only=True):
    # load
    points_case, signals_case, electrodes_case, masks_case, targets_case = load_case(
        case, dataset
    )
    nres = signals_case.shape[0]
    nlevel = targets_case.reshape(nres, -1, 512, 512).shape[1]

    # predict
    preds_all = []
    targets_all = []

    points = points_case.reshape(4, -1, nlevel, 512, 512, 3)
    target = targets_case.reshape(4, -1, nlevel, 512, 512, 1)
    mask = masks_case.reshape(4, -1, nlevel, 512, 512, 1)
    if electrode_level_only:
        levels = torch.arange(nlevel)
        electrode_levels = torch.linspace(levels[1], levels[-2], 4).numpy().astype(int)
        points = points[:, :, electrode_levels].reshape(4, -1, 3)
        target = target[:, :, electrode_levels].reshape(4, -1, 1)
        mask = mask[:, :, electrode_levels].reshape(4, -1, 1)
        nlevel = len(electrode_levels)

    for i in range(points.shape[0]):
        _, pred, _ = testing(
            model,
            [
                signals_case[i].unsqueeze(0).float(),
                electrodes_case[i].unsqueeze(0).float(),
                points[i].unsqueeze(0).float(),
            ],
            device=device,
            wandb_log=False,
            electrode_level_only=True,
            point_levels_3d=nlevel,
        )
        targets_all.append(target[i].detach().cpu().squeeze())
        preds_all.append(pred.detach().cpu().squeeze())
    preds_all = torch.concatenate(preds_all, axis=0)
    targets_all = torch.concatenate(targets_all, axis=0)
    preds_case, targets_case, lung_masks_case = pred_postprocess(
        preds_all, targets_all, resolution=512, nres=nres, nlevel=nlevel
    )
    preds_case, targets_case, lung_masks_case = (
        preds_case.squeeze(0),
        targets_case.squeeze(0),
        lung_masks_case.squeeze(0),
    )
    eroded_masks_case = mask.reshape(4, nlevel, 512, 512)
    return (
        preds_case,
        targets_case,
        lung_masks_case,
        eroded_masks_case,
        electrodes_case,
        signals_case,
        points_case,
    )


def pred_postprocess(preds, targets, resolution=512, nlevel=4, nres=4, cond_value=0.2):
    targets = targets.reshape(-1, nres, nlevel, resolution, resolution)
    preds = preds.reshape(-1, nres, nlevel, resolution, resolution)
    lung_mask = targets[:, 0] == cond_value
    lung_mask = (
        lung_mask.unsqueeze(1)
        .tile(1, nres, 1, 1, 1)
        .reshape(-1, resolution, resolution)
        .numpy()
    )
    lung_mask_shrunken = torch.tensor(
        [
            binary_erosion(mask, structure=np.ones((30, 30))).astype(np.uint8)
            for mask in lung_mask
        ]
    )
    lung_mask_shrunken = lung_mask_shrunken.reshape(
        -1, nres, nlevel, resolution, resolution
    )
    return preds.numpy(), targets.numpy(), lung_mask_shrunken.numpy()


# SIGNAL NOISE
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
    # calculate signal power (since the signal is standard normal, its variance is 1, so power is also 1)
    signal_power = 1

    # calculate the desired noise power
    snr_linear = 10 ** (dB_value / 20)
    noise_power = signal_power / snr_linear
    # sample noise from a normal distribution with variance equal to the linear power
    noise = np.random.normal(scale=noise_power)
    return noise


def sample_noise(snr_low, snr_high, nres=4):
    snr_cycle = generate_cosine_function(snr_high, snr_low, 13)
    snr_cycle = snr_cycle.reshape(1, 1, 13)
    snr_cycle = snr_cycle.repeat(16, axis=1)
    snr_cycle = snr_cycle.repeat(nres, axis=0)
    noise = get_noise(snr_cycle)
    return noise


# PLOTTER
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


def plot_lung_contour(img, mask, cond_value, ax, show_std=False):
    contours = measure.find_contours(mask, level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")
    # Get the coordinates of the mask
    center_x, center_y = mask.shape[1] / 2, mask.shape[0]
    # Compute the average pixel value of the rectangle
    # error = np.sum((img[mask==1]-cond_value)**2)/np.sum(mask==1)
    mae = np.mean(np.abs(img[mask == 1] - cond_value))
    rmse = np.sqrt(np.mean((img[mask == 1] - cond_value) ** 2))
    rmse = np.round(rmse, 4)
    if np.isnan(rmse):
        rmse = "-"
    mean = img[mask == 1].mean()
    std = (img[mask == 1] - cond_value).std()
    # Annotate the rectangle with the average pixel value
    if show_std:
        ax.text(
            center_x,
            center_y,
            f"{rmse:.4f}\n({(np.round(std, 4)):.4f})",
            weight="bold",
            color="white",
            ha="center",
            va="top",
            bbox=dict(facecolor="red", alpha=0.8),
        )
    else:
        if rmse != "-":
            ax.text(
                center_x,
                center_y,
                f"{rmse:.4f}",
                weight="bold",
                color="white",
                ha="center",
                va="top",
                bbox=dict(facecolor="red", alpha=0.8),
            )
        else:
            ax.text(
                center_x,
                center_y,
                f"{rmse}",
                weight="bold",
                color="white",
                ha="center",
                va="top",
                bbox=dict(facecolor="red", alpha=0.8),
            )


def plot_sirt_contour(
    tomogram,
    cond_value,
    center1=(35, 65),
    center2=(95, 65),
    axes=(7, 4),
    ax=None,
    show_std=False,
    return_error=False,
):
    # Define the size of the image
    height, width = 128, 128  # Example dimensions

    # Create a blank mask (all zeros)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the parameters for the ellipses
    center1 = (35, 65)
    center2 = (95, 65)
    axes = (7, 4)  # Lengths of the semi-major and semi-minor axes

    # Draw the first ellipse
    cv2.ellipse(
        mask,
        center1,
        axes,
        angle=90,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    # Draw the second ellipse
    cv2.ellipse(
        mask,
        center2,
        axes,
        angle=90,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    contours = measure.find_contours(mask, level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")
    # Get the coordinates of the mask
    center_x, center_y = mask.shape[1] / 2, mask.shape[0]
    # Compute the average pixel value of the rectangle
    # error = np.sum((img[mask==1]-cond_value)**2)/np.sum(mask==1)
    mae = np.mean(np.abs(tomogram[mask == 255] - cond_value))
    rmse = np.sqrt(np.mean((tomogram[mask == 255] - cond_value) ** 2))
    mean = tomogram[mask == 255].mean()
    std = (tomogram[mask == 255] - cond_value).std()
    if return_error:
        return rmse
    rmse = np.round(rmse, 4)
    # Annotate the rectangle with the average pixel value
    if show_std:
        ax.text(
            center_x,
            center_y,
            f"{(np.round(rmse, 4)):.4f}({(np.round(std, 4)):.4f})",
            weight="bold",
            color="white",
            ha="center",
            va="top",
            bbox=dict(facecolor="red", alpha=0.8),
        )
    else:
        ax.text(
            center_x,
            center_y,
            f"{(np.round(rmse, 4)):.4f}",
            weight="bold",
            color="white",
            ha="center",
            va="top",
            bbox=dict(facecolor="red", alpha=0.8),
        )
    return ax


def plot_one_level(
    targets_case,
    preds_case,
    lung_masks_case,
    cmap,
    level=1,
    nres=4,
    show_cbar=True,
    save=None,
    aspect_ratio=1,
):
    level -= 1
    # filter data
    targets_plot = targets_case[:, level]
    lung_masks_plot = lung_masks_case[:, level]
    preds_plot = preds_case[:, level]

    # set up figure
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))
    if show_cbar:
        cbar_ax = fig.add_axes(
            [0.15, -0.01, 0.74, 0.04]
        )  # [left, bottom, width, height]

    # GT across levels (choose cond. 0.05)
    for res in range(nres):
        cond_value = np.mean(targets_plot[res][lung_masks_plot[res] == 1])
        t = remove_empty_space(targets_plot[res])
        axes[0, res].imshow(t, vmin=0, vmax=0.7, cmap=cmap)
        axes[0, res].set_aspect(aspect_ratio)
        axes[0, res].axis("off")
        axes[0, res].set_title(f"{cond_value:.2f} S/m")

        p, m = remove_empty_space(preds_plot[res], lung_masks_plot[res])
        axes[1, res].imshow(p, vmin=0, vmax=0.7, cmap=cmap)
        axes[1, res].set_aspect(aspect_ratio)
        axes[1, res].axis("off")
        plot_lung_contour(p, m, cond_value, axes[1, res])
    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Conductivity (S/m)")
    spacing = 0.4
    x_location = 0.1
    y_location = 0.7
    fig.text(
        x_location,
        y_location,
        f"Ground Truth",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(
        x_location,
        y_location - 1 * spacing,
        f"RESIST",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    plt.show()
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_columns(
    targets_case,
    preds_case,
    lung_masks_case,
    cmap,
    nlevel=4,
    nres=4,
    save=None,
    aspect_ratio=1,
):
    for i in range(nres):
        fig, axes = plt.subplots(nlevel, 2, figsize=(10, int(nlevel * 4)))
        if nlevel == 1:
            axes = axes.reshape(1, 2)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        for level in range(nlevel):
            if level == 0:
                axes[level, 0].set_title("Ground Truth")
                axes[level, 1].set_title("Model Tomogram")
            # GT
            target = remove_empty_space(targets_case[i, level])
            axes[level, 0].imshow(target, vmin=0, vmax=0.7, cmap=cmap)
            axes[level, 0].axis("off")
            axes[level, 0].set_aspect(aspect_ratio)
            # Pred + Mask
            cond_value = np.mean(targets_case[i, level][lung_masks_case[i, level] == 1])
            p, m = remove_empty_space(preds_case[i, level], lung_masks_case[i, level])
            axes[level, 1].imshow(p, vmin=0, vmax=0.7, cmap=cmap)
            # axes[level,1].imshow(m, cmap='Greys', alpha=0.3)
            axes[level, 1].set_aspect(aspect_ratio)
            axes[level, 1].axis("off")
            plot_lung_contour(p, m, cond_value, axes[level, 1])

        # Add colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Conductivity (S/m)")
        # fig.suptitle(f'{test_dataset.cases[i]}')
        plt.show()
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=600)
    plt.close(fig)


def plot_grid(
    targets_case,
    preds_case,
    lung_masks_case,
    nres=4,
    nlevel=4,
    show_cbar=True,
    cbar_orientation="vertical",
    save=None,
    y_horizontal_line=0.01,
    y_vertical_line=0.25,
    aspect_ratio=1,
):
    ncols = nres + 1
    nrows = nlevel + 1
    # set up figure
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(int((ncols) * 1.8), int(1.7 * nrows) - 1)
    )
    if show_cbar:
        if cbar_orientation == "vertical":
            cbar_ax = fig.add_axes(
                [0.95, 0.15, 0.02, 0.7]
            )  # [left, bottom, width, height] # vertical
        elif cbar_orientation == "horizontal":
            cbar_ax = fig.add_axes(
                [0.15, 0.95, 0.75, 0.02]
            )  # [left, bottom, width, height]  # horizontal

    axes[0, 0].text(0.5, 0.5, "Ground Truth", ha="center", va="center", rotation=45)
    axes[0, 0].text(0.5, 0.0, "Level", ha="center", va="center", rotation=0)
    axes[0, 0].text(0.97, 0.4, " Conductivity", ha="center", va="center", rotation=90)
    axes[0, 0].axis("off")

    for level in range(nlevel):
        t = remove_empty_space(targets_case[0, level])
        axes[level + 1, 0].imshow(t, vmin=0, vmax=0.7, cmap=cmap)
        axes[level + 1, 0].set_aspect(aspect_ratio)
        axes[level + 1, 0].axis("off")
        for res in range(nres):
            cond_value = np.mean(
                targets_case[res, level][lung_masks_case[res, level] == 1]
            )
            # GT across cond. values (choose level 0)
            if level == 0:
                t = remove_empty_space(targets_case[res, level])
                axes[0, res + 1].imshow(t, vmin=0, vmax=0.7, cmap=cmap)
                axes[0, res + 1].set_aspect(aspect_ratio)
                axes[0, res + 1].axis("off")
                axes[level, int(res + 1)].set_title(f"{cond_value:.2f} S/m")
            p, m = remove_empty_space(
                preds_case[res, level], lung_masks_case[res, level]
            )
            axes[level + 1, res + 1].imshow(p, vmin=0, vmax=0.7, cmap=cmap)
            axes[level + 1, res + 1].set_aspect(aspect_ratio)
            # axes[level+1,res+1].imshow(m, cmap='Greys', alpha=0.1)
            axes[level + 1, res + 1].axis("off")
            plot_lung_contour(p, m, cond_value, axes[level + 1, res + 1])

            axes[level + 1, res + 1].axis("off")
            axes[level + 1, res + 1].set_facecolor("grey")

    # add colorbar to the figure
    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation=cbar_orientation)
        cbar.set_label("Conductivity (S/m)")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.set_ticks_position("top")

    # Draw horizontal and vertical lines separating the lower right 4x4 grid
    # Use the figure's add_artist method to add lines at the required positions
    # Get the positions of the first row and first column of the lower right 4x4 grid
    top_left_of_4x4 = axes[1, 1].get_position()
    bottom_right_of_4x4 = axes[nrows - 1, ncols - 1].get_position()
    top_left = axes[0, 0].get_position()
    top_first = axes[0, 1].get_position()
    second_left = axes[1, 0].get_position()

    # Adjusted positions for the lines to be outside the plots
    horizontal_line_y = top_left_of_4x4.y1 + 0.0
    vertical_line_x = top_left_of_4x4.x0 - 0.0

    # Draw a horizontal line
    fig.add_artist(
        plt.Line2D(
            [top_left.x0, bottom_right_of_4x4.x1],
            [second_left.y1 + y_horizontal_line, second_left.y1 + y_horizontal_line],
            color="black",
            linewidth=2,
        )
    )

    # Draw a vertical line
    fig.add_artist(
        plt.Line2D(
            [vertical_line_x - 0.015, vertical_line_x - 0.015],
            [bottom_right_of_4x4.y0, top_first.y1],
            color="black",
            linewidth=2,
        )
    )
    plt.show()
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_sirt(path, ax=None, threshold=50, return_array=False):
    if ax is None:
        fig, ax = plt.subplots()
    tomogram = read_egt(path)
    tomogram = np.where(tomogram > threshold, threshold, tomogram)
    tomogram = np.where(tomogram == 0, 0, 1 / tomogram)
    tomogram = np.rot90(tomogram)
    ax.imshow(tomogram, cmap=cmap, vmin=0, vmax=0.7)
    # ax.colorbar()
    if return_array:
        return tomogram, ax
    return ax


def plot_sirt_comparison(
    targets_case,
    preds_case,
    lung_masks_case,
    case,
    cmap,
    save=None,
    show_cbar=True,
    show_sirt=True,
    aspect_ratio=1,
):
    rho = [5, 10, 15, 20]
    fig, axes = plt.subplots(3, len(rho), figsize=[7, 5])
    cbar_ax = fig.add_axes([0.93, 0.06, 0.03, 0.75])  # [left, bottom, width, height]

    targets_plot = targets_case[:, 0].squeeze()
    pred_plot = preds_case[:, 0].squeeze()
    mask_plot = lung_masks_case[:, 0].squeeze()

    for i, r in enumerate(rho):
        cond_value = np.mean(targets_plot[i][mask_plot[i] == 1])
        t = remove_empty_space(targets_plot[i])
        axes[0, i].imshow(t, vmin=0, vmax=0.7, cmap=cmap)
        axes[0, i].set_aspect(aspect_ratio)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{cond_value:.2f} S/m")
        # RESIST
        p, m = remove_empty_space(pred_plot[i], mask_plot[i])
        axes[1, i].imshow(p, vmin=0, vmax=0.7, cmap=cmap)
        axes[1, i].set_aspect(aspect_ratio)
        # axes[1,i].imshow(m, cmap='Greys', alpha=0.3)
        plot_lung_contour(p, m, cond_value, axes[1, i])
        axes[1, i].axis("off")

        # SIRT
        if show_sirt:
            if os.path.exists(
                "data/raw/"
                + case
                + "/tomograms_rad/"
                + case
                + "_16adj_rt2_rho_"
                + str(r)
                + "_z1_1.egt"
            ):
                sirt_path = (
                    "data/raw/"
                    + case
                    + "/tomograms_rad/"
                    + case
                    + "_16adj_rt2_rho_"
                    + str(r)
                    + "_z1_1.egt"
                )
            else:
                sirt_path = "data/raw/" + case + "/tomograms_rad/level_1_" + str(r) + ".egt"
            tomogram, axes[2, i] = plot_sirt(
                sirt_path, ax=axes[2, i], return_array=True
            )
            plot_sirt_contour(tomogram, cond_value, ax=axes[2, i])
        axes[2, i].axis("off")
    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Conductivity (S/m)")
    else:
        cbar_ax.axis("off")
    spacing = 0.2
    x_location = 0.08
    y_location = 0.7
    fig.text(
        x_location,
        y_location,
        f"Ground\nTruth",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(
        x_location,
        y_location - 1 * spacing,
        f"RESIST",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(
        x_location,
        y_location - 2.4 * spacing,
        f"SIRT",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    for j in range(4):  # Iterate over the columns in the first row
        bbox = axes[0, j].get_position()  # Get the current position of the axes
        new_bbox = [bbox.x0, bbox.y0 - 0.06, bbox.width, bbox.height]
        axes[0, j].set_position(new_bbox)  # Set the new position with increased size
    # plt.tight_layout()
    plt.show()
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=600)
    plt.close(fig)
