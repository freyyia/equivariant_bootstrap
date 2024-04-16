"""
Demo for Poisson Deblurring using Reflected PnP-ULA
====================================================================================================
Building on DPIR method for PnP image deblurring Tutorial from the Deepinv Library
https://deepinv.github.io/deepinv/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.html

"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models import GSDRUNet
from deepinv.optim.prior import RED
from deepinv.optim.data_fidelity import PoissonLikelihood
from torchvision import transforms
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.utils import load_url_image
import wandb
from tqdm import tqdm
import numpy as np

from deepinv.utils import (
    cal_psnr,
    zeros_like,
    wandb_plot_curves,
    plot_curves,
    plot
)


def test(
    model,
    test_dataloader,
    physics,
    device="cpu",
    plot_images=False,
    save_folder="results",
    plot_metrics=False,
    verbose=True,
    plot_only_first_batch=True,
    wandb_vis=False,
    wandb_setup={},
    step=0,
    online_measurements=False,
    plot_measurements=True,
    **kwargs,
):
    r"""

    Modifying this function to test a sampling method
    Original docstring:
    Tests a reconstruction network.

    This function computes the PSNR of the reconstruction network on the test set,
    and optionally plots the reconstructions as well as the metrics computed along the iterations.
    Note that by default only the first batch is plotted.

    :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param torch.utils.data.DataLoader test_dataloader: Test data loader, which should provide a tuple of (x, y) pairs.
        See :ref:`datasets <datasets>` for more details.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s)
        used by the reconstruction network at test time.
    :param torch.device device: gpu or cpu.
    :param bool plot_images: Plot the ground-truth and estimated images.
    :param str save_folder: Directory in which to save plotted reconstructions.
    :param bool plot_metrics: plot the metrics to be plotted w.r.t iteration.
    :param bool verbose: Output training progress information in the console.
    :param bool plot_only_first_batch: Plot only the first batch of the test set.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    :param dict wandb_setup: Dictionary with the setup for wandb, see https://docs.wandb.ai/quickstart for more details.
    :param int step: Step number for wandb visualization.
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``.
    :param bool plot_measurements: Plot the measurements y. default=True.
    :returns: A tuple of floats (test_psnr, test_std_psnr, linear_std_psnr, linear_std_psnr) with the PSNR of the
        reconstruction network and a simple linear inverse on the test set.
    """
    save_folder = Path(save_folder)

    psnr_init = []
    psnr_net = []

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)

    show_operators = 5

    if wandb_vis:
        if wandb.run is None:
            wandb.init(**wandb_setup)
        psnr_data = []

    for g in range(G):
        dataloader = test_dataloader[g]
        if verbose:
            print(f"Processing data of operator {g+1} out of {G}")
        for i, batch in enumerate(tqdm(dataloader, disable=not verbose)):
            with torch.no_grad():
                if online_measurements:
                    (
                        x,
                        _,
                    ) = batch  # In this case the dataloader outputs also a class label
                    x = x.to(device)
                    physics_cur = physics[g]
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        physics_cur.module.noise_model.__init__()
                    else:
                        physics_cur.reset()
                    y = physics_cur(x)
                else:
                    x, y = batch
                    if type(x) is list or type(x) is tuple:
                        x = [s.to(device) for s in x]
                    else:
                        x = x.to(device)
                    physics_cur = physics[g]

                    y = y.to(device)

                # if plot_metrics:
                #     x1, metrics = model(y, physics_cur, x_gt=x, compute_metrics=True)
                # else:
                x1, _ = model(y, physics[g], x_init=x)
                print(x1)

                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        x_init = physics_cur.module.A_adjoint(y)
                    else:
                        x_init = physics_cur.A_adjoint(y)
                elif hasattr(physics_cur, "A_dagger"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        x_init = physics_cur.module.A_dagger(y)
                    else:
                        x_init = physics_cur.A_dagger(y)
                else:
                    x_init = zeros_like(x)

                cur_psnr_init = cal_psnr(x_init, x)
                cur_psnr = cal_psnr(x1, x)
                psnr_init.append(cur_psnr_init)
                psnr_net.append(cur_psnr)

                if wandb_vis:
                    psnr_data.append([g, i, cur_psnr_init, cur_psnr])

                if plot_images:
                    save_folder_im = (
                        (save_folder / ("G" + str(g))) if G > 1 else save_folder
                    ) / "images"
                    save_folder_im.mkdir(parents=True, exist_ok=True)
                else:
                    save_folder_im = None
                if plot_metrics:
                    save_folder_curve = (
                        (save_folder / ("G" + str(g))) if G > 1 else save_folder
                    ) / "curves"
                    save_folder_curve.mkdir(parents=True, exist_ok=True)

                if plot_images or wandb_vis:
                    if g < show_operators:
                        if not plot_only_first_batch or (
                            plot_only_first_batch and i == 0
                        ):
                            if plot_measurements and len(y.shape) == 4:
                                imgs = [y, x_init, x1, x]
                                name_imgs = ["Input", "No learning", "Recons.", "GT"]
                            else:
                                imgs = [x_init, x1, x]
                                name_imgs = ["No learning", "Recons.", "GT"]

                            print(imgs)
                            fig = plot(
                                imgs,
                                titles=name_imgs,
                                save_dir=save_folder_im if plot_images else None,
                                show=plot_images,
                                return_fig=True,
                            )
                            if wandb_vis:
                                wandb.log(
                                    {
                                        f"Test images batch_{i} (G={g}) ": wandb.Image(
                                            fig
                                        )
                                    }
                                )

                # if plot_metrics:
                #     plot_curves(metrics, save_dir=save_folder_curve, show=True)
                #     if wandb_vis:
                #         wandb_plot_curves(metrics, batch_idx=i, step=step)

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    linear_psnr = np.mean(psnr_init)
    linear_std_psnr = np.std(psnr_init)
    if verbose:
        print(
            f"Test PSNR: No learning rec.: {linear_psnr:.2f}+-{linear_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. "
        )
    if wandb_vis:
        wandb.log({"Test PSNR": test_psnr}, step=step)

    return test_psnr, test_std_psnr, linear_psnr, linear_std_psnr



# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"


# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------
# In this example, we use the Set3C dataset and a motion blur kernel from
# `Levin et al. (2009) <https://ieeexplore.ieee.org/abstract/document/5206815/>`_.
#

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Set up the variable to fetch dataset and operators.
method = "RPnP-ULA"
dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 32
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)

# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_torch = load_degradation("Levin09.npy", DEG_DIR / "kernels", index=kernel_index)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions
dataset = load_dataset(dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)


# %%
# Generate a dataset of blurred images and load it.
# --------------------------------------------------------------------------------
# We use the BlurFFT class from the physics module to generate a dataset of blurred images.


noise_level_img = 1.0/20  # Poisson Noise gain for the degradation
n_channels = 3  # 3 for color images, 1 for gray-scale images
p = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=kernel_torch,
    device=device,
    noise_model=dinv.physics.PoissonNoise(gain=noise_level_img),
)

# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 1 if torch.cuda.is_available() else 0

n_images_max = 3  # Maximal number of images to restore from the input dataset
# Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
operation = "deblur"
measurement_dir = DATA_DIR / dataset_name / operation
dinv_dataset_path = dinv.datasets.generate_dataset(
    train_dataset=dataset,
    test_dataset=None,
    physics=p,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
)

batch_size = 3  # batch size for testing. As the number of iterations is fixed, we can use batch_size > 1
# and restore multiple images in parallel.
dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)

# %%
# Set up the R-PnP-ULA algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------

# test image to set parameters
url = "https://huggingface.co/datasets/deepinv/set3c/resolve/main/starfish.png?download=true"
x = load_url_image(url=url, img_size=256).to(device)
test_noisy_im = p(x)
image_mean = (1/noise_level_img)*torch.mean(test_noisy_im)
beta = image_mean * 0.02

# set specific parameters for PnP ULA
eps = 20
L_y =  (1/noise_level_img)**2*(torch.max(test_noisy_im)/beta**2)
delta_max = 1.0/(1.0/eps+L_y)
delta_frac = 2
delta = delta_max*delta_frac
alpha = 1.07
params_algo = {"stepsize": delta.cpu(), "g_param": alpha, "noise_lvl": eps}
early_stop = False  # Do not stop algorithm with convergence criteria

# Select the data fidelity term
data_fidelity = PoissonLikelihood(gain=noise_level_img, bkg=beta)

# download weights 
# file_name = "Prox-DRUNet.ckpt"
# url = "https://huggingface.co/deepinv/gradientstep/resolve/main/Prox-DRUNet.ckpt?download=true"
# weights = torch.hub.load_state_dict_from_url(
#         url, map_location=lambda storage, loc: storage, file_name=file_name
#     )


# Specify the denoising prior

# The GSPnP prior corresponds to a RED prior with an explicit `g`.
# We thus write a class that inherits from RED for this custom prior.
class GSPnP(RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


ckpt_path = "/home/others/tk2017/private/code/bregman_sampling/BregmanPnP/GS_denoising/ckpts/Prox-DRUNet.ckpt"
# Specify the Denoising prior
backbone = GSPnP(
    denoiser=dinv.models.GSDRUNet(pretrained=ckpt_path).to(device)
)
#backbone = dinv.models.GSDRUNet(pretrained=ckpt_path).to(device)

prior = dinv.optim.ScorePrior(backbone)

#backbone = dinv.models.DnCNN(in_channels=x.shape[1], out_channels=x.shape[1],
#                                                pretrained='download_lipschitz').to(device)
#prior = dinv.optim.ScorePrior(backbone)

# set up the ULA algorithm
thinning = 30
burnin = .1
MC = 300
model = dinv.sampling.ULA(prior, data_fidelity, step_size=20*params_algo["stepsize"],
                            sigma=1/20, alpha=torch.tensor(params_algo["g_param"]), verbose=True,
                            max_iter=int(MC*thinning/(.95-burnin)),
                            thinning=thinning, save_chain=True, burnin_ratio=burnin, clip=(0., 1.),
                            thresh_conv=1e-4)



# %%
# Evaluate the model on the problem.
# --------------------------------------------------------------------
# The test function evaluates the model on the test dataset and computes the metrics.
#

save_folder = RESULTS_DIR / method / operation / dataset_name
wandb_vis = True  # plot curves and images in Weight&Bias.
plot_metrics = True  # plot metrics. Metrics are saved in save_folder.
plot_images = True  # plot images. Images are saved in save_folder.

dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

test(
    model=model,
    test_dataloader=dataloader,
    physics=p,
    device=device,
    plot_images=plot_images,
    save_folder=save_folder,
    plot_metrics=plot_metrics,
    verbose=True,
    wandb_vis=wandb_vis,
    step=1,
    plot_only_first_batch=False,  # By default only the first batch is plotted.
)
