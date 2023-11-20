import torch
from autodmri import estimator as ade
from . import options, plotting, io
import scipy.stats as sst
import scipy.special as ssp
import logging
import numpy as np
import typing
import torch.special as tos

log_module = logging.getLogger(__name__)


def get_noise_stats_across_slices(data_nii: torch.tensor, fit_config: options.FitConfig,
                                  dim_z: int = -2, dim_t: int = -1, num_cores_mp: int = 16):
    """
    compute noise statistics using autodmri, identifying voxels belonging to noise distribution.
    advisable to check the output (especially mask) visually.
    compute the noise distribution across slices. fails if there are not enough sample free voxels within a slice
    (eg. slab selective acquisitions with tight fov).

    inputs
    nii_data: torch tensor assumed to be 4D: [x, y, z, t]
    fit_config: FitConfig configuration object for the fitting
    dim_z: int (default -2) slice dimension
    dim_t: int (default -1) time dimension
    num_cores_mp: number of cores for multiprocessing
    """
    # take first echo
    data_input = torch.moveaxis(data_nii, dim_t, 0)[0]
    # if slice dimension was after time dim, we need to account for this
    if dim_z > dim_t:
        dim_z -= 1
    # if slice dimension was counted from back we need to up it
    if dim_z < 0:
        dim_z += 1

    # for now only take slab axis one echo
    s, n, m = ade.estimate_from_dwis(
        data=data_input.numpy(force=True), axis=dim_z, return_mask=True,
        exclude_mask=None, ncores=num_cores_mp, method="moments", verbose=2, fast_median=False
    )

    if fit_config.visualize:
        plotting.plot_img(m, fit_config=fit_config, name="autodmri_mask")
        plotting.plot_noise_sigma_n(sigma=s, n=n, fit_config=fit_config, name="autodmri_sigma_n")

    # assign values [we get only dim_z axis dimension]
    sigma = torch.from_numpy(s)
    num_channels = torch.from_numpy(n)
    mask = torch.from_numpy(m)

    return sigma, num_channels, mask


def laguerre_half(x: torch.tensor):
    a = torch.exp(x / 2)
    b_1 = (1 - x) * tos.i0(-x / 2)
    b_2 = x * tos.i1(-x / 2)
    return a * (b_1 - b_2)


def noise_mean_rice(data: torch.tensor, sigma: torch.tensor):
    a = sigma * torch.sqrt(torch.tensor(torch.pi / 2))
    v = torch.nan_to_num(
            torch.divide(- data ** 2, 2 * sigma**2),
            nan=0.0, posinf=0.0
        )
    b = laguerre_half(v)
    return a * b


def noise_mean(data: np.ndarray, sigma: typing.Union[float, np.ndarray], n: typing.Union[float, np.ndarray]):
    n = int(np.round(n))
    _amp_sig = np.divide(
        data ** 2, 2 * sigma ** 2,
        where=sigma > 1e-12
    )
    _denominator = np.power(2, n - 1) * ssp.factorial(n - 1)
    _numerator = ssp.factorial2(2 * n - 1) * np.sqrt(np.pi / 2)
    _mean_factor = _numerator / _denominator
    return sigma * _mean_factor * ssp.hyp1f1(-0.5, n, -_amp_sig)


def noise_js(fit_config: options.FitConfig, nii_data: torch.tensor):
    # load in noise mask
    if fit_config.nii_noise_mask:
        use_noise = True
        noise_mask, _ = io.load_nii_data(fit_config.nii_noise_mask)
        noise_mask = noise_mask.to(torch.bool)
    else:
        use_noise = False
        noise_mask = None

    # noise business
    if use_noise:
        # mask only drawn for one echo, extend to all
        noise_vals = nii_data[torch.moveaxis(noise_mask[None], 0, -1).expand(
            *[-1] * len(nii_data.shape[:-1]), nii_data.shape[-1]
        )]
        # we have some 0 we need to scrub for the statistics
        noise_vals = noise_vals[noise_vals > 1]

        # better numpy
        noise_vals = noise_vals.numpy(force=True)
        # fit 0 centered chi
        params = sst.chi.fit(noise_vals, 20, floc=0.0)
        # get histogram bins
        counts, bins = np.histogram(noise_vals, bins=np.arange(np.max(noise_vals)))
        bins = 0.5 * (bins[:-1] + bins[1:])

        # extract results
        num_channels = params[0]
        sigma = params[2]
        # get fit
        chi_pdf = sst.chi(num_channels, loc=0, scale=sigma).pdf(bins)

        if fit_config.visualize:
            # plot histogramm
            plotting.plot_hist_fit(data=noise_vals, fit_line=chi_pdf, fit_config=fit_config, name="noise_fit")

    else:
        num_channels = None
        sigma = None

    return num_channels, sigma


def get_noise_stats_3d_from_echoes():
    # we calculate noise statistics per echo time and per slice in three axes
    # num_channels = torch.zeros(nii_data.shape[:-1])
    # sigma = torch.zeros(nii_data.shape[:-1])
    # mask = torch.zeros(nii_data.shape[:-1], dtype=torch.bool)
    # we assume the low snr regions in the first echoes to be in actual sample free areas and try to use each
    # of the first three echoes to calculate the noise statistics for the 3d axes.
    # this assumes the underlying noise stats be driven by spatial variations rather than time
    # also it assumes a "global" noise statistic throughout the slice of interest (independent of slice orientation)
    # imposed by outside voxels. Might be reasonable with higher snr where the high signal voxels
    # approach gaussian shape regardless of distribution parameter changes in nc-chi
    # ToDo This is actually a neat idea, however for the iron sleep data, there are too many slices where there is no
    #   formally signal free voxels, hence the extraction fails and gives spurious results.
    #   revise this for jstmc or whole brain (not slab) output and may introduce additional smoothing
    # for ax_idx in tqdm.trange(3, desc="processing axes"):
    #     # num_channels, sigma = noise_js(fit_config=fit_config, nii_data=nii_data, fig_path=fig_path)
    #     s, n, m = estimator.estimate_from_dwis(
    #         data=nii_data[:, :, :, ax_idx].numpy(force=True), axis=ax_idx, return_mask=True,
    #         exclude_mask=None, ncores=16, method="moments", verbose=0, fast_median=False
    #     )
    #     # assign values [we get one axis dimension]
    #     sigma = torch.moveaxis(sigma, ax_idx, 0)
    #     sigma += torch.from_numpy(s[:, None, None])
    #     sigma = torch.moveaxis(sigma, 0, ax_idx)
    #     num_channels = torch.moveaxis(num_channels, ax_idx, 0)
    #     num_channels += torch.from_numpy(n[:, None, None])
    #     num_channels = torch.moveaxis(num_channels, 0, ax_idx)
    #     mask = mask | torch.from_numpy(m).to(torch.bool)
    #     # sigma[:, echo_idx] = torch.from_numpy(s)
    #     # num_channels[:, echo_idx] = torch.from_numpy(n)
    #     # mask[:, :, :, echo_idx] = torch.from_numpy(m)
    # sigma /= 3
    # num_channels /= 3
    # use_noise = True
    # # save_imgs
    # file_name = fig_path.parent.joinpath("autodmri_mask").with_suffix(".nii")
    # log_module.info(f"write file: {file_name.as_posix()}")
    # img = nib.Nifti1Image(mask.to(torch.int).numpy(force=True), affine=nii_affine)
    # nib.save(img, file_name.as_posix())
    # file_name = fig_path.parent.joinpath("autodmri_n").with_suffix(".nii")
    # log_module.info(f"write file: {file_name.as_posix()}")
    # img = nib.Nifti1Image(num_channels.numpy(force=True), affine=nii_affine)
    # nib.save(img, file_name.as_posix())
    # file_name = fig_path.parent.joinpath("autodmri_sigma").with_suffix(".nii")
    # log_module.info(f"write file: {file_name.as_posix()}")
    # img = nib.Nifti1Image(sigma.numpy(force=True), affine=nii_affine)
    # nib.save(img, file_name.as_posix())

    pass
