import typing

import numpy as np
import simple_parsing as sp
import dataclasses as dc
import logging

log_module = logging.getLogger(__name__)


@dc.dataclass
class ImageAcqParameters(sp.helpers.Serializable):
    # define all recon parameters we ship in interface
    n_read: int = -1
    n_phase: int = -1
    n_slice: int = -1

    resolution_read: float = 0.0
    resolution_phase: float = 0.0
    resolution_slice: float = 0.0

    read_dir: str = ""
    os_factor: int = -1
    acc_read: bool = False
    acc_factor_phase: float = 0.0

    etl: int = -1
    te: list = sp.field(default_factory=lambda: [0.0])


@dc.dataclass
class NavigatorAcqParameters(ImageAcqParameters):
    lines_per_nav: int = -1
    num_of_nav: int = -1
    nav_acc_factor: int = -1
    nav_resolution_scaling: float = 0.0


@dc.dataclass
class ReconParameters(sp.Serializable):
    multi_echo_img: ImageAcqParameters = ImageAcqParameters()
    navigator_img: NavigatorAcqParameters = NavigatorAcqParameters()

    def set_recon_params(
            self, img_n_read: int, img_n_phase: int, img_n_slice: int, img_resolution_read: float,
            img_resolution_phase: float, img_resolution_slice: float, read_dir: str, os_factor: int,
            acc_read: bool, acc_factor_phase: float, etl: int, te: typing.Union[list, np.ndarray]
    ):
        if isinstance(te, np.ndarray):
            te = te.tolist()
        self.multi_echo_img.n_read = img_n_read
        self.multi_echo_img.n_phase = img_n_phase
        self.multi_echo_img.n_slice = img_n_slice
        self.multi_echo_img.resolution_read = img_resolution_read
        self.multi_echo_img.resolution_phase = img_resolution_phase
        self.multi_echo_img.resolution_slice = img_resolution_slice
        self.multi_echo_img.read_dir = read_dir
        self.multi_echo_img.acc_read = acc_read
        self.multi_echo_img.acc_factor_phase = acc_factor_phase
        self.multi_echo_img.etl = etl
        self.multi_echo_img.te = te
        self.multi_echo_img.os_factor = os_factor

    def set_navigator_params(
            self, lines_per_nav: int, num_of_nav: int, nav_acc_factor: int, nav_resolution_scaling: float,
            num_of_navs_per_tr: int, os_factor: int = 0
    ):
        self.navigator_img.n_read = int(self.multi_echo_img.n_read * nav_resolution_scaling)
        self.navigator_img.n_phase = int(self.multi_echo_img.n_phase * nav_resolution_scaling)
        self.navigator_img.n_slice = num_of_navs_per_tr
        self.navigator_img.resolution_read = self.multi_echo_img.n_read / nav_resolution_scaling
        self.navigator_img.resolution_phase = self.multi_echo_img.n_phase / nav_resolution_scaling
        self.navigator_img.resolution_slice = self.multi_echo_img.n_slice / nav_resolution_scaling
        self.navigator_img.read_dir = self.multi_echo_img.read_dir
        self.navigator_img.acc_read = False
        self.navigator_img.acc_factor_phase = nav_acc_factor
        self.navigator_img.lines_per_nav = lines_per_nav
        self.navigator_img.num_of_nav = num_of_nav
        self.navigator_img.nav_acc_factor = nav_acc_factor
        self.navigator_img.nav_resolution_scaling = nav_resolution_scaling
        self.navigator_img.etl = 1
        if os_factor > 0:
            self.navigator_img.os_factor = os_factor
        else:
            self.navigator_img.os_factor = self.multi_echo_img.os_factor
