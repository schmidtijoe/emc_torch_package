import torch


class Loss:
    def __init__(self, name: str = "", value: torch.tensor = torch.zeros(1), emphasis: float = 1.0):
        self.name: str = name
        self.value: torch.tensor = value
        self.emphasis: float = emphasis

    def calculate(self, input_tensor: torch.tensor, target_tensor: torch.tensor):
        return NotImplementedError

    def get_loss(self):
        return self.emphasis * self.value


class ShapeLoss(Loss):
    def __init__(self, emphasis: float = 1.0):
        super().__init__(name="ShapeLoss", emphasis=emphasis)

    def calculate(self, input_tensor: torch.tensor, target_tensor: torch.tensor):
        target_mag = target_tensor[0]
        target_phase = target_tensor[1]
        target_z = target_tensor[2]
        # average over all dims except b1
        while input_tensor.shape.__len__() > 3:
            input_tensor = torch.mean(input_tensor, dim=0)
        # calculate error - want to get MSE loss across all slice profiles - magnitude, phase and z (b1 in first dim)
        # and then sum
        # ToDo: Emphasize shape in loss function!
        self.value = 10 * (torch.sum(torch.nn.MSELoss()(torch.norm(input_tensor[:, :, :2], dim=-1), target_mag))) + \
                     torch.sum(torch.nn.MSELoss()(input_tensor[:, :, 2], target_z)) + \
                     torch.sum(
                         torch.nn.MSELoss()(torch.angle(input_tensor[:, :, 0] + 1j * input_tensor[:, :, 1]),
                                            target_phase))


class SmoothenessLoss(Loss):
    def __init__(self, emphasis: float = 1.0, name: str = ""):
        super().__init__(name=f"SmoothenessLoss{name}", emphasis=emphasis)

    def calculate(self, input_tensor: torch.tensor, target_tensor: torch.tensor):
        self.value = torch.sum(torch.abs(torch.gradient(input_tensor, dim=-1)[0]))


class RampLoss(Loss):
    def __init__(self, emphasis: float = 1.0, name: str = ""):
        super().__init__(name=f"RampLoss{name}", emphasis=emphasis)

    def calculate(self, input_tensor: torch.tensor, target_tensor: torch.tensor):
        if input_tensor.shape.__len__() < 2:
            input_tensor = input_tensor[None, :]
        # minimize first and last entry
        self.value = torch.sum(input_tensor[:, 0] ** 2 + input_tensor[:, -1] ** 2)


class PowerAmpLoss(Loss):
    def __init__(self, emphasis: float = 1.0, name: str = ""):
        super().__init__(name=f"AmplitudeLoss{name}", emphasis=emphasis)

    def calculate(self, input_tensor: torch.tensor, target_tensor: torch.tensor):
        if input_tensor.shape.__len__() < 2:
            input_tensor = input_tensor[None, :]
        self.value = torch.sum(input_tensor ** 2) / input_tensor.shape[0]


class LossOptimizer:
    def __init__(self,
                 lambda_shape: float = 1.0, lambda_smootheness_p: float = 1.0, lambda_smootheness_g: float = 1.0,
                 lambda_ramp_p: float = 1.0, lambda_ramp_g: float = 1.0, lambda_power_p: float = 1.0,
                 lambda_amp_g: float = 1.0
                 ):
        self.value: torch.tensor = torch.zeros(0)
        # define losses
        self.shape_loss = ShapeLoss(emphasis=lambda_shape)
        # smootheness
        self.smootheness_loss = SmoothenessLoss()
        # pulse
        self.smootheness_loss_px = SmoothenessLoss(name="PX", emphasis=lambda_smootheness_p)
        self.smootheness_loss_py = SmoothenessLoss(name="PY", emphasis=lambda_smootheness_p)
        # gradients
        self.smootheness_loss_g = SmoothenessLoss(name="G", emphasis=lambda_smootheness_g)
        self.smootheness_loss_gr = SmoothenessLoss(name="GR", emphasis=lambda_smootheness_g)
        # ramps
        self.ramp_loss = RampLoss()
        # pulse
        self.ramp_loss_px = RampLoss(name="PX", emphasis=lambda_ramp_p)
        self.ramp_loss_py = RampLoss(name="PY", emphasis=lambda_ramp_p)
        # gradients
        self.ramp_loss_g = RampLoss(name="G", emphasis=lambda_ramp_g)
        self.ramp_loss_gr = RampLoss(name="GR", emphasis=lambda_ramp_g)
        # amplitudes
        self.amp_loss = PowerAmpLoss()
        # power pulse
        self.amp_loss_px = PowerAmpLoss(name="PX", emphasis=lambda_power_p)
        self.amp_loss_py = PowerAmpLoss(name="PY", emphasis=lambda_power_p)
        # amplitudes grad
        self.amp_loss_g = PowerAmpLoss(name="G", emphasis=lambda_amp_g)
        self.amp_loss_gr = PowerAmpLoss(name="GR", emphasis=lambda_amp_g)

    def get_registered_loss_dict(self) -> dict:
        d = {
            "loss": self.value,
            self.shape_loss.name: self.shape_loss.value,
            self.smootheness_loss.name: self.smootheness_loss.value,
            "smootheness_loss_p": self.smootheness_loss_px.value + self.smootheness_loss_py.value,
            "smootheness_loss_g": self.smootheness_loss_g.value + self.smootheness_loss_gr.value,
            "ramp_loss_p": self.ramp_loss_px.value + self.ramp_loss_py.value,
            "ramp_loss_g": self.ramp_loss_g.value + self.ramp_loss_gr.value,
            "amp_loss_p": self.amp_loss_px.value + self.amp_loss_py.value,
            "amp_loss_g": self.amp_loss_g.value + self.amp_loss_gr.value,
        }
        return d

    def get_loss(self):
        return self.value

    def calculate_smootheness_loss(self, p_x, p_y, g, g_r):
        self.smootheness_loss_px.calculate(p_x, p_x)
        self.smootheness_loss_py.calculate(p_y, p_y)
        self.smootheness_loss_g.calculate(g, g)
        self.smootheness_loss_gr.calculate(g_r, g_r)
        # combined
        self.smootheness_loss.value = self.smootheness_loss_px.get_loss() + self.smootheness_loss_py.get_loss()
        self.smootheness_loss.value += self.smootheness_loss_g.get_loss() + self.smootheness_loss_gr.get_loss()

    def calculate_ramp_loss(self, p_x, p_y, g, g_r):
        self.ramp_loss_px.calculate(p_x, p_x)
        self.ramp_loss_py.calculate(p_y, p_y)
        self.ramp_loss_g.calculate(g, g)
        self.ramp_loss_gr.calculate(g_r, g_r)
        # combined
        self.ramp_loss.value = self.ramp_loss_px.get_loss() + self.ramp_loss_py.get_loss()
        self.ramp_loss.value += self.ramp_loss_g.get_loss() + self.ramp_loss_gr.get_loss()

    def calculate_amp_loss(self, p_x, p_y, g, g_r):
        self.amp_loss_px.calculate(p_x, p_x)
        self.amp_loss_py.calculate(p_y, p_y)
        self.amp_loss_g.calculate(g, g)
        self.amp_loss_gr.calculate(g_r, g_r)
        # combined
        self.amp_loss.value = self.amp_loss_px.get_loss() + self.amp_loss_py.get_loss()
        self.amp_loss.value += self.amp_loss_g.get_loss() + self.amp_loss_gr.get_loss()

    def calculate_loss(self, magnetization: torch.tensor, target_profile: torch.tensor,
                       p_x: torch.tensor, p_y: torch.tensor, g: torch.tensor, g_r: torch.tensor):
        # magnetization is tensor [n_t2, n_b1, n_samples, 4]
        # target is tensor [n_samples, 4]
        # want individual b1 profiles to match target as closely as possible
        self.shape_loss.calculate(input_tensor=magnetization, target_tensor=target_profile)

        # amplitudes
        self.calculate_amp_loss(p_x, p_y, g, g_r)

        # ToDo: emphasize pulse smoothness over grad!
        self.calculate_smootheness_loss(p_x, p_y, g, g_r)

        # enforce easy ramps
        self.calculate_ramp_loss(p_x, p_y, g, g_r)

        self.value = self.shape_loss.get_loss() + self.smootheness_loss.get_loss()
