from .linear_quadratic_regulator import LQR
from . import value_networks
from . import gradient_networks
from . import control_networks
from .utilities import load_NN, create_NN
from .model_factory import available_models, register

register('LQR', LQR)
register('ValueNN', value_networks.ValueNN)
register('ValueQRnet', value_networks.ValueQRnet)

register('GradientNN', gradient_networks.GradientNN)
register('GradientQRnet', gradient_networks.GradientQRnet)
register('GradientJacQRnet', gradient_networks.GradientJacQRnet)
register('GradientMatQRnet', gradient_networks.GradientMatQRnet)

register('ControlNN', control_networks.ControlNN)
register('ControlQRnet', control_networks.ControlQRnet)
register('ControlJacQRnet', control_networks.ControlJacQRnet)
register('ControlMatQRnet', control_networks.ControlMatQRnet)
