#These loss can be used for both In-training Calibration and Post-hoc Calibration
from .Soft_binned_ECE import soft_binning_ece
from .AvUS import get_avuc_loss
from .Soft_AvUS import get_soft_avuc_loss
from .DECE import DECE
