import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./../config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
sys.path.append(PROJECT_DIR)

from src.config import ModelConfig
from src.covfunc import CovarianceFunction
from src.surrogates.gp import GaussianProcess
from src.surrogates.stp import tStudentProcess
# from surrogates.gp import GaussianProcessMCMC
# from src.surrogates.stp import tStudentProcess, tStudentProcessMCMC


__all__ = [
    "ModelConfig",
    "CovarianceFunction",
    "GaussianProcess",
    "tStudentProcess",
    # "GaussianProcessMCMC",
    # "tStudentProcess",
    # "tStudentProcessMCMC",
]
