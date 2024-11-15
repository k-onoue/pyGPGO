import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
sys.path.append(PROJECT_DIR)

from src.config import ModelConfig
from src.covfunc import CovarianceFunction
from src.surrogates.gp import GaussianProcess
from src.surrogates.stp import tStudentProcess
from src.test_functions import SinusoidalSynthetic, BraninHoo, Hartmann6
from src.acquisition import *
# from surrogates.gp import GaussianProcessMCMC
# from src.surrogates.stp import tStudentProcess, tStudentProcessMCMC


__all__ = [
    "ModelConfig",
    "CovarianceFunction",
    "GaussianProcess",
    "tStudentProcess",
    "SinusoidalSynthetic",
    "BraninHoo",
    "Hartmann6",
    # "GaussianProcessMCMC",
    # "tStudentProcess",
    # "tStudentProcessMCMC",
]
