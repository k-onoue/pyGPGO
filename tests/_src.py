import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
sys.path.append(PROJECT_DIR)

from src.covfunc import squaredExponential, matern32, matern52
from src.surrogates.gp import GaussianProcess, GaussianProcessMCMC
from src.surrogates.stp import tStudentProcess, tStudentProcessMCMC


__all__ = [
    "squaredExponential",
    "matern32",
    "matern52",
    "GaussianProcess",
    "GaussianProcessMCMC",
    "tStudentProcess",
    "tStudentProcessMCMC",
]
