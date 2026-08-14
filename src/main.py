from src.utils.logger import log
from src.utils.seed import set_seed
from src.config import DEVICE

set_seed()

log("Drug-Microbe Interaction Prediction")

log(f"Running on {DEVICE}")