
import pytest
from src.options import NetworkOptions
import logging
import wandb

wandb.init(project="glaucoma", entity="sudonuma")

@pytest.fixture
def logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    return logger


@pytest.fixture
def test_options():
    options = NetworkOptions()
    opts = options.parse()
    return opts