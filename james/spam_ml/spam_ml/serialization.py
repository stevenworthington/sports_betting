from . import SPAM_DATA_ROOT

import os

model_data_prefix_dir = "models"
model_data_cache_path = os.path.join(SPAM_DATA_ROOT, model_data_prefix_dir)
os.makedirs(model_data_cache_path, exist_ok=True)


def generate_model_name():
    pass
