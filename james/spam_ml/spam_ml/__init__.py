import os

SPAM_DATA_ROOT = os.getenv("SPAM_DATA_ROOT")
if SPAM_DATA_ROOT is None:
    SPAM_DATA_ROOT = os.path.join(os.path.expanduser("~"), ".spam_cache")
