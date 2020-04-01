import yaml
import os

try:
    f = open("config.yaml")
    config = yaml.load(f)
except Exception as e:
    import logging
    logging.exception(e)
    config = {}
