import json


def load_config(path):
    return json.load(open(path))
