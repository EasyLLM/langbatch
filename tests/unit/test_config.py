import json

config = None
with open("test_config.json", "r") as f:
    config = json.load(f)

def test_config():
    assert config is not None