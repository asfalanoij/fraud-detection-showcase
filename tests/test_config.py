from fraud.config import load_config


def test_load_config_parses_cost():
    cfg = load_config("config/config.yaml")
    assert cfg.seed == 42
    assert cfg.cost.fp == 4.0
    assert cfg.split.train == 0.8
