# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
from util.dist import sync_tensor

import os
import yaml


from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["init_modules", "zero_last_gamma", "AverageMeter", "parse_with_yaml",
            "parse_unknown_args",
            "partial_update_config",
            "resolve_and_load_config",
            "load_config",
            "dump_config",
           ]


def init_modules(model: nn.Module or list[nn.Module], init_type="trunc_normal") -> None:
    _DEFAULT_INIT_PARAM = {"trunc_normal": 0.02}

    if isinstance(model, list):
        for sub_module in model:
            init_modules(sub_module, init_type)
    else:
        init_params = init_type.split("@")
        init_params = float(init_params[1]) if len(init_params) > 1 else None

        if init_type.startswith("trunc_normal"):
            init_func = lambda param: nn.init.trunc_normal_(
                param, std=(init_params or _DEFAULT_INIT_PARAM["trunc_normal"])
            )
        else:
            raise NotImplementedError

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                init_func(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                init_func(m.weight)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                weight = getattr(m, "weight", None)
                bias = getattr(m, "bias", None)
                if isinstance(weight, torch.nn.Parameter):
                    init_func(weight)
                if isinstance(bias, torch.nn.Parameter):
                    bias.data.zero_()


def zero_last_gamma(model: nn.Module, init_val=0) -> None:
    import models.nn.ops as ops

    for m in model.modules():
        if isinstance(m, ops.ResidualBlock) and isinstance(m.shortcut, ops.IdentityLayer):
            if isinstance(m.main, (ops.DSConv, ops.MBConv, ops.FusedMBConv)):
                parent_module = m.main.point_conv
            elif isinstance(m.main, ops.ResBlock):
                parent_module = m.main.conv2
            elif isinstance(m.main, ops.ConvLayer):
                parent_module = m.main
            elif isinstance(m.main, (ops.LiteMLA)):
                parent_module = m.main.proj
            else:
                parent_module = None
            if parent_module is not None:
                norm = getattr(parent_module, "norm", None)
                if norm is not None:
                    nn.init.constant_(norm.weight, init_val)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: torch.Tensor or int or float) -> torch.Tensor or int or float:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg


# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

def parse_with_yaml(config_str: str) -> str or dict:
    try:
        # add space manually for dict
        if "{" in config_str and "}" in config_str and ":" in config_str:
            out_str = config_str.replace(":", ": ")
        else:
            out_str = config_str
        return yaml.safe_load(out_str)
    except ValueError:
        # return raw string if parsing fails
        return config_str


def parse_unknown_args(unknown: list) -> dict:
    """Parse unknown args."""
    index = 0
    parsed_dict = {}
    while index < len(unknown):
        key, val = unknown[index], unknown[index + 1]
        index += 2
        if not key.startswith("--"):
            continue
        key = key[2:]

        # try parsing with either dot notation or full yaml notation
        # Note that the vanilla case "--key value" will be parsed the same
        if "." in key:
            # key == a.b.c, val == val --> parsed_dict[a][b][c] = val
            keys = key.split(".")
            dict_to_update = parsed_dict
            for key in keys[:-1]:
                if not (key in dict_to_update and isinstance(dict_to_update[key], dict)):
                    dict_to_update[key] = {}
                dict_to_update = dict_to_update[key]
            dict_to_update[keys[-1]] = parse_with_yaml(val)  # so we can parse lists, bools, etc...
        else:
            parsed_dict[key] = parse_with_yaml(val)
    return parsed_dict


def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        if key in config and isinstance(partial_config[key], dict) and isinstance(config[key], dict):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]
    return config


def resolve_and_load_config(path: str, config_name="config.yaml") -> dict:
    path = os.path.realpath(os.path.expanduser(path))
    if os.path.isdir(path):
        config_path = os.path.join(path, config_name)
    else:
        config_path = path
    if os.path.isfile(config_path):
        pass
    else:
        raise Exception(f"Cannot find a valid config at {path}")
    config = load_config(config_path)
    return config


class SafeLoaderWithTuple(yaml.SafeLoader):
    """A yaml safe loader with python tuple loading capabilities."""

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoaderWithTuple.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderWithTuple.construct_python_tuple)


def load_config(filename: str) -> dict:
    """Load a yaml file."""
    filename = os.path.realpath(os.path.expanduser(filename))
    return yaml.load(open(filename), Loader=SafeLoaderWithTuple)


def dump_config(config: dict, filename: str) -> None:
    """Dump a config file"""
    filename = os.path.realpath(os.path.expanduser(filename))
    yaml.dump(config, open(filename, "w"), sort_keys=False)