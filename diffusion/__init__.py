# from pathlib import Path
from pydoc import locate

from omegaconf import OmegaConf


OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)


def _resolver_exact_int_div(a, b):
    div = a // b
    assert div * b == a, f"exact_int_div resolver: {a} cannot be exactly divided by {b}."
    return div


OmegaConf.register_new_resolver("exact_int_div", _resolver_exact_int_div)
OmegaConf.register_new_resolver("if", lambda a, b, c: b if a else c)
OmegaConf.register_new_resolver("locate", lambda name: locate(name))
OmegaConf.register_new_resolver("int", lambda s: int(s))


def _resolver_call(func, kwargs):
    return func(**kwargs)


OmegaConf.register_new_resolver("call", _resolver_call)
