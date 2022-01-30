import functools
import inspect
import logging

import kvt

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += (
            f"(name={self._name}, items={list(self._obj_dict.keys())})"
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def obj_dict(self):
        return self._obj_dict

    def get(self, key: str):
        return self._obj_dict.get(key, None)

    def register(self, obj):
        """Register a callable object.

        Args:
            obj: callable object to be registered
        """
        if not callable(obj):
            raise ValueError(f"object must be callable")

        obj_name = obj.__name__
        self._obj_dict[obj_name] = obj
        return obj


def build_from_config(
    config, registry, default_args=None, match_object_args=False
):
    """Build a callable object from configuation dict.

    Args:
        config (dict): Configuration dict. It should contain the key "name".
        registry (:obj:`Registry`): The registry to search the name from.
        default_args (dict, optional): Default initialization argments.
    """
    if config is None:
        return None

    assert isinstance(config, dict) and "name" in config
    assert isinstance(default_args, dict) or default_args is None

    name = config["name"]
    name = name.replace("-", "_")
    obj = registry.get(name)
    if obj is None:
        raise KeyError(f"{name} is not in the {registry.name} registry")

    logger.info(f"Loaded {name} path: {inspect.getfile(obj)}")

    args = dict()
    if ("params" in config) and (config["params"] is not None):
        args.update(config["params"])
    if default_args is not None:
        args.update(default_args)

    if match_object_args:
        if inspect.isclass(obj):
            obj_args = inspect.getfullargspec(obj.__init__).args
        else:
            obj_args = inspect.getfullargspec(obj).args
        valid_args = set(args.keys()) & set(obj_args)
        invalid_args = set(args.keys()) - set(obj_args)
        args = {k: v for k, v in args.items() if k in valid_args}
        if len(invalid_args):
            logger.info(f"Ignore args: {invalid_args}")

    if (name in kvt.registry.METRICS._obj_dict.keys()) and (
        inspect.isfunction(obj)
    ):
        o = functools.partial(obj, **args)
    else:
        o = obj(**args)

    return o
