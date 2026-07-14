from importlib import import_module


_LAZY_IMPORTS = {
    "AdroitEnv": (".adroit", "AdroitEnv"),
    "DexArtEnv": (".dexart", "DexArtEnv"),
    "MetaWorldEnv": (".metaworld", "MetaWorldEnv"),
    "MetaWorldMultiViewEnv": (".metaworld", "MetaWorldMultiViewEnv"),
    "make_dmc_env": (".dmc", "make_dmc_env"),
    "make_dmc_env_2d": (".dmc", "make_dmc_env_2d"),
    "DMCEnv": (".dmc", "DMCEnv"),
    "UR5Env": (".ur5", "UR5Env"),
    "FrankaEnv": (".franka", "FrankaEnv"),
    "FrankaPourEnv": (".franka_pour", "FrankaPourEnv"),
    "FlippingEnv": (".flipping", "FlippingEnv"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
