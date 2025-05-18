from omegaconf import OmegaConf

OmegaConf.register_new_resolver("tuple_int", lambda *args: tuple(map(int, args)))
OmegaConf.register_new_resolver("tuple_bool", lambda *args: tuple(map(bool, args)))
