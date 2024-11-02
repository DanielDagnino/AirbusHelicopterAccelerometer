import re
from typing import Optional


def _modify(config, modifiers):
    for k_mod, v_mod in modifiers.items():
        for k, v in config.items():
            if isinstance(v, str):
                match = re.match('^\{.*\}$', v)
                if match and match.group(0) == '{' + k_mod + '}':
                    config[k] = v_mod
                else:
                    # match = re.match('\{.*\}', v)
                    # if match and match.group(0) == '{' + k_mod + '}':
                    config[k] = v.replace("{" + k_mod + "}", str(v_mod))
            elif isinstance(v, dict):
                config[k] = dict_modifier(v, None, modifiers)


def dict_modifier(config: dict, modifiers: Optional[str], pre_modifiers: dict = None) -> dict:
    """

    TODO: It would be nice to allow extend this function to allow modifications in list.
        Example:
            n_hidden_a: ["{N_FEAT_IN}", "{N_H2}"]
            n_hidden_b: ["{N_H2}", 1]
        Right now, it is not possible. We can change the fills that are not inside lists.

    Args:
        config:
        modifiers:
        pre_modifiers:

    Returns:

    """
    # Apply modifiers to config.
    if modifiers is not None:
        modifiers = config.pop(modifiers, None)
        if modifiers is not None:
            _modify(config, modifiers)

    # Apply pre_modifier to modifier.
    if pre_modifiers is not None:
        _modify(config, pre_modifiers)

    return config
