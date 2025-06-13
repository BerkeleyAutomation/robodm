import os
import tempfile
from typing import Any, Dict, List

import numpy as np

from robodm.feature import FeatureType


def data_to_tf_schema(data: Dict[str, Any]) -> Dict[str, FeatureType]:
    """
    Convert data to a tf schema
    """
    data = _flatten(data)
    schema: Dict[str, Any] = {}
    for k, v in data.items():
        if "/" in k:  # make the subkey to be within dict
            main_key, sub_key = k.split("/")
            if main_key not in schema:
                schema[main_key] = {}
            schema[main_key][sub_key] = FeatureType.from_data(v).to_tf_feature_type(
                first_dim_none=True
            )
            # replace first element of shape with None
        else:
            schema[k] = FeatureType.from_data(v).to_tf_feature_type(first_dim_none=True)
    return schema


# flatten the data such that all data starts with root level tree (observation and action)
def _flatten(data, parent_key="", sep="/"):
    items = {}
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items

def _flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


import h5py


def recursively_read_hdf5_group(group):
    if isinstance(group, h5py.Dataset):
        return np.array(group)
    elif isinstance(group, h5py.Group):
        return {key: recursively_read_hdf5_group(value) for key, value in group.items()}
    else:
        raise TypeError("Unsupported HDF5 group type")
