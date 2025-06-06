import os
import time
from logging import getLogger
import numpy as np
import pickle
import av
import pandas as pd
import pyarrow.feather as feather
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
import tensorflow as tf
import fastparquet as fp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

class BaseLoader():
    def __init__(self, data_path):
        super(BaseLoader, self).__init__()
        self.data_dir = data_path
        self.logger = getLogger(__name__)
        self.index = 0

    def __len__(self):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError


class BaseExporter():
    def __init__(self):
        super(BaseExporter, self).__init__()
        self.logger = getLogger(__name__)

    def export(self, loader: BaseLoader, output_path: str):
        raise NotImplementedError

class RTXLoader(BaseLoader):
    def __init__(self, data_path, split):
        super(RTXLoader, self).__init__(data_path)
        builder = tfds.builder_from_directory(data_path)
        self.ds = builder.as_dataset(split=split)

    def __len__(self):
        return len(self.ds)
    
    def __next__(self):
        if self.index < len(self.ds):
            self.index += 1
            nested_dataset = self.ds.__iter__()
            trajectory = list(nested_dataset)[0]["steps"]
            ret = []
            for step_data in trajectory:
                step = {}
                for dataset_key, element in step_data.items():
                    if dataset_key == "observation":
                        step["observation"] = {}
                        for obs_key, obs_element in element.items():
                            step["observation"][obs_key] = np.array(obs_element)
                    elif dataset_key == "action":
                        step["action"] = {}
                        for action_key, action_element in element.items():
                            step["action"][action_key] = np.array(action_element)
                    else:
                        step[dataset_key] = np.array(element)
                ret.append(step)
            return ret
        else:
            raise StopIteration
    
    def __iter__(self):
        return self

class RTXExporter(BaseExporter):
    def __init__(self):
        super(RTXExporter, self).__init__()

    def export(self, loader: BaseLoader, output_path: str):
        raise NotImplementedError


class ParquetExporter(BaseExporter):
    def __init__(self):
        super(ParquetExporter, self).__init__()
        self.logger = getLogger(__name__)

    def _serialize(self, data):
        return pickle.dumps(np.array(data))

    def _step_data_to_df(self, step_data):
        step = {}
        for dataset_key, element in step_data.items():
            if dataset_key == "observation":
                for obs_key, obs_element in element.items():
                    step[obs_key] = self._serialize(obs_element)
            elif dataset_key == "action":
                for action_key, action_element in element.items():
                    step[action_key] = self._serialize(action_element)
            else:
                step[dataset_key] = self._serialize(element)
        return step

    def export(self, loader: BaseLoader, output_path: str):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        i = 0
        for steps_data in loader:
            step_data_as_df = [self._step_data_to_df(step_data) for step_data in steps_data]
            combined_df = pd.DataFrame(step_data_as_df)
            output_file = os.path.join(output_path, f'output_{i}.parquet')
            fp.write(output_file, combined_df)
            print(f'Exported {output_file}')
            i += 1

class ParquetLoader(BaseLoader):
    def __init__(self, data_path):
        super(ParquetLoader, self).__init__(data_path)
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.parquet')]
        self.index = 0

    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            result = self._load_file(self.files[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration

    def _load_file(self, filename):
        table = pq.read_table(filename)
        df = table.to_pandas()
        steps = [dict(row) for _, row in df.iterrows()]
        return steps

class FeatherExporter(BaseExporter):
    def __init__(self):
        super(FeatherExporter, self).__init__()
        self.logger = getLogger(__name__)

    def _serialize(self, data):
        return pickle.dumps(np.array(data))

    def _step_data_to_df(self, step_data):
        step = {}
        for dataset_key, element in step_data.items():
            if dataset_key == "observation":
                for obs_key, obs_element in element.items():
                    step[obs_key] = self._serialize(obs_element)
            elif dataset_key == "action":
                for action_key, action_element in element.items():
                    step[action_key] = self._serialize(action_element)
            else:
                step[dataset_key] = self._serialize(element)
        return step

    def export(self, loader: BaseLoader, output_path: str):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        i = 0
        for steps_data in loader:
            step_data_as_df = [self._step_data_to_df(step_data) for step_data in steps_data]
            combined_df = pd.DataFrame(step_data_as_df)
            output_file = os.path.join(output_path, f'output_{i}.feather')
            feather.write_feather(combined_df, output_file)
            print(f'Exported {output_file}')
            i += 1

class FeatherLoader(BaseLoader):
    def __init__(self, data_path):
        super(FeatherLoader, self).__init__(data_path)
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.feather')]
        self.index = 0

    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            result = self._load_file(self.files[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration

    def _load_file(self, filename):
        df = feather.read_feather(filename)
        steps = [dict(row) for _, row in df.iterrows()]
        return steps

def iterate_parquet(rtx_loader, parquet_exporter, n):
    read_time, write_time, data_size = 0, 0, 0
    path = os.path.expanduser("~/fog_x_fork/examples/dataloader/pq_output/")
    os.makedirs(path, exist_ok=True)

    stop = time.time()
    parquet_exporter.export(rtx_loader, path)
    write_time += time.time() - stop

    parquet_loader = ParquetLoader(path)

    stop = time.time()
    for i, data in enumerate(parquet_loader):
        print(f"Reading trajectory {i}")
        if i == n: break
    read_time += time.time() - stop

    for i in range(n):
        data_size += os.path.getsize(f"{path}/output_{i}.parquet")

    return read_time, write_time, data_size / 1024**2

def iterate_feather(rtx_loader, feather_exporter, n):
    read_time, write_time, data_size = 0, 0, 0
    path = os.path.expanduser("~/fog_x/examples/dataloader/feather_output/")
    os.makedirs(path, exist_ok=True)

    stop = time.time()
    feather_exporter.export(rtx_loader, path)
    write_time += time.time() - stop

    feather_loader = FeatherLoader(path)

    stop = time.time()
    for i, data in enumerate(feather_loader):
        print(f"Reading trajectory {i}")
        if i == n: break
    read_time += time.time() - stop

    for i in range(n):
        data_size += os.path.getsize(f"{path}/output_{i}.feather")

    return read_time, write_time, data_size / 1024**2

# parquet use case 
rtx_loader = RTXLoader(os.path.expanduser("~/datasets/berkeley_autolab_ur5/0.1.0"), split='train[:51]')
parquet_exporter = ParquetExporter()
rt, wt, mb = iterate_parquet(rtx_loader, parquet_exporter, 51)

print(f"\nParquet Data: \nMem. size = {mb:.4f} MB; Num. traj = 51")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {51 / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {51 / wt :.4f} traj/s")

# feather pyarrow version use case
feather_exporter = FeatherExporter()
rt, wt, mb = iterate_feather(rtx_loader, feather_exporter, 51)

print(f"\nFeather Data: \nMem. size = {mb:.4f} MB; Num. traj = 51")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {51 / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {51 / wt :.4f} traj/s")
