import os
import sys
import time
import polars as pl
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")
from fog_x.dataset import Dataset


PATH += "/datasets/"
NAME = "test_convert"
dataset = Dataset(name=NAME, path=PATH)

N = 51
MB = 1024 * 1024
SDC = dataset.db_manager.step_data_connector


def measure_traj(read_func, write_func):
    read_time, write_time, data_size = 0, 0, 0

    for i in range(N):
        print(f"Measuring trajectory {i}")
        path = f"{SDC.path}/{NAME}_{i}-0.parquet"

        stop = time.time()
        traj = read_func(path)
        read_time += time.time() - stop

        data_size += os.path.getsize(path)
        path = PATH + "/temp.parquet"

        stop = time.time()
        write_func(path, traj)
        write_time += time.time() - stop

        os.remove(path)
    return read_time, write_time, data_size / MB


if __name__ == "__main__":

    fx_dict = {"name": "Fog-X",
        "read_func": lambda path: SDC.tables[path.split("/")[-1].split("-")[0]].collect(),
        "write_func": lambda path, data: data.write_parquet(path)
    }
    pl_dict = {"name": "Polars",
        "read_func": lambda path: pl.read_parquet(path),
        "write_func": lambda path, data: data.write_parquet(path)
    }
    pq_dict = {"name": "PyArrow",
        "read_func": lambda path: pl.from_arrow(pq.read_table(path)),
        "write_func": lambda path, data: pq.write_table(data.to_arrow(), path)
    }
    pd_dict = {"name": "Pandas",
        "read_func": lambda path: pd.read_parquet(path),
        "write_func": lambda path, data: data.to_parquet(path)
    }

    for lib in [fx_dict, pl_dict, pq_dict, pd_dict]:
        rt, wt, mb = measure_traj(lib["read_func"], lib["write_func"])

        print(f"\n{lib['name']}: \nDisk size = {mb:.4f} MB; Num. traj = {N}")
        print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
        print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")