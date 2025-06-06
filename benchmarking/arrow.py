import os
import sys
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as ft
from pyarrow import RecordBatchFileReader as Reader
from pyarrow import RecordBatchFileWriter as Writer

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")

PATH += "/datasets/"
NAME = "test_convert"
out_dir = PATH + "arrow_convert"


def pq_to_arrow(in_dir):
    table = pq.read_table(in_dir)
    file = in_dir.split("/")[-1].split("-")[0]

    with pa.OSFile(f"{out_dir}/REG/{file}.arrow", "wb") as sink:
        with Writer(sink, table.schema) as w:
            w.write_table(table)

    with pa.ipc.new_file(f"{out_dir}/IPC/{file}.arrow", table.schema) as w:
        w.write_table(table)

    ft.write_feather(table, f"{out_dir}/FTH/{file}.feather", "uncompressed")


N = 51
MB = 1024 * 1024

if False:
    for i in range(N):
        pq_to_arrow(f"{PATH}/{NAME}/{NAME}_{i}-0.parquet")

def measure_traj(read_func, write_func, name):
    read_time, write_time, data_size = 0, 0, 0

    for i in range(N):
        print(f"Measuring trajectory {i}")

        extn = "arrow" if (name[0] == "A") else "feather"
        path = f"{out_dir}/{name[-3:]}/{NAME}_{i}.{extn}"

        stop = time.time()
        traj = read_func(path)
        read_time += time.time() - stop

        data_size += os.path.getsize(path)
        path = f"{PATH}/temp.{extn}"

        stop = time.time()
        write_func(path, traj)
        write_time += time.time() - stop

        os.remove(path)
    return read_time, write_time, data_size / MB


if __name__ == "__main__":

    reg_dict = {"name": "Arrow_REG",
           "read_func": lambda path: Reader(pa.OSFile(path, "rb")).read_all(),
          "write_func": lambda path, data: Writer(pa.OSFile(path, "wb"), data.schema).write_table(data)
    }
    ipc_dict = {"name": "Arrow_IPC",
           "read_func": lambda path: pa.ipc.open_file(path).read_all(),
          "write_func": lambda path, data: pa.ipc.new_file(path, data.schema).write_table(data)
    }
    fth_dict = {"name": "Feather_FTH",
           "read_func": lambda path: ft.read_table(path),
          "write_func": lambda path, data: ft.write_feather(data, path)
    }
    pd_dict = {"name": "Pandas_FTH",
          "read_func": lambda path: pd.read_feather(path),
         "write_func": lambda path, data: data.to_feather(path)
    }

    for lib in [reg_dict, ipc_dict, fth_dict, pd_dict]:
        rt, wt, mb = measure_traj(lib["read_func"], lib["write_func"], lib["name"])

        print(f"\n{lib['name']}: \nDisk size = {mb:.4f} MB; Num. traj = {N}")
        print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
        print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")