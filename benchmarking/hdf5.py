import os
import sys
import time
import h5py
import numpy as np
import pyarrow.parquet as pq

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")

PATH += "/datasets/"
NAME = "test_convert"
out_dir = PATH + "hdf5_convert"


def pq_to_hdf5(in_dir):
    table = pq.read_table(in_dir)
    file = in_dir.split("/")[-1].split("-")[0]
    
    with h5py.File(f"{out_dir}/{file}.hdf5", "w") as hdf5:
        for name in table.column_names:
            data = table[name].to_numpy()
            dtype = data.dtype
            try:
                hdf5.create_dataset(name, data=data, dtype=dtype, compression=None)
            except Exception:
                dtype = h5py.string_dtype("utf-8")

                data = np.array([str(x) for x in data], dtype=dtype)
                hdf5.create_dataset(name, data=data, dtype=dtype, compression=None)

N = 51
MB = 1024 * 1024

if False:
    for i in range(N):
        pq_to_hdf5(f"{PATH}{NAME}/{NAME}_{i}-0.parquet")

def measure_traj(read_func, write_func):
    read_time, write_time, data_size = 0, 0, 0

    for i in range(N):
        print(f"Measuring trajectory {i}")
        path = f"{out_dir}/{NAME}_{i}.hdf5"

        stop = time.time()
        traj = read_func(path)
        read_time += time.time() - stop

        data_size += os.path.getsize(path)
        path = PATH + "/temp.hdf5"

        stop = time.time()
        write_func(path, traj)
        write_time += time.time() - stop

        os.remove(path)
    return read_time, write_time, data_size / MB


if __name__ == "__main__":

    def read_h5(path):
        data = {}
        with h5py.File(path, "r") as file:
            for key in file.keys():
                if isinstance(file[key], h5py.Dataset):
                    data[key] = file[key][:]
        return data
    
    def write_h5(path, data):
        with h5py.File(path, "w") as file:
            for name, dset in data.items():
                file.create_dataset(name, data=dset)

    rt, wt, mb = measure_traj(read_h5, write_h5)

    print(f"\nHDF5: \nDisk size = {mb:.4f} MB; Num. traj = {N}")
    print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
    print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")