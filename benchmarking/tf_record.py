import os
import sys
import time
import numpy as np

PATH = "/shared/ryanhoque/datasets/datasets/"
os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf

PATH += "/datasets/"
NAME = "berkeley_autolab_ur5"
MB = 1024 * 1024
N = 51

def measure_traj():
    read_time, write_time, disk_size, memr_size = 0, 0, 0, 0

    for i in range(N):
        print(f"Measuring trajectory {i}")
        path = f"{PATH}/{NAME}/{NAME}-train.tfrecord-{str(i).zfill(5)}-of-00412"

        TFRD = tf.data.TFRecordDataset
        stop = time.time()

        for record in TFRD(path):
            data = record.numpy()
            memr_size += len(data)
        read_time += time.time() - stop

        disk_size += os.path.getsize(path)
        out_path = PATH + "/temp.tfrecord"

        for record in TFRD(path):
            data = record.numpy()
            stop = time.time()

            with tf.io.TFRecordWriter(out_path) as w:
                w.write(data)
            write_time += time.time() - stop
            os.remove(out_path)

    return read_time, write_time, disk_size / MB, memr_size / MB


if __name__ == "__main__":
    rt, wt, ds, ms = measure_traj()

    print(f"\nTF_Record: \nDisk size = {ds:.4f} MB; Mem. size = {ms:.4f} MB; Num. traj = {N}")
    print(f"Read:  latency = {rt:.4f} s; throughput = {ds / rt :.4f} MB/s, {N / rt :.4f} traj/s")
    print(f"Write: latency = {wt:.4f} s; throughput = {ds / wt :.4f} MB/s, {N / wt :.4f} traj/s")