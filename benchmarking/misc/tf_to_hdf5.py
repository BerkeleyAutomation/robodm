import os
import sys
import tensorflow as tf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# NOTE: this code assumes you have the paths: home/username/datasets/berkeley_autolab_ur5/TRAIN.tfrecord

def parse_tfrecord(raw_record):
    FLF = tf.io.FixedLenFeature

    features = {'steps': tf.io.VarLenFeature(tf.string)}
    contexts = tf.io.parse_single_example(raw_record, features)
    steps = tf.sparse.to_dense(contexts['steps'])

    steps_features = {
        'is_last':                                  FLF([], tf.bool),
        'reward':                                   FLF([], tf.float32),
        'is_first':                                 FLF([], tf.bool),
        'action/gripper_closedness_action':         FLF([], tf.float32),
        'action/rotation_delta':                    FLF([3], tf.float32),
        'action/terminate_episode':                 FLF([], tf.float32),
        'action/world_vector':                      FLF([3], tf.float32),
        'observation/natural_language_embedding':   FLF([512], tf.float32),
        'observation/image':                        FLF([480, 640, 3], tf.uint8),
        'observation/hand_image':                   FLF([480, 640, 3], tf.uint8),
        'observation/natural_language_instruction': FLF([], tf.string),
        'observation/image_with_depth':             FLF([480, 640, 1], tf.float32),
        'observation/robot_state':                  FLF([15], tf.float32),
        'is_terminal':                              FLF([], tf.bool),
    }
    return [tf.io.parse_single_example(s, steps_features) for s in steps ]


def tfrecord_to_df(tf_file):
    dataset = tf.data.TFRecordDataset(tf_file)
    all_records = []

    for raw_record in dataset:
        for step in parse_tfrecord(raw_record):
            record = {}
            for key, value in step.items():
                record[key] = value.numpy()
            all_records.append(record)
    
    return pd.DataFrame(all_records)


def tf_to_parquet(in_dir, out_dir, prefix, suffix):
    for i in range(1):
        j = str(i).zfill(5)

        input_file  = f"{in_dir}{prefix}{j}-of-00412"
        output_file = f"{out_dir}{prefix}{j}{suffix}"
        
        print(f"Reading {input_file}")
        
        df = tfrecord_to_df(input_file)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        
        print(f"Writing {output_file}")


DIR = os.path.expanduser("~") + "/datasets/"
in_dir  = DIR + "berkeley_autolab_ur5/"
out_dir = DIR + "parquet/"
prefix  = "berkeley_autolab_ur5-train.tfrecord-"
suffix  = ".parquet"

tf_to_parquet(in_dir, out_dir, prefix, suffix)


def disk_size(file):
    return os.path.getsize(file)

def tf_memory(tf_file):
    dataset = tf.data.TFRecordDataset(tf_file)
    return sys.getsizeof([record.numpy() for record in dataset])

def pq_memory(pq_file):
    return pq.read_table(pq_file).nbytes

MB = 1024 * 1024
tf_file = in_dir + prefix + "00000-of-00412"
pq_file = out_dir + prefix + "00000.parquet"

print(f"TFRecord size on disk: {disk_size(tf_file) / MB:.2f} MB")
print(f"Parquet  size on disk: {disk_size(pq_file) / MB:.2f} MB")
print(f"TFRecord size in memory: {tf_memory(tf_file) / MB:.2f} MB")
print(f"Parquet  size in memory: {pq_memory(pq_file) / MB:.2f} MB")