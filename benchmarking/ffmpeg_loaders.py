import os
import av
import sys
import time
import pickle
import numpy as np
from logging import getLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow_datasets as tfds
import tensorflow as tf

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")
PATH += "/datasets/"


class BaseLoader():
    def __init__(self, path):
        super(BaseLoader, self).__init__()
        self.path = path
        self.logger = getLogger(__name__)

    def __len__(self):
        raise NotImplementedError

    def __iter___(self):
        raise NotImplementedError


class BaseWriter():
    def __init__(self):
        super(BaseWriter, self).__init__()
        self.logger = getLogger(__name__)

    def export(self, loader, path):
        raise NotImplementedError


class RTXLoader(BaseLoader):

    def __init__(self, path, split):
        super(RTXLoader, self).__init__(path)
        builder = tfds.builder_from_directory(path)
        self.ds = builder.as_dataset(split)
        self.index = 0

    def __len__(self):
        return len(self.ds)
    
    def __next__(self):
        if self.index < len(self.ds):
            self.index += 1
            nest_ds = self.ds.__iter__()
            traj = list(nest_ds)[0]["steps"]
            data = []

            for step_data in traj:
                step = {}
                for key, val in step_data.items():

                    if key == "observation":
                        step["observation"] = {}
                        for obs_key, obs_val in val.items():
                            step["observation"][obs_key] = np.array(obs_val)

                    elif key == "action":
                        step["action"] = {}
                        for act_key, act_val in val.items():
                            step["action"][act_key] = np.array(act_val)
                    else:
                        step[key] = np.array(val)

                data.append(step)
            return data
        else:
            self.index = 0
            raise StopIteration
    
    def __iter__(self):
        return self


class RTXWriter(BaseWriter):

    def __init__(self):
        super(RTXWriter, self).__init__()
        self.wr = tf.io.TFRecordWriter

    def export(self, loader, path):
        for i, data in enumerate(loader):
            # data_list = list(data["steps"].as_numpy_iterator())
            # data_flat = flatten(data_list)

            with self.wr(f"{path}/output_{i}.tfrecord") as writer:
                for dl in data:
                    writer.write(dl)

    def export_temp(self, data, path):
        with self.wr(path + "/temp.tfrecord") as writer:
            for dl in data:
                writer.write(dl)


N = 51
MB = 1024 * 1024

def flatten(data):
    if isinstance(data, dict):
        data = list(data.values())
    if isinstance(data, list):
        return [x for lst in data for x in flatten(lst)]
    if isinstance(data, bytes):
        return [data]
    return [data.tobytes()]


def iterate_rtx(num_iters):
    read_time, write_time, data_size = 0, 0, 0

    # Read time
    stop = time.time()
    rtx_loader = RTXLoader(PATH + "berkeley_autolab_ur5", f"train[:{N}]")

    for i, data in enumerate(rtx_loader):
        print(f"Reading trajectory {i}")
        if i == num_iters:
            break
    read_time = time.time() - stop

    # Write time
    rtx_writer = RTXWriter()
    for i, data in enumerate(rtx_loader):
        print(f"Writing trajectory {i}")
        flat = flatten(data)

        stop = time.time()
        rtx_writer.export_temp(flat, PATH)
        write_time += time.time() - stop
        if i == num_iters:
            break

    # Data size
    for i in range(num_iters):
        name = "berkeley_autolab_ur5"
        num = f"{str(i).zfill(5)}-of-00412"
        data_size += os.path.getsize(f"{PATH}/{name}/{name}-train.tfrecord-{num}")

    return read_time, write_time, data_size / MB


rt, wt, mb = iterate_rtx(N)
print(f"\nRTX Data: \nMem. size = {mb:.4f} MB; Num. traj = {N}")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")


class MKVLoader(BaseLoader):

    def __init__(self, path):
        super(MKVLoader, self).__init__(path)
        self.files = [path + f for f in os.listdir(path) if f.endswith(".mkv")]
        self.index = 0

    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self):
            result = self.files[self.index]
            self.index += 1
            return self._parse_mkv_file(result)
        else:
            raise StopIteration
    
    def _parse_mkv_file(self, file):
        inp_cont = av.open(file)
        print(file)

        vid_str1 = inp_cont.streams.video[0]
        vid_str1.thread_type = "AUTO"
        vid_str2 = inp_cont.streams.video[1]
        vid_str2.thread_type = "AUTO"
        dpth_str  = inp_cont.streams.video[2]
        dpth_str.thread_type = "AUTO"
        data_str = inp_cont.streams[3]

        dec_str1, dec_str2, dec_dpth, dec_data = [], [], [], []

        pkt_counter = 0
        for packet in inp_cont.demux(vid_str1, vid_str2, dpth_str, data_str):
            pkt_counter += 1
            if packet.stream.index == vid_str1.index: 
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format="rgb24")
                        dec_str1.append(image)
            elif packet.stream.index == vid_str2.index:
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format="rgb24")
                        dec_str2.append(image)
            elif packet.stream.index == dpth_str.index:
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format="gray")
                        dec_dpth.append(image)
            elif packet.stream.index == data_str.index:
                pib = bytes(packet)
                if pib:
                    non_dict = pickle.loads(pib)
                    dec_data.append(non_dict)
            else:
                print("Unknown stream")

        print(pkt_counter, len(dec_str1), len(dec_str2), len(dec_dpth), len(dec_data))
        inp_cont.close()


class MKVWriter(BaseWriter):

    def __init__(self):
        super(MKVWriter, self).__init__()

    def create_frame(self, image, stream):
        frame = av.VideoFrame.from_ndarray(np.array(image), format="rgb24")
        frame.pict_type = "NONE"
        frame.time_base = stream.time_base
        return frame
    
    def create_frame_depth(self, image, stream):
        image = np.array(image)

        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3:
            image = image[:,:,0]

        frame = av.VideoFrame.from_ndarray(image, format="gray")
        frame.pict_type = "NONE"
        frame.time_base = stream.time_base
        return frame

    def export(self, loader, path):
        i = -1
        for traj in loader:
            i += 1
            outdir = f"{path}/output_{i}.mkv"
            output = av.open(outdir, mode="w")
            print(outdir)

            vid_str1 = output.add_stream('libx264', rate=1)
            vid_str1.width = 640
            vid_str1.height = 480
            vid_str1.pix_fmt = 'yuv420p'

            vid_str2 = output.add_stream('libx264', rate=1)
            vid_str2.width = 640
            vid_str2.height = 480
            vid_str2.pix_fmt = 'yuv420p'

            dpth_str = output.add_stream('libx264', rate=1)
            data_str = output.add_stream('rawvideo', rate=1)

            ts = 0
            for step in traj:
                obs = step["observation"].copy()
                obs.pop("image")
                obs.pop("hand_image")
                obs.pop("image_with_depth")
                misc_step = step.copy()
                misc_step["observation"] = obs

                misc_byte = pickle.dumps(misc_step)
                packet = av.Packet(misc_byte)
                packet.stream = data_str
                packet.pts = ts
                output.mux(packet)

                image = np.array(step["observation"]["image"])
                frame = self.create_frame(image, vid_str1)
                frame.pts = ts
                packet = vid_str1.encode(frame)
                output.mux(packet)

                hand_image = np.array(step["observation"]["hand_image"])
                frame = self.create_frame(hand_image, vid_str2)
                frame.pts = ts
                packet = vid_str2.encode(frame)
                output.mux(packet)

                frame = self.create_frame_depth(step["observation"]["image_with_depth"], dpth_str)
                # frame.pts = ts
                packet = dpth_str.encode(frame)
                output.mux(packet)
                ts += 1

            for packet in vid_str1.encode():
                output.mux(packet)

            for packet in vid_str2.encode():
                output.mux(packet)

            for packet in dpth_str.encode():
                output.mux(packet)
            output.close()


def iterate_mkv(loader, num_iters):
    read_time, write_time, data_size = 0, 0, 0
    path = PATH + "mkv_convert/"

    if len(os.listdir(path)) != 0:
        raise SystemError(f"{path} must be empty.")

    # Write time
    stop = time.time()
    mkv_writer = MKVWriter()
    mkv_writer.export(loader, path)
    write_time = time.time() - stop

    # Read time
    mkv_loader = MKVLoader(path)
    stop = time.time()
    for i, data in enumerate(mkv_loader):
        if i == num_iters:
            break
    read_time = time.time() - stop

    # Data size
    for i in range(num_iters):
        data_size += os.path.getsize(f"{path}/output_{i}.mkv")

    return read_time, write_time, data_size / MB


rtx_loader = RTXLoader(PATH + "berkeley_autolab_ur5", f"train[:{N}]")
rt, wt, mb = iterate_mkv(rtx_loader, N)

print(f"\nMKV Data: \nMem. size = {mb:.4f} MB; Num. traj = {N}")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")