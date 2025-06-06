import os
import av
import sys
import time
import pickle
import shutil
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

    def __len__(self):
        return len(self.ds)
    
    def __iter__(self):
        return self.ds.__iter__()


class RTXWriter(BaseWriter):
    def __init__(self):
        super(RTXWriter, self).__init__()
        self.wr = tf.io.TFRecordWriter

    def export(self, loader, path):
        for i, data in enumerate(loader):
            data_list = list(data["steps"].as_numpy_iterator())
            data_flat = flatten(data_list)

            with self.wr(f"{path}/output_{i}.tfrecord") as writer:
                for dl in data_flat:
                    writer.write(dl)

    def export_temp(self, data_flat, path):
        with self.wr(path + "/temp.tfrecord") as writer:
            for dl in data_flat:
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


def iterate_rtx(loader, writer, n):
    read_time, write_time, data_size = 0, 0, 0

    for i, data in enumerate(loader):
        print(f"Measuring trajectory {i}")

        stop = time.time()
        data_list = list(data["steps"].as_numpy_iterator())
        read_time += time.time() - stop

        name = "berkeley_autolab_ur5"
        num = f"{str(i).zfill(5)}-of-00412"
        data_size += os.path.getsize(f"{PATH}/{name}/{name}-train.tfrecord-{num}")

        data_flat = flatten(data_list)
        stop = time.time()
        writer.export_temp(data_flat, PATH)
        write_time += time.time() - stop

        if i == n: break
    return read_time, write_time, data_size / MB


rtx_loader = RTXLoader(PATH + "berkeley_autolab_ur5", f"train[:{N}]")
rtx_writer = RTXWriter()
rt, wt, mb = iterate_rtx(rtx_loader, rtx_writer, N)

print(f"\nRTX Data: \nMem. size = {mb:.4f} MB; Num. traj = {N}")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")


class MVKWriter(BaseWriter):
    def __init__(self):
        super(MVKWriter, self).__init__()

    # Function to create a frame from numpy array
    def create_frame(self, image, stream):
        frame = av.VideoFrame.from_ndarray(np.array(image), format='rgb24')
        frame.pict_type = 'NONE'
        frame.time_base = stream.time_base
        return frame
    
    # Function to create a frame from numpy array
    def create_frame_depth(self, image, stream):
        image_array = np.array(image)
        # if float, convert to uint8
        if image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)
        # if 3 dim, convert to 2 dim
        if len(image_array.shape) == 3:
            image_array = image_array[:,:,0]
        frame = av.VideoFrame.from_ndarray(image_array, format='gray')
        frame.pict_type = 'NONE'
        frame.time_base = stream.time_base
        return frame

    def export(self, loader, path):
        # Create an output container
        i = -1
        for traj_tensor in loader:
            i += 1
            trajectory = dict(traj_tensor)
            output = av.open(f'{path}/output_{i}.mkv', mode='w')

            # Define video streams (assuming images are 640x480 RGB)
            video_stream_1 = output.add_stream('libx264', rate=1)
            video_stream_1.width = 640
            video_stream_1.height = 480
            video_stream_1.pix_fmt = 'yuv420p'

            video_stream_2 = output.add_stream('libx264', rate=1)
            video_stream_2.width = 640
            video_stream_2.height = 480
            video_stream_2.pix_fmt = 'yuv420p'

            # Define custom data stream for vectors
            depth_stream = output.add_stream('libx264', rate=1)
            data_stream = output.add_stream('rawvideo', rate=1)

            ts = 0
            # convert step data to stream
            for step_tensor in trajectory["steps"]:
                step = dict(step_tensor)
                observation = step_tensor["observation"].copy()
                observation.pop("image")
                observation.pop("hand_image")
                observation.pop("image_with_depth")
                non_image_data_step = step.copy()
                non_image_data_step["observation"] = observation

                non_image_data_bytes = pickle.dumps(non_image_data_step)
                packet = av.Packet(non_image_data_bytes)
                packet.stream = data_stream
                packet.pts = ts
                output.mux(packet)

                image = np.array(step["observation"]["image"])
                # Create a frame from the numpy array
                frame = self.create_frame(image, video_stream_1)
                frame.pts = ts
                packet = video_stream_1.encode(frame)
                output.mux(packet)

                hand_image =np.array(step["observation"]["hand_image"])
                # Create a frame from the numpy array
                frame = self.create_frame(hand_image, video_stream_2)
                frame.pts = ts
                packet = video_stream_2.encode(frame)
                output.mux(packet)

                # Create a frame from the numpy array
                frame = self.create_frame_depth(step["observation"]["image_with_depth"], depth_stream)
                # frame.pts = ts
                # Encode the frame
                packet = depth_stream.encode(frame)
                # Write the packet to the output file
                output.mux(packet)
                ts += 1

            # Flush the remaining frames
            for packet in video_stream_1.encode():
                output.mux(packet)

            for packet in video_stream_2.encode():
                output.mux(packet)

            for packet in depth_stream.encode():
                output.mux(packet)
            output.close()


class MKVLoader(BaseLoader):
    def __init__(self, path):
        super(MKVLoader, self).__init__(path)
        self.files = [path + f for f in os.listdir(path) if f.endswith('.mkv')]
        self.index = 0

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return idx
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            result = self.files[self.index]
            self.index += 1
            return self._parse_mkv_file(result)
        else:
            raise StopIteration
    
    def _parse_mkv_file(self, filename):
        print(filename)
        input_container = av.open(filename)

        video_stream1 = input_container.streams.video[0] 
        video_stream1.thread_type = 'AUTO'
        video_stream2 = input_container.streams.video[1] 
        video_stream2.thread_type = 'AUTO'
        depth_stream = input_container.streams.video[2] 
        depth_stream.thread_type = 'AUTO'
        data_stream = input_container.streams[3] 

        decoded_stream_1 = []
        decoded_stream_2 = []
        decoded_stream_depth = []
        decoded_stream_data = []

        pkt_counter = 0
        for packet in input_container.demux(video_stream1, video_stream2, depth_stream, data_stream):
            pkt_counter += 1
            if packet.stream.index == video_stream1.index: 
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format='rgb24')
                        decoded_stream_1.append(image)
            elif packet.stream.index == video_stream2.index:
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format='rgb24')
                        decoded_stream_2.append(image)
            elif packet.stream.index == depth_stream.index:
                frame = packet.decode()
                if frame:
                    for f in frame:
                        image = f.to_ndarray(format='gray')
                        decoded_stream_depth.append(image)
            elif packet.stream.index == data_stream.index:
                packet_in_bytes = bytes(packet)
                if packet_in_bytes:
                    non_dict = pickle.loads(packet_in_bytes)
                    decoded_stream_data.append(non_dict)
            else:
                print("Unknown stream")
        print(pkt_counter, len(decoded_stream_1), len(decoded_stream_2), len(decoded_stream_depth), len(decoded_stream_data))
        input_container.close()


def iterate_mkv(rtx_loader, mkv_writer, n):
    read_time, write_time, data_size = 0, 0, 0
    path = PATH + "mkv_convert/"

    if len(os.listdir(path)) != 0:
        raise SystemError(f"{path} must be empty.")

    stop = time.time()
    mkv_writer.export(rtx_loader, path)
    write_time += time.time() - stop

    mkv_loader = MKVLoader(path)

    stop = time.time()
    for i, data in enumerate(mkv_loader):
        print(f"Reading trajectory {i}")
        if i == n: break
    read_time += time.time() - stop

    for i in range(n):
        data_size += os.path.getsize(f"{path}/output_{i}.mkv")

    return read_time, write_time, data_size / MB


mkv_writer = MVKWriter()
rt, wt, mb = iterate_mkv(rtx_loader, mkv_writer, N)

print(f"\nMKV Data: \nMem. size = {mb:.4f} MB; Num. traj = {N}")
print(f"Read:  latency = {rt:.4f} s; throughput = {mb / rt :.4f} MB/s, {N / rt :.4f} traj/s")
print(f"Write: latency = {wt:.4f} s; throughput = {mb / wt :.4f} MB/s, {N / wt :.4f} traj/s")