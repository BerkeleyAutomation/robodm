# TF_Record: 
# Disk size (MB) = 8039.7179 MB  ; Num. traj = 51
# Read:  latency = 15.5696 s; throughput = 516.3728 MB/s, 3.2756 (traj/s)
# Write: latency = 36.6353 s; throughput = 219.4528 MB/s, 1.3921 (traj/s)

"""###"""

# Fog-X:
# Disk size (MB) = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 42.1416 s ; throughput = 159.1630 MB/s, 1.2102 (traj/s)
# Write: latency = 258.8635 s; throughput = 25.9109 MB/s,  0.1970 (traj/s)

# Polars:
# Disk size (MB) = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 22.9974 s ; throughput = 291.6581 MB/s, 2.2176 (traj/s)
# Write: latency = 241.4137 s; throughput = 27.7838 MB/s,  0.2113 (traj/s)

# PyArrow:
# Disk size (MB) = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 27.9808 s ; throughput = 239.7136 MB/s, 1.8227 (traj/s)
# Write: latency = 80.9986 s ; throughput = 82.8086 MB/s,  0.6296 (traj/s)

# Pandas:
# Disk size (MB) = 6707.3818 MB   ; Num. traj = 51
# Read:  latency = 49.1056 s ; throughput = 136.5909 MB/s, 1.0386 (traj/s)
# Write: latency = 103.6820 s; throughput = 64.6918 MB/s,  0.4919 (traj/s)

"""###"""

# Arrow_REG:
# Disk size (MB) = 14651.9697 MB ; Num. traj = 51
# Read:  latency = 39.5354 s; throughput = 370.6041 MB/s, 1.2900 (traj/s)
# Write: latency = 26.4765 s; throughput = 553.3952 MB/s, 1.9262 (traj/s)

# Arrow_IPC:
# Disk size (MB) = 14651.9697 MB ; Num. traj = 51
# Read:  latency = 44.0609 s; throughput = 332.5391 MB/s, 1.1575 (traj/s)
# Write: latency = 25.1574 s; throughput = 582.4108 MB/s, 2.0272 (traj/s)

# Feather_FTH:
# Disk size (MB) = 14651.9697 MB; Num. traj = 51
# Read:  latency = 47.5646 s; throughput = 308.0433 MB/s, 1.0722 (traj/s)
# Write: latency = 35.4595 s; throughput = 413.2030 MB/s, 1.4383 (traj/s)

# Pandas_FTH:
# Disk size (MB) = 14651.9697 MB; Num. traj = 51
# Read:  latency = 26.6213 s; throughput = 550.3844 MB/s, 1.9158 (traj/s)
# Write: latency = 45.9015 s; throughput = 319.2046 MB/s, 1.1111 (traj/s)

"""###"""

# HDF5:
# Disk size (MB) = 44319.6689 MB  ; Num. traj = 51
# Read:  latency = 191.5964 s; throughput = 231.3179 MB/s, 0.2662 (traj/s)
# Write: latency = 129.6112 s; throughput = 341.9432 MB/s, 0.3935 (traj/s)

import os
import matplotlib.pyplot as plt
PATH = os.path.expanduser("~") + "/fog_x_fork/benchmarking/plots/"

formats = {
    "TFRecord": {
        "Disk size (MB)": 8039.7179,
        "Read latency (s)": 15.5696,
        "Read throughput MB/s": 516.3728,
        "Read throughput (traj/s)": 3.2756,
        "Write latency (s)": 36.6353,
        "Write throughput MB/s": 219.4528,
        "Write throughput (traj/s)": 1.3921,
    },
    "Parquet": {
        "Disk size (MB)": 6707.3818,
        "Read latency (s)": 27.9808,
        "Read throughput MB/s": 239.7136,
        "Read throughput (traj/s)": 1.8227,
        "Write latency (s)": 80.9986,
        "Write throughput MB/s": 82.8086,
        "Write throughput (traj/s)": 0.6296,
    },
    "Arrow": {
        "Disk size (MB)": 14651.9697,
        "Read latency (s)": 44.0609,
        "Read throughput MB/s": 332.5391,
        "Read throughput (traj/s)": 1.1575,
        "Write latency (s)": 25.1574,
        "Write throughput MB/s": 582.4108,
        "Write throughput (traj/s)": 2.0272,
    },
    "HDF5": {
        "Disk size (MB)": 44319.6689,
        "Read latency (s)": 191.5964,
        "Read throughput MB/s": 231.3179,
        "Read throughput (traj/s)": 0.2662,
        "Write latency (s)": 129.6112,
        "Write throughput MB/s": 341.9432,
        "Write throughput (traj/s)": 0.3935,
    },
    "MKV": {
        "Disk size (MB)": 135.6294,
        "Read latency (s)": 11.8973,
        "Read throughput MB/s": 11.4,
        "Read throughput (traj/s)": 4.2867,
        "Write latency (s)": 205.9287,
        "Write throughput MB/s": 0.6586,
        "Write throughput (traj/s)": 0.2477,
    },
}
metrics = [
    "Disk size (MB)", 
    "Read latency (s)", 
    "Read throughput (traj/s)", 
    "Write latency (s)", 
    "Write throughput (traj/s)"
]
colors = {
    "TFRecord": "cyan",
    "Parquet": "pink",
    "Arrow": "purple",
    "HDF5": "green",
    "MKV": "orange"
}
F_SIZE = 12

for metric in metrics:
    plt.figure(figsize=(9, 6))
    labels = []
    values = []

    for label, format in formats.items():
        labels.append(label)
        values.append(format[metric])
        colist = [colors[l] for l in labels]
        
    bars = plt.bar(labels, values, color=colist)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{round(height, 2)} {metric.split(" ")[-1]}',
            ha='center',
            va='bottom',
            fontsize=F_SIZE
        )
    plt.title(metric)
    plt.yticks(fontsize=F_SIZE)
    plt.xticks(fontsize=F_SIZE)
    plt.tight_layout()
    
    save_path = PATH + f"{metric.replace("/", "_")}.png"
    plt.savefig(save_path)
    plt.close()