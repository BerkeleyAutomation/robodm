import os
import sys
import polars as pl
import pyarrow.dataset as ds

PATH = os.path.expanduser("~")
sys.path.append(PATH + "/fog_x_fork")
from fog_x.dataset import Dataset

PATH += "/datasets/"
NAME = "test_convert"
dataset = Dataset(name=NAME, path=PATH)

if False:
    dataset.load_rtx_episodes(name="berkeley_autolab_ur5", split="train[:51]")

# Debugging: path = "/home/joshzhang/datasets"
    # Starts with "/" but does not end with it
    # Should not initialize any cloud buckets!

"""###"""

# Retrieve episode-level metadata as a pandas DataFrame
episode_info = dataset.get_episode_info()

# Codebase map:
db_manager  = dataset.db_manager                # DatabaseManager
traj_info_c = db_manager.episode_info_connector # DataFrameC, child of (PolarsC)
step_data_c = db_manager.step_data_connector    # LazyFrameC, child of (PolarsC)

# Call tracing:
    # dataset.get_episode_info()
    # db_manager.get_episode_info_table()
    # traj_info_c.select_table(dataset_name)
    # -> traj_info_c.tables[dataset_name]

episode_info = traj_info_c.tables[NAME]     # Same object!
print(episode_info)

"""###"""

# Obtain the actual episodes containing their step data
episode_0 = dataset.read_by(episode_info)[0]

# Call tracing:
    # dataset.read_by(episode_info)
    # :: i = list(episode_info["episode_id"])[0]
    # db_manager.get_step_table(i)
    # step_data_c.select_table(dataset_name_i)
    # -> step_data_c.tables[dataset_name_i]

episode_0 = step_data_c.tables[NAME + "_0"] # Same object!
print(episode_0.collect())

"""###"""

# Retrieve Fog-X dataset step data as a Polars LazyFrame
step_data = dataset.get_step_data()

# Call tracing:
    # dataset.get_step_data()
    # db_manager.get_step_table_all()
    # step_data_c.get_dataset_table()
    # :: d = ds.dataset(dataset_path, format="parquet")
    # -> pl.scan_pyarrow_dataset(d)

d = ds.dataset(PATH + NAME, format="parquet")
step_data = pl.scan_pyarrow_dataset(d)      # Same object!
print(step_data.collect())