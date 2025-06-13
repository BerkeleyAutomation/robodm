import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import robodm

# Prevent tensorflow from allocating GPU memory
tf.config.set_visible_devices([], "GPU")

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("robodm").setLevel(logging.DEBUG)


def main():
    """
    This example demonstrates converting an Open-X Embodiment (OXE)
    dataset episode to robodm format and loading it back.
    """

    def _transpose_list_of_dicts(list_of_dicts):
        """Converts a list of nested dictionaries to a nested dictionary of lists."""
        if not list_of_dicts:
            return {}

        # Base case: if the first element is not a dictionary, it's a leaf.
        if not isinstance(list_of_dicts[0], dict):
            return list_of_dicts

        dict_of_lists = {}
        # Assume all dicts in the list have the same keys as the first one.
        for key in list_of_dicts[0].keys():
            # Recursively process the values for each key.
            dict_of_lists[key] = _transpose_list_of_dicts(
                [d[key] for d in list_of_dicts]
            )
        return dict_of_lists

    # 1. Load an episode from an OXE dataset
    # We use `fractal20220817_data/bridge_from_patak_to_aloha_space` as used in the
    # reference notebook.
    # NOTE: This might take a significant amount of time on the first run
    # as it needs to download the dataset index and relevant files.
    print("Loading OXE dataset from tensorflow_datasets...")
    builder = tfds.builder_from_directory(builder_dir=
        "gs://gresearch/robotics/fractal20220817_data/0.1.0"
    )

    # Load the first episode from the training split.
    ds = builder.as_dataset(split="train[:1]")
    episode = next(iter(tfds.as_numpy(ds)))

    # The episode contains 'steps' which is a tf.data.Dataset object.
    # We first convert it into a list of step dictionaries.
    steps_list = list(episode["steps"])

    if not steps_list:
        print("Episode is empty, exiting.")
        return

    # Now, we transpose this list of dictionaries into a dictionary of lists.
    # This is the format `from_dict_of_lists` expects.
    episode_steps = _transpose_list_of_dicts(steps_list)

    num_steps = len(episode_steps["observation"]["image"])
    print(f"Loaded episode with {num_steps} steps.")

    # Let's check the shape of an image from the original dataset
    original_image_shape = episode_steps["observation"]["image"][0].shape
    print(f"Original image shape: {original_image_shape}")

    # 2. Convert to robodm format and save
    path = "./oxe_bridge_example.vla" #os.path.join(tempfile.gettempdir(), "oxe_bridge_example.vla")
    print(f"Converting and saving to {path}...")

    # `from_dict_of_lists` is perfect for this. It takes a dictionary
    # where keys are feature names and values are lists (or arrays) of data
    # for each timestep. The nested dictionary from OXE is flattened automatically.
    robodm.Trajectory.from_dict_of_lists(data=episode_steps, path=path, video_codec="libx264")
    print("Conversion successful.")

    # 3. Load the trajectory back
    print("Loading trajectory back with robodm...")
    traj = robodm.Trajectory(path=path, mode="r")
    loaded_data = traj.load()
    traj.close()

    # 4. Verify the loaded data
    loaded_num_steps = len(loaded_data["observation/image"])
    print(f"Loaded trajectory with {loaded_num_steps} timesteps")
    print(f"Image shape from robodm: {loaded_data['observation/image'][0].shape}")
    print(f"Loaded keys: {loaded_data.keys()}")
    
    # write all images to disk
    for i in range(loaded_num_steps):
        from PIL import Image
        import os
        os.makedirs("images", exist_ok=True)
        image = loaded_data["observation/image"][i]
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f"images/image_{i}.png")
    
    # Compare shapes and number of steps
    assert loaded_num_steps == num_steps
    assert loaded_data["observation/image"][0].shape == original_image_shape
    print("\nVerification successful: Number of steps and image shapes match.")

    # Clean up
    # os.remove(path)
    print(f"Cleaned up temporary file: {path}")


if __name__ == "__main__":
    main() 