"""Script to ingest Open-X/Droid trajectories for the agentic framework.

This script demonstrates how to use the robodm-agentic framework to query
trajectories after ingesting data from Open-X Embodiment datasets.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the parent robodm package to path to import robodm
current_dir = Path(__file__).parent
robodm_root = current_dir.parent
sys.path.insert(0, str(robodm_root))

import robodm

# Add current directory to path for local imports
sys.path.insert(0, str(current_dir))
from core.agent import RoboDMAgent
from core.robodm_interface import RoboDMInterface

logger = logging.getLogger(__name__)


async def ingest_oxe_data(output_dir: str = "./oxe_trajectories", num_trajectories: int = 100):
    """Ingest Open-X Embodiment data for agentic querying.
    
    Args:
        output_dir: Directory to store converted trajectories
        num_trajectories: Number of trajectories to ingest
    """
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError:
        print("TensorFlow and tensorflow_datasets required for OXE ingestion.")
        print("Install with: pip install tensorflow tensorflow_datasets")
        return False
    
    # Prevent TF from allocating GPU memory
    tf.config.set_visible_devices([], "GPU")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Ingesting {num_trajectories} Open-X Embodiment trajectories...")
    
    try:
        # Load dataset
        builder = tfds.builder_from_directory(
            builder_dir="gs://gresearch/robotics/fractal20220817_data/0.1.0"
        )
        
        # Get dataset
        ds = builder.as_dataset(split=f"train[:{num_trajectories}]")
        
        converted_count = 0
        
        for i, episode in enumerate(tfds.as_numpy(ds)):
            try:
                # Convert episode steps to list of dictionaries
                steps_list = list(episode["steps"])
                
                if not steps_list:
                    print(f"Skipping empty episode {i}")
                    continue
                
                # Transpose to dictionary of lists format
                episode_steps = _transpose_list_of_dicts(steps_list)
                
                # Save as robodm trajectory
                output_path = os.path.join(output_dir, f"oxe_trajectory_{i:06d}.vla")
                
                robodm.Trajectory.from_dict_of_lists(
                    data=episode_steps, 
                    path=output_path, 
                    video_codec="libx264"
                )
                
                converted_count += 1
                if converted_count % 10 == 0:
                    print(f"Converted {converted_count}/{num_trajectories} trajectories")
                    
            except Exception as e:
                logger.warning(f"Failed to convert episode {i}: {e}")
                continue
        
        print(f"Successfully ingested {converted_count} trajectories to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to ingest OXE data: {e}")
        return False


def _transpose_list_of_dicts(list_of_dicts):
    """Convert list of dictionaries to dictionary of lists."""
    if not list_of_dicts:
        return {}
    
    if not isinstance(list_of_dicts[0], dict):
        return list_of_dicts
    
    dict_of_lists = {}
    for key in list_of_dicts[0].keys():
        dict_of_lists[key] = _transpose_list_of_dicts(
            [d[key] for d in list_of_dicts]
        )
    return dict_of_lists


async def demo_agentic_queries(data_dir: str):
    """Demonstrate agentic queries on ingested data.
    
    Args:
        data_dir: Directory containing ingested trajectory files
    """
    print(f"\\n=== Demonstrating Agentic Queries ===")
    
    # Initialize the agentic framework
    robodm_interface = RoboDMInterface(data_dir, pattern="*.vla")
    
    try:
        from clients.llm_client import LLMClient
        from clients.vlm_client import VLMClient
        
        # Initialize with local models (adjust as needed)
        llm_client = LLMClient(model="qwen2.5:7b", provider="ollama")
        vlm_client = VLMClient(model="llava:7b", provider="ollama")
        
        agent = RoboDMAgent(
            robodm_interface=robodm_interface,
            llm_client=llm_client,
            vlm_client=vlm_client,
            enable_vision=True
        )
        
    except Exception as e:
        print(f"Could not initialize full agent: {e}")
        print("Running with basic functionality only...")
        agent = RoboDMAgent(robodm_interface=robodm_interface, enable_vision=False)
    
    # Example queries for robotics trajectories
    robotics_queries = [
        "How many trajectories do we have in total?",
        "Find me trajectories that are longer than 50 timesteps",
        "Sample 5 random trajectories and show their metadata",
        "Find successful trajectories",
        "What are the common feature names across all trajectories?",
        "Show me trajectories with image data",
        "Find trajectories that might have failed based on their metadata",
        "Compare the lengths of different trajectories",
    ]
    
    # Add vision queries if VLM is available
    if agent.enable_vision:
        robotics_queries.extend([
            "Show me visual frames from trajectories and describe the robot actions",
            "Find trajectories where the robot appears to be grasping objects",
            "Analyze the visual content of random trajectory frames",
        ])
    
    print(f"Running {len(robotics_queries)} example queries...")
    
    for i, query in enumerate(robotics_queries, 1):
        print(f"\\n--- Query {i}: {query} ---")
        
        try:
            result = await agent.query(query)
            
            if result.success:
                print("✓ Success")
                print(f"Answer: {result.answer[:200]}...")  # Truncate long answers
                
                if result.frames:
                    print(f"Analyzed {len(result.frames)} visual frames")
            else:
                print("✗ Failed")
                print(f"Error: {result.error}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    # Cleanup
    agent.close()


async def main():
    """Main function to run ingestion and demo."""
    print("RoboDM Agentic Framework - OXE Ingestion Demo")
    print("=" * 50)
    
    # Configuration
    output_dir = "./oxe_trajectories"
    num_trajectories = 10  # Start small for demo
    
    # Check if data already exists
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Found existing data in {output_dir}")
        use_existing = input("Use existing data? (y/n): ").lower().startswith('y')
        
        if not use_existing:
            import shutil
            shutil.rmtree(output_dir)
            success = await ingest_oxe_data(output_dir, num_trajectories)
            if not success:
                return
    else:
        # Ingest new data
        success = await ingest_oxe_data(output_dir, num_trajectories)
        if not success:
            return
    
    # Demo agentic queries
    await demo_agentic_queries(output_dir)
    
    print("\\nDemo complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
