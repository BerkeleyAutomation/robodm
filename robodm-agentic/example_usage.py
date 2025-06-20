"""Example usage of the RoboDM Agentic framework."""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the parent robodm package to Python path
current_dir = Path(__file__).parent
robodm_root = current_dir.parent
sys.path.insert(0, str(robodm_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import from local modules with proper path handling
sys.path.insert(0, str(current_dir))
from core.agent import RoboDMAgent
from core.robodm_interface import RoboDMInterface
from clients.llm_client import LLMClient
from clients.vlm_client import VLMClient

async def main():
    """Example usage of the RoboDM Agentic framework."""
    
    # 1. Initialize the RoboDM interface
    # Point to your trajectory data directory
    data_path = "/path/to/your/robodm/data"  # Update this path
    
    # For demo, we'll check if we can find the oxe example
    current_dir = Path(__file__).parent
    oxe_example_path = current_dir.parent / "oxe_bridge_example.vla"
    
    if oxe_example_path.exists():
        data_path = str(oxe_example_path)
        print(f"Using OXE example data: {data_path}")
    else:
        # Try to find any .vla files in current directory or parent
        vla_files = list(current_dir.glob("*.vla")) + list(current_dir.parent.glob("*.vla"))
        if vla_files:
            data_path = str(vla_files[0])
            print(f"Using found VLA file: {data_path}")
        else:
            print("No trajectory data found. Please update the data_path variable.")
            return
    
    # Initialize RoboDM interface
    robodm_interface = RoboDMInterface(data_path)
    
    # 2. Initialize clients (these will gracefully handle missing dependencies)
    try:
        llm_client = LLMClient(
            model="qwen2.5:7b",  # Adjust model as needed
            provider="ollama"    # or "openai" with appropriate API key
        )
    except Exception as e:
        print(f"Could not initialize LLM client: {e}")
        print("Please install ollama or configure OpenAI API key")
        llm_client = None
    
    try:
        vlm_client = VLMClient(
            model="llava:7b",    # Adjust model as needed  
            provider="ollama"
        )
    except Exception as e:
        print(f"Could not initialize VLM client: {e}")
        vlm_client = None
    
    # 3. Initialize the agent
    agent = RoboDMAgent(
        robodm_interface=robodm_interface,
        llm_client=llm_client,
        vlm_client=vlm_client,
        enable_vision=vlm_client is not None
    )
    
    # 4. Test the setup
    print("\\n=== Testing Setup ===")
    test_results = await agent.test_setup()
    for component, status in test_results.items():
        status_str = "✓" if status else "✗"
        print(f"{status_str} {component}: {'OK' if status else 'Failed'}")
    
    if not any(test_results.values()):
        print("Setup failed. Please check your configuration.")
        return
    
    # 5. Example queries
    example_queries = [
        "How many trajectories do we have?",
        "Show me information about all trajectories",
        "Find me trajectories that are successful",
        "Sample 3 random trajectories",
        "What features are available in the first trajectory?",
    ]
    
    # Add vision-specific queries if VLM is available
    if vlm_client:
        example_queries.extend([
            "Show me frames from trajectories and describe what you see",
            "Find trajectories with interesting visual content",
        ])
    
    print("\\n=== Running Example Queries ===")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\\n--- Query {i}: {query} ---")
        
        try:
            result = await agent.query(query)
            
            if result.success:
                print("✓ Query successful")
                print(f"Generated code:\\n{result.generated_code}")
                print(f"\\nAnswer: {result.answer}")
                
                if result.frames:
                    print(f"Analyzed {len(result.frames)} frames")
            else:
                print("✗ Query failed")
                print(f"Error: {result.error}")
                
        except Exception as e:
            print(f"✗ Query exception: {e}")
    
    # 6. Interactive mode (optional)
    print("\\n=== Interactive Mode ===")
    print("Enter queries (or 'quit' to exit):")
    
    while True:
        try:
            user_query = input("\\n> ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_query:
                continue
                
            result = await agent.query(user_query)
            
            if result.success:
                print(f"\\nAnswer: {result.answer}")
            else:
                print(f"\\nError: {result.error}")
                
        except KeyboardInterrupt:
            print("\\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # 7. Cleanup
    agent.close()
    print("\\nCleaned up resources.")


async def demo_mcp_server():
    """Demonstrate the MCP server functionality."""
    from mcp.server import RoboDMMCPServer
    
    print("\\n=== MCP Server Demo ===")
    
    # Initialize interface and server
    data_path = "/path/to/your/data"  # Update as needed
    robodm_interface = RoboDMInterface(data_path)
    mcp_server = RoboDMMCPServer(robodm_interface)
    
    # Get server info
    server_info = mcp_server.get_server_info()
    print(f"MCP Server Info: {server_info}")
    
    # Note: To run the actual MCP server, you would use:
    # await mcp_server.run_stdio()
    # This connects the server to stdio for MCP client communication


if __name__ == "__main__":
    print("RoboDM Agentic Framework Example")
    print("=" * 40)
    
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to demo MCP server
    # asyncio.run(demo_mcp_server())
