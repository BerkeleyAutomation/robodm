"""Model Context Protocol (MCP) server for RoboDM agentic framework.

The Model Context Protocol is a standardized way for AI assistants to connect 
with external data sources and tools. This server exposes RoboDM functionality
as MCP tools and resources.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Sequence
import logging

logger = logging.getLogger(__name__)

try:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    # Create mock types for when MCP is not available
    class MockTypes:
        class Tool:
            def __init__(self, name: str, description: str, inputSchema: Dict):
                self.name = name
                self.description = description
                self.inputSchema = inputSchema
        
        class Resource:
            def __init__(self, uri: str, name: str, description: str, mimeType: str = "text/plain"):
                self.uri = uri
                self.name = name
                self.description = description
                self.mimeType = mimeType
        
        class TextContent:
            def __init__(self, type: str, text: str):
                self.type = type
                self.text = text
        
        class CallToolResult:
            def __init__(self, content: List):
                self.content = content
        
        class ListResourcesResult:
            def __init__(self, resources: List):
                self.resources = resources
        
        class ReadResourceResult:
            def __init__(self, contents: List):
                self.contents = contents
    
    types = MockTypes()

from ..core.robodm_interface import RoboDMInterface


class RoboDMMCPServer:
    """MCP Server for RoboDM trajectory data access."""
    
    def __init__(self, robodm_interface: RoboDMInterface, server_name: str = "robodm-agentic"):
        """Initialize MCP server.
        
        Args:
            robodm_interface: RoboDM interface instance
            server_name: Name of the MCP server
        """
        self.robodm = robodm_interface
        self.server_name = server_name
        
        if not HAS_MCP:
            logger.warning("MCP package not available. Install with: pip install mcp")
            self.server = None
        else:
            self.server = Server(server_name)
            self._register_handlers()
    
    def _register_handlers(self):
        """Register MCP handlers for tools and resources."""
        if not self.server:
            return
            
        # Register tool handlers
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available RoboDM tools."""
            return [
                types.Tool(
                    name="get_all_trajectories",
                    description="Get list of all trajectory IDs in the database",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="get_trajectory_metadata",
                    description="Get metadata for a specific trajectory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "trajectory_id": {
                                "type": "string",
                                "description": "ID of the trajectory"
                            }
                        },
                        "required": ["trajectory_id"]
                    }
                ),
                types.Tool(
                    name="get_trajectory_data",
                    description="Load complete trajectory data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "trajectory_id": {
                                "type": "string",
                                "description": "ID of the trajectory"
                            },
                            "features": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific features to load (optional)"
                            }
                        },
                        "required": ["trajectory_id"]
                    }
                ),
                types.Tool(
                    name="filter_trajectories",
                    description="Filter trajectories by metadata criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "object",
                                "description": "Filter criteria as key-value pairs"
                            }
                        },
                        "required": ["criteria"]
                    }
                ),
                types.Tool(
                    name="search_trajectories",
                    description="Search trajectories by text query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="count_trajectories",
                    description="Count trajectories matching criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "object",
                                "description": "Filter criteria (optional)"
                            }
                        },
                        "required": []
                    }
                ),
                types.Tool(
                    name="sample_trajectories",
                    description="Randomly sample N trajectories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of trajectories to sample"
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed (optional)"
                            }
                        },
                        "required": ["n"]
                    }
                ),
                types.Tool(
                    name="get_trajectory_frames",
                    description="Get visual frames from a trajectory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "trajectory_id": {
                                "type": "string",
                                "description": "ID of the trajectory"
                            }
                        },
                        "required": ["trajectory_id"]
                    }
                ),
                types.Tool(
                    name="slice_trajectory",
                    description="Get a subset of trajectory data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "trajectory_id": {
                                "type": "string",
                                "description": "ID of the trajectory"
                            },
                            "start": {
                                "type": "integer",
                                "description": "Start index (optional)"
                            },
                            "end": {
                                "type": "integer",
                                "description": "End index (optional)"
                            },
                            "step": {
                                "type": "integer",
                                "description": "Step size (optional)"
                            }
                        },
                        "required": ["trajectory_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
            """Handle tool calls."""
            try:
                if name == "get_all_trajectories":
                    result = self.robodm.get_all_trajectories()
                    
                elif name == "get_trajectory_metadata":
                    trajectory_id = arguments["trajectory_id"]
                    result = self.robodm.get_trajectory_metadata(trajectory_id)
                    
                elif name == "get_trajectory_data":
                    trajectory_id = arguments["trajectory_id"]
                    features = arguments.get("features")
                    result = self.robodm.get_trajectory_data(trajectory_id, features)
                    # Convert numpy arrays to lists for JSON serialization
                    result = self._serialize_data(result)
                    
                elif name == "filter_trajectories":
                    criteria = arguments["criteria"]
                    result = self.robodm.filter_trajectories_by_metadata(criteria)
                    
                elif name == "search_trajectories":
                    query = arguments["query"]
                    result = self.robodm.search_trajectories(query)
                    
                elif name == "count_trajectories":
                    criteria = arguments.get("criteria")
                    result = self.robodm.count_trajectories(criteria)
                    
                elif name == "sample_trajectories":
                    n = arguments["n"]
                    seed = arguments.get("seed")
                    result = self.robodm.sample_trajectories(n, seed)
                    
                elif name == "get_trajectory_frames":
                    trajectory_id = arguments["trajectory_id"]
                    frames = self.robodm.get_trajectory_frames(trajectory_id)
                    # For MCP, we'll return metadata about frames rather than raw data
                    result = {
                        "trajectory_id": trajectory_id,
                        "num_frames": len(frames),
                        "frame_info": f"Found {len(frames)} frames in trajectory {trajectory_id}"
                    }
                    
                elif name == "slice_trajectory":
                    trajectory_id = arguments["trajectory_id"]
                    start = arguments.get("start")
                    end = arguments.get("end")
                    step = arguments.get("step")
                    result = self.robodm.slice_trajectory(trajectory_id, start, end, step)
                    result = self._serialize_data(result)
                    
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps(result, indent=2, default=str)
                        )
                    ]
                )
                
            except Exception as e:
                logger.error(f"Tool call failed: {name} - {e}")
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text", 
                            text=f"Error: {str(e)}"
                        )
                    ]
                )
        
        # Register resource handlers
        @self.server.list_resources()
        async def handle_list_resources() -> types.ListResourcesResult:
            """List available RoboDM resources."""
            resources = []
            
            try:
                # Get all trajectories as resources
                trajectories = self.robodm.get_all_trajectories()
                
                for traj_id in trajectories[:50]:  # Limit to first 50 for performance
                    resources.append(
                        types.Resource(
                            uri=f"robodm://trajectory/{traj_id}",
                            name=f"Trajectory {traj_id}",
                            description=f"RoboDM trajectory data for {traj_id}",
                            mimeType="application/json"
                        )
                    )
                
                # Add summary resource
                resources.append(
                    types.Resource(
                        uri="robodm://summary",
                        name="Database Summary",
                        description="Summary of the RoboDM database",
                        mimeType="application/json"
                    )
                )
                
            except Exception as e:
                logger.error(f"Error listing resources: {e}")
            
            return types.ListResourcesResult(resources=resources)
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> types.ReadResourceResult:
            """Read a specific resource."""
            try:
                if uri.startswith("robodm://trajectory/"):
                    # Extract trajectory ID from URI
                    traj_id = uri.split("/")[-1]
                    metadata = self.robodm.get_trajectory_metadata(traj_id)
                    
                    content = types.TextContent(
                        type="text",
                        text=json.dumps(metadata, indent=2, default=str)
                    )
                    
                elif uri == "robodm://summary":
                    # Provide database summary
                    all_trajectories = self.robodm.get_all_trajectories()
                    summary = {
                        "total_trajectories": len(all_trajectories),
                        "sample_trajectories": all_trajectories[:10],
                        "available_functions": list(self.robodm.get_available_functions().keys())
                    }
                    
                    content = types.TextContent(
                        type="text",
                        text=json.dumps(summary, indent=2)
                    )
                    
                else:
                    raise ValueError(f"Unknown resource URI: {uri}")
                
                return types.ReadResourceResult(contents=[content])
                
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                error_content = types.TextContent(
                    type="text",
                    text=f"Error reading resource: {str(e)}"
                )
                return types.ReadResourceResult(contents=[error_content])
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON transmission."""
        if hasattr(data, 'tolist'):  # numpy array
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        else:
            return data
    
    async def run_stdio(self):
        """Run the MCP server over stdio."""
        if not self.server:
            raise RuntimeError("MCP server not available. Install mcp package.")
            
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )
    
    async def run_websocket(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server over WebSocket."""
        if not self.server:
            raise RuntimeError("MCP server not available. Install mcp package.")
        
        # WebSocket server implementation would go here
        # This is a placeholder for now
        logger.info(f"WebSocket MCP server would run on ws://{host}:{port}")
        raise NotImplementedError("WebSocket server not yet implemented")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the MCP server."""
        return {
            "name": self.server_name,
            "mcp_available": HAS_MCP,
            "num_trajectories": len(self.robodm.get_all_trajectories()) if self.robodm else 0,
            "available_tools": list(self.robodm.get_available_functions().keys()) if self.robodm else [],
        }
