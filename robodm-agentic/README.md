# RoboDM Agentic Framework

An agentic robotics data management framework built on top of RoboDM that enables natural language querying of trajectory data using Large Language Models (LLMs) and Vision-Language Models (VLMs).

## Features

- **Natural Language Queries**: Ask questions about trajectory data in plain English
- **Code Generation**: LLMs automatically generate RoboDM query code
- **Visual Analysis**: VLMs analyze trajectory frames to answer visual questions
- **Model Context Protocol (MCP)**: Standardized interface for AI assistants
- **Multi-Modal Support**: Combine text and visual analysis
- **Safe Execution**: Sandboxed code execution environment
- **Flexible Backends**: Support for Ollama, OpenAI, and other providers

## Quick Start

### 1. Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install Ollama (recommended for local models)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull qwen2.5:7b  # For code generation
ollama pull llava:7b    # For vision analysis
```

### 2. Basic Usage

```python
import asyncio
from robodm_agentic import RoboDMAgent, RoboDMInterface

async def main():
    # Initialize with your trajectory data
    robodm_interface = RoboDMInterface("/path/to/your/trajectories")
    agent = RoboDMAgent(robodm_interface)
    
    # Ask natural language questions
    result = await agent.query("find me all failed trajectories")
    print(result.answer)
    
    # Visual analysis
    result = await agent.query("show me trajectories with hidden views")
    print(result.answer)

asyncio.run(main())
```

### 3. Run Example

```bash
cd robodm-agentic
python example_usage.py
```

## Architecture

### Core Components

- **RoboDMAgent**: Main interface for natural language queries
- **RoboDMInterface**: Abstraction layer for RoboDM data access
- **LLMClient**: Interface for Language Models (Ollama, OpenAI)
- **VLMClient**: Interface for Vision-Language Models
- **CodeExecutor**: Safe execution of generated code
- **RoboDMMCPServer**: Model Context Protocol server

### Workflow

1. **User Query**: Natural language question about trajectories
2. **Code Generation**: LLM generates RoboDM query code
3. **Execution**: Safe execution of generated code
4. **Frame Extraction**: Extract visual frames if needed
5. **Visual Analysis**: VLM analyzes frames and answers question
6. **Response**: Combined textual and visual answer

## Example Queries

```python
# Basic queries
"How many trajectories do we have?"
"Find me all successful trajectories"
"Show me 5 random trajectories"

# Visual queries
"Find trajectories with hidden views"
"Show me trajectories where the robot is grasping"
"Find trajectories with red objects"

# Complex queries
"Find failed trajectories and analyze what went wrong visually"
"Compare successful vs failed trajectories"
"Find trajectories longer than 100 timesteps"
```

## Model Context Protocol (MCP)

The framework includes MCP server support for integration with AI assistants:

```python
from robodm_agentic import RoboDMMCPServer

# Create MCP server
robodm_interface = RoboDMInterface("/path/to/data")
mcp_server = RoboDMMCPServer(robodm_interface)

# Run over stdio for MCP clients
await mcp_server.run_stdio()
```

## Configuration

### LLM Configuration

```python
# Ollama (local)
llm_client = LLMClient(
    model="qwen2.5:7b",
    provider="ollama"
)

# OpenAI
llm_client = LLMClient(
    model="gpt-4",
    provider="openai",
    api_key="your-api-key"
)
```

### VLM Configuration

```python
# Ollama (local)
vlm_client = VLMClient(
    model="llava:7b",
    provider="ollama"
)

# OpenAI Vision
vlm_client = VLMClient(
    model="gpt-4-vision-preview",
    provider="openai",
    api_key="your-api-key"
)
```

## Development

### Project Structure

```
robodm-agentic/
├── core/
│   ├── agent.py           # Main RoboDMAgent
│   ├── robodm_interface.py # RoboDM data interface
│   └── code_executor.py   # Safe code execution
├── clients/
│   ├── llm_client.py      # LLM interface
│   └── vlm_client.py      # VLM interface
├── mcp/
│   └── server.py          # MCP server implementation
├── example_usage.py       # Usage examples
└── requirements.txt       # Dependencies
```

### Adding Custom Functions

Extend the RoboDM interface with custom functions:

```python
class CustomRoboDMInterface(RoboDMInterface):
    def custom_analysis(self, trajectory_id: str):
        # Your custom analysis logic
        pass
    
    def get_available_functions(self):
        functions = super().get_available_functions()
        functions["custom_analysis"] = "Perform custom trajectory analysis"
        return functions
```

## Integration with Existing RoboDM

This framework is designed as a module on top of RoboDM:

1. **Data Compatibility**: Works with existing `.vla` trajectory files
2. **API Compatibility**: Uses standard RoboDM APIs under the hood
3. **Non-Intrusive**: Doesn't modify core RoboDM functionality
4. **Extensible**: Easy to add new analysis functions

## Milestones

- [x] Core framework architecture
- [x] LLM integration for code generation
- [x] VLM integration for visual analysis
- [x] Model Context Protocol support
- [x] Safe code execution environment
- [ ] Ingestion of 100 Open-X/Droid trajectories
- [ ] Advanced query optimization
- [ ] Distributed query processing
- [ ] Web interface for queries

## Dependencies

### Required
- Python 3.8+
- numpy
- Pillow (PIL)

### Optional
- ollama (for local models)
- openai (for OpenAI API)
- mcp (for Model Context Protocol)

## License

BSD-3-Clause (same as RoboDM)
