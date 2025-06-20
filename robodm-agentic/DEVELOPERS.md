# RoboDM Agentic Framework - Developer Documentation

This document provides comprehensive information for developers working on or extending the RoboDM Agentic Framework.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Development Setup](#development-setup)
- [Design Patterns](#design-patterns)
- [Extension Points](#extension-points)
- [Testing](#testing)
- [Model Context Protocol](#model-context-protocol)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The RoboDM Agentic Framework is built as a modular system that sits on top of the core RoboDM library. It follows a multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  Natural Language Queries, MCP Clients, Interactive Shell  │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Agent Layer                            │
│         RoboDMAgent - Main orchestration logic             │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                         │
│  LLMClient  │  VLMClient  │  CodeExecutor  │  MCPServer   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Interface Layer                           │
│              RoboDMInterface                               │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                              │
│              Core RoboDM Library                           │
│           Trajectory Files (.vla)                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**: Each component is self-contained and can be used independently
2. **Extensibility**: Easy to add new models, functions, or data sources
3. **Safety**: Code execution is sandboxed and restricted
4. **Compatibility**: Works with existing RoboDM data and APIs
5. **Flexibility**: Supports multiple LLM/VLM providers and deployment scenarios

## Project Structure

```
robodm-agentic/
├── __init__.py                 # Package initialization and exports
├── README.md                   # User documentation
├── DEVELOPERS.md              # This file
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
├── example_usage.py           # Basic usage examples
├── ingest_oxe_demo.py        # OXE data ingestion demo
│
├── core/                      # Core framework components
│   ├── __init__.py
│   ├── agent.py              # Main RoboDMAgent class
│   ├── robodm_interface.py   # RoboDM data access abstraction
│   └── code_executor.py      # Safe code execution engine
│
├── clients/                   # External service clients
│   ├── __init__.py
│   ├── llm_client.py         # Language model interface
│   └── vlm_client.py         # Vision-language model interface
│
└── mcp/                      # Model Context Protocol implementation
    ├── __init__.py
    └── server.py             # MCP server for AI assistant integration
```

## Core Components

### 1. RoboDMAgent (`core/agent.py`)

The main orchestrator that coordinates between all components to process natural language queries.

**Key Methods:**
- `query(user_query)`: Main entry point for processing queries
- `batch_query(queries)`: Process multiple queries in parallel
- `test_setup()`: Verify all components are working

**Workflow:**
1. Analyze query to determine if vision analysis is needed
2. Generate code using LLM
3. Execute code safely
4. Extract frames if needed
5. Analyze with VLM if applicable
6. Return comprehensive result

### 2. RoboDMInterface (`core/robodm_interface.py`)

Abstraction layer that provides a clean API for accessing RoboDM trajectory data.

**Key Features:**
- Trajectory discovery and caching
- Metadata extraction and filtering
- Frame extraction for vision analysis
- Search and sampling capabilities

**Available Functions:**
```python
get_all_trajectories()          # List all trajectory IDs
get_trajectory_metadata(id)     # Get metadata for trajectory
get_trajectory_data(id)         # Load complete trajectory data
get_trajectory_frames(id)       # Extract visual frames
filter_trajectories_by_metadata(criteria)  # Filter by criteria
search_trajectories(query)      # Text-based search
sample_trajectories(n)          # Random sampling
```

### 3. LLMClient (`clients/llm_client.py`)

Interface for Large Language Models that generate RoboDM query code.

**Supported Providers:**
- **Ollama**: Local model hosting (recommended for development)
- **OpenAI**: Cloud-based models (GPT-3.5, GPT-4)
- **Anthropic**: Claude models (future extension)

**Code Generation Process:**
1. Build system prompt with available functions and examples
2. Send user query to LLM
3. Extract and clean generated Python code
4. Return executable code

### 4. VLMClient (`clients/vlm_client.py`)

Interface for Vision-Language Models that analyze trajectory frames.

**Supported Models:**
- **LLaVA**: Open-source vision-language model
- **GPT-4 Vision**: OpenAI's multimodal model

**Frame Processing:**
- Handles multiple image formats (numpy arrays, PIL Images, raw bytes)
- Automatic format conversion and encoding
- Batch processing for multiple frames

### 5. CodeExecutor (`core/code_executor.py`)

Safe execution environment for LLM-generated code.

**Safety Features:**
- Restricted built-ins (no file I/O, import restrictions)
- Execution timeout
- Captured stdout/stderr
- Exception handling

**Execution Context:**
- Pre-loaded with `robodm` interface
- Safe mathematical and utility functions
- No access to dangerous operations

### 6. RoboDMMCPServer (`mcp/server.py`)

Model Context Protocol server for integration with AI assistants.

**MCP Tools Exposed:**
- All RoboDMInterface functions as MCP tools
- JSON schema validation for parameters
- Resource listing and reading capabilities

## Development Setup

### Prerequisites

1. **Python Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

2. **Install Dependencies**:
   ```bash
   cd robodm-agentic
   pip install -r requirements.txt
   ```

3. **Install Ollama** (for local models):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull qwen2.5:7b
   ollama pull llava:7b
   ```

4. **Run Setup Script**:
   ```bash
   ./setup.sh
   ```

### Development Workflow

1. **Make Changes**: Edit source files in the appropriate directories
2. **Test Locally**: Run `python example_usage.py` to test changes
3. **Add Tests**: Add test cases for new functionality
4. **Documentation**: Update docstrings and documentation
5. **Lint Code**: Ensure code follows style guidelines

### Environment Variables

Set these environment variables for development:

```bash
# For OpenAI integration
export OPENAI_API_KEY="your-api-key"

# For custom model endpoints
export LLM_BASE_URL="http://localhost:11434"  # Ollama default
export VLM_BASE_URL="http://localhost:11434"

# For debugging
export ROBODM_AGENTIC_DEBUG=1
export ROBODM_AGENTIC_LOG_LEVEL=DEBUG
```

## Design Patterns

### 1. Factory Pattern

Used for creating clients based on provider type:

```python
def create_llm_client(provider: str, **kwargs) -> LLMClient:
    if provider == "ollama":
        return OllamaLLMClient(**kwargs)
    elif provider == "openai":
        return OpenAILLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### 2. Strategy Pattern

Different execution strategies for code safety:

```python
class CodeExecutor:           # Basic executor
class RestrictedCodeExecutor: # Production-safe executor
class SandboxedCodeExecutor:  # Future: containerized execution
```

### 3. Adapter Pattern

RoboDMInterface adapts the core RoboDM API for agentic use:

```python
class RoboDMInterface:
    def __init__(self, data_path):
        self._trajectories = self._discover_trajectories(data_path)
    
    def get_trajectory_frames(self, traj_id):
        # Adapts robodm.Trajectory.load() for frame extraction
```

### 4. Observer Pattern

For monitoring query execution and performance:

```python
class QueryObserver:
    def on_query_start(self, query): pass
    def on_code_generated(self, code): pass
    def on_execution_complete(self, result): pass
```

## Extension Points

### 1. Adding New LLM Providers

Create a new client class implementing the LLM interface:

```python
class CustomLLMClient(LLMClient):
    async def generate_query_code(self, user_query, available_functions):
        # Implement your provider's API calls
        pass
```

### 2. Adding New Functions to RoboDM Interface

Extend the interface with custom analysis functions:

```python
class ExtendedRoboDMInterface(RoboDMInterface):
    def custom_analysis_function(self, trajectory_id):
        # Your custom logic here
        pass
    
    def get_available_functions(self):
        functions = super().get_available_functions()
        functions["custom_analysis_function"] = "Description of function"
        return functions
```

### 3. Custom Code Execution Environments

Create specialized executors for different use cases:

```python
class MLCodeExecutor(CodeExecutor):
    def __init__(self, robodm_interface):
        super().__init__(robodm_interface)
        # Add ML libraries to safe imports
        self.add_safe_import("sklearn")
        self.add_safe_import("numpy", "np")
```

### 4. Custom MCP Tools

Add specialized tools to the MCP server:

```python
@mcp_server.tool()
async def custom_trajectory_analysis(trajectory_id: str) -> Dict:
    # Custom analysis logic
    return {"analysis": "result"}
```

## Testing

### Unit Tests

Create test files following the pattern `test_<module>.py`:

```python
# tests/test_agent.py
import pytest
from robodm_agentic.core.agent import RoboDMAgent

class TestRoboDMAgent:
    def test_query_basic(self):
        # Test basic query functionality
        pass
```

### Integration Tests

Test complete workflows:

```python
# tests/test_integration.py
async def test_end_to_end_query():
    # Test complete query workflow
    pass
```

### Manual Testing

Use the example scripts for manual testing:

```bash
# Test basic functionality
python example_usage.py

# Test with OXE data
python ingest_oxe_demo.py
```

## Model Context Protocol

### MCP Architecture

The framework implements MCP to enable integration with AI assistants:

```
AI Assistant (Claude, GPT-4, etc.)
         │
         ▼
   MCP Client
         │
         ▼ (stdio/websocket)
   RoboDMMCPServer
         │
         ▼
   RoboDMInterface
         │
         ▼
     RoboDM Data
```

### MCP Tools

Each RoboDMInterface function is exposed as an MCP tool with:
- JSON schema for parameter validation
- Proper error handling
- Serialization for complex data types

### MCP Resources

Trajectories are exposed as MCP resources:
- URI format: `robodm://trajectory/{id}`
- Metadata available through resource reading
- Summary resource for database overview

## Contributing

### Code Style

Follow these guidelines:
- Use type hints for all function parameters and returns
- Document all public methods with docstrings
- Follow PEP 8 naming conventions
- Keep functions focused and small

### Pull Request Process

1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit pull request with clear description
5. Address review feedback

### Commit Messages

Use conventional commit format:
```
feat: add support for new VLM provider
fix: handle empty trajectory gracefully
docs: update API documentation
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure robodm-agentic is in Python path
   - Check virtual environment activation
   - Verify dependencies are installed

2. **Model Connection Issues**:
   - Verify Ollama is running: `ollama list`
   - Check API keys for cloud providers
   - Test model endpoints manually

3. **Code Execution Failures**:
   - Check generated code syntax
   - Verify RoboDM interface functions
   - Review execution logs

4. **Performance Issues**:
   - Limit number of frames for vision analysis
   - Use sampling for large datasets
   - Consider async processing for batch queries

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('robodm_agentic').setLevel(logging.DEBUG)
```

### Profiling

Profile query execution:

```python
import cProfile
cProfile.run('asyncio.run(agent.query("your query"))')
```

## Future Enhancements

### Planned Features

1. **Distributed Processing**: Support for Ray or Dask for large-scale queries
2. **Web Interface**: Browser-based query interface
3. **Query Optimization**: Caching and query planning
4. **Custom Models**: Fine-tuned models for robotics domain
5. **Real-time Queries**: Streaming query processing
6. **Advanced Security**: Containerized code execution

### Research Directions

1. **Multimodal Reasoning**: Better integration of vision and language
2. **Query Planning**: Automatic optimization of complex queries
3. **Domain Adaptation**: Robotics-specific model training
4. **Federated Learning**: Privacy-preserving distributed queries

---

For more information or questions, please refer to the main README.md or open an issue in the repository.
