"""LLM client for code generation using various models."""

import json
import asyncio
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class LLMClient:
    """Client for interacting with Language Models for code generation."""
    
    def __init__(self, 
                 model: str = "qwen2.5:7b", 
                 provider: str = "ollama",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize LLM client.
        
        Args:
            model: Model name/identifier
            provider: "ollama", "openai", or "anthropic"
            api_key: API key for hosted providers
            base_url: Base URL for API (for custom endpoints)
        """
        self.model = model
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        
        # Initialize client based on provider
        if self.provider == "ollama":
            if not HAS_OLLAMA:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        elif self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate_query_code(self, user_query: str, available_functions: Dict[str, str]) -> str:
        """Generate RoboDM query code based on user query."""
        
        system_prompt = self._build_system_prompt(available_functions)
        
        try:
            if self.provider == "ollama":
                response = await self._call_ollama(system_prompt, user_query)
            elif self.provider == "openai":
                response = await self._call_openai(system_prompt, user_query)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            return self._extract_code(response)
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return f"# Error generating code: {e}\\nprint('Query failed to generate code')"
    
    def _build_system_prompt(self, available_functions: Dict[str, str]) -> str:
        """Build system prompt with available functions and examples."""
        
        functions_doc = "\\n".join([f"- {name}: {desc}" for name, desc in available_functions.items()])
        
        return f"""You are a code generator for RoboDM trajectory queries. Generate Python code that uses the robodm interface to answer user queries about robotics trajectories.

Available RoboDM Interface Functions:
{functions_doc}

IMPORTANT RULES:
1. Use the variable name 'robodm' to access the interface (it's already initialized)
2. Return executable Python code only, no explanations or markdown
3. Always return a result that can be displayed to the user
4. Handle errors gracefully with try/except blocks
5. Use print() statements to show results to the user

Example queries and their corresponding code:

Query: "find failed trajectories"
Code:
```python
try:
    failed_trajs = []
    for traj_id in robodm.get_all_trajectories():
        if robodm.get_trajectory_status(traj_id) == 'failed':
            failed_trajs.append(traj_id)
    print(f"Found {len(failed_trajs)} failed trajectories:")
    for traj_id in failed_trajs:
        print(f"  - {traj_id}")
    result = failed_trajs
except Exception as e:
    print(f"Error finding failed trajectories: {e}")
    result = []
```

Query: "find trajectories with hidden views"
Code:
```python
try:
    hidden_view_trajs = []
    for traj_id in robodm.get_all_trajectories():
        try:
            data = robodm.get_trajectory_data(traj_id)
            # Look for features that might indicate hidden views
            features = list(data.keys())
            if any('hidden' in feat.lower() or 'occlud' in feat.lower() for feat in features):
                hidden_view_trajs.append(traj_id)
        except Exception:
            continue
    print(f"Found {len(hidden_view_trajs)} trajectories with potential hidden views:")
    for traj_id in hidden_view_trajs:
        print(f"  - {traj_id}")
    result = hidden_view_trajs
except Exception as e:
    print(f"Error searching for hidden views: {e}")
    result = []
```

Query: "count successful trajectories"
Code:
```python
try:
    success_count = robodm.count_trajectories({"status": "success"})
    total_count = robodm.count_trajectories()
    print(f"Successful trajectories: {success_count} out of {total_count}")
    result = success_count
except Exception as e:
    print(f"Error counting trajectories: {e}")
    result = 0
```

Query: "show me 5 random trajectories"
Code:
```python
try:
    sample_trajs = robodm.sample_trajectories(5)
    print(f"Random sample of {len(sample_trajs)} trajectories:")
    for traj_id in sample_trajs:
        metadata = robodm.get_trajectory_metadata(traj_id)
        length = metadata.get('length', 'unknown')
        print(f"  - {traj_id}: {length} timesteps")
    result = sample_trajs
except Exception as e:
    print(f"Error sampling trajectories: {e}")
    result = []
```
"""
    
    async def _call_ollama(self, system_prompt: str, user_query: str) -> str:
        """Call Ollama API."""
        try:
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def _call_openai(self, system_prompt: str, user_query: str) -> str:
        """Call OpenAI-compatible API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Remove markdown code blocks if present
        lines = response.strip().split('\\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                continue
            elif in_code_block or not any(line.strip().startswith(marker) for marker in ['```', '#', '*', '-'] if not line.strip().startswith('#')):
                code_lines.append(line)
        
        code = '\\n'.join(code_lines).strip()
        
        # If no code was extracted, return the original response
        if not code:
            code = response.strip()
            
        return code
    
    async def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        try:
            test_response = await self.generate_query_code(
                "test connection", 
                {"test": "test function"}
            )
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
