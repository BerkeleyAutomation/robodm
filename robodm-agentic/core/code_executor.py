"""Code executor for running generated RoboDM queries safely."""

import asyncio
import sys
import io
import traceback
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Executes generated code safely with access to RoboDM interface."""
    
    def __init__(self, robodm_interface):
        """Initialize code executor with RoboDM interface.
        
        Args:
            robodm_interface: RoboDMInterface instance for data access
        """
        self.robodm_interface = robodm_interface
        self.execution_globals = {
            'robodm': robodm_interface,
            '__builtins__': {
                # Safe built-ins only
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'max': max,
                'min': min,
                'sum': sum,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'getattr': getattr,
            }
        }
    
    async def execute(self, code: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute generated code safely.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing code:\\n{code}")
        
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        execution_result = {
            'success': False,
            'result': None,
            'output': '',
            'error': None,
            'code': code
        }
        
        try:
            # Create local variables for execution
            exec_locals = {}
            
            # Run code with timeout
            await asyncio.wait_for(
                self._run_code(code, self.execution_globals, exec_locals),
                timeout=timeout
            )
            
            # Get the result if it was set
            if 'result' in exec_locals:
                execution_result['result'] = exec_locals['result']
            
            execution_result['success'] = True
            
        except asyncio.TimeoutError:
            execution_result['error'] = f"Code execution timed out after {timeout} seconds"
            logger.error(f"Code execution timeout: {timeout}s")
            
        except Exception as e:
            execution_result['error'] = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Code execution error: {e}\\n{traceback.format_exc()}")
            
        finally:
            # Restore stdout and capture output
            sys.stdout = old_stdout
            execution_result['output'] = captured_output.getvalue()
            captured_output.close()
        
        return execution_result
    
    async def _run_code(self, code: str, globals_dict: Dict, locals_dict: Dict):
        """Run code in a separate thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._execute_sync,
            code,
            globals_dict,
            locals_dict
        )
    
    def _execute_sync(self, code: str, globals_dict: Dict, locals_dict: Dict):
        """Synchronously execute code."""
        try:
            # Compile and execute the code
            compiled_code = compile(code, '<generated>', 'exec')
            exec(compiled_code, globals_dict, locals_dict)
        except Exception as e:
            # Re-raise the exception to be caught by the async wrapper
            raise e
    
    def add_safe_import(self, module_name: str, alias: Optional[str] = None):
        """Add a safe module to the execution environment.
        
        Args:
            module_name: Name of module to import
            alias: Alias for the module (optional)
        """
        try:
            module = __import__(module_name)
            import_name = alias or module_name
            self.execution_globals[import_name] = module
            logger.info(f"Added safe import: {module_name} as {import_name}")
        except ImportError:
            logger.warning(f"Could not import module: {module_name}")
    
    def add_safe_function(self, name: str, function):
        """Add a safe function to the execution environment.
        
        Args:
            name: Name to use in execution environment
            function: Function to add
        """
        self.execution_globals[name] = function
        logger.info(f"Added safe function: {name}")


class RestrictedCodeExecutor(CodeExecutor):
    """More restricted code executor for production use."""
    
    def __init__(self, robodm_interface):
        super().__init__(robodm_interface)
        
        # Remove potentially dangerous built-ins
        restricted_builtins = self.execution_globals['__builtins__'].copy()
        
        # Remove dangerous functions
        dangerous_functions = [
            'exec', 'eval', 'compile', 'open', 'input', 'raw_input',
            '__import__', 'globals', 'locals', 'vars', 'dir', 'help'
        ]
        
        for func in dangerous_functions:
            restricted_builtins.pop(func, None)
        
        self.execution_globals['__builtins__'] = restricted_builtins
    
    def _execute_sync(self, code: str, globals_dict: Dict, locals_dict: Dict):
        """Execute code with additional restrictions."""
        # Check for potentially dangerous code patterns
        dangerous_patterns = [
            'import ',
            'from ',
            '__',
            'exec(',
            'eval(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                raise SecurityError(f"Potentially dangerous code pattern detected: {pattern}")
        
        # Execute with restrictions
        super()._execute_sync(code, globals_dict, locals_dict)


class SecurityError(Exception):
    """Raised when potentially dangerous code is detected."""
    pass
