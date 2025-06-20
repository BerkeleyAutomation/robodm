"""Main agentic interface for RoboDM trajectory querying."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .robodm_interface import RoboDMInterface
from .code_executor import CodeExecutor
from ..clients.llm_client import LLMClient
from ..clients.vlm_client import VLMClient

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of an agentic query."""
    query: str
    generated_code: str
    execution_result: Dict[str, Any]
    frames: List[Any]
    answer: str
    success: bool
    error: Optional[str] = None


class RoboDMAgent:
    """Main agent for querying RoboDM trajectories using natural language."""
    
    def __init__(self, 
                 robodm_interface: RoboDMInterface,
                 llm_client: Optional[LLMClient] = None,
                 vlm_client: Optional[VLMClient] = None,
                 enable_vision: bool = True):
        """Initialize the RoboDM agent.
        
        Args:
            robodm_interface: Interface to RoboDM data
            llm_client: LLM client for code generation
            vlm_client: VLM client for visual analysis
            enable_vision: Whether to enable vision analysis
        """
        self.robodm = robodm_interface
        self.llm = llm_client or LLMClient()
        self.vlm = vlm_client if enable_vision else None
        if enable_vision and vlm_client is None:
            try:
                self.vlm = VLMClient()
            except Exception as e:
                logger.warning(f"Could not initialize VLM client: {e}")
                self.vlm = None
                
        self.executor = CodeExecutor(self.robodm)
        self.logger = logging.getLogger(__name__)
        self.enable_vision = enable_vision and self.vlm is not None
    
    async def query(self, user_query: str, include_frames: bool = None) -> QueryResult:
        """Process a natural language query about trajectories.
        
        Args:
            user_query: Natural language query
            include_frames: Whether to extract and analyze frames (auto-detect if None)
            
        Returns:
            QueryResult with all execution details and answer
        """
        self.logger.info(f"Processing query: {user_query}")
        
        # Auto-detect if frames should be included
        if include_frames is None:
            include_frames = self._should_include_frames(user_query)
        
        try:
            # Step 1: Generate RoboDM query code using LLM
            self.logger.info("Generating code with LLM...")
            code = await self.llm.generate_query_code(
                user_query, 
                self.robodm.get_available_functions()
            )
            
            # Step 2: Execute the generated code
            self.logger.info("Executing generated code...")
            execution_result = await self.executor.execute(code)
            
            # Step 3: Extract frames if needed and vision is enabled
            frames = []
            if include_frames and self.enable_vision and execution_result['success']:
                frames = await self._extract_frames_from_result(execution_result)
            
            # Step 4: Use VLM to analyze frames and answer question (if vision enabled)
            if self.enable_vision and self.vlm and (frames or execution_result['success']):
                self.logger.info("Analyzing with VLM...")
                answer = await self.vlm.analyze_and_answer(
                    user_query, frames, execution_result
                )
            else:
                # Generate text-only answer
                answer = self._generate_text_answer(user_query, execution_result)
            
            return QueryResult(
                query=user_query,
                generated_code=code,
                execution_result=execution_result,
                frames=frames,
                answer=answer,
                success=execution_result['success']
            )
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return QueryResult(
                query=user_query,
                generated_code="",
                execution_result={'success': False, 'error': str(e)},
                frames=[],
                answer=f"Query failed: {str(e)}",
                success=False,
                error=str(e)
            )
    
    def _should_include_frames(self, query: str) -> bool:
        """Determine if query likely needs visual analysis."""
        vision_keywords = [
            'frame', 'image', 'visual', 'see', 'look', 'appearance', 
            'color', 'object', 'robot', 'action', 'movement', 'gesture',
            'hidden', 'occluded', 'view', 'camera', 'scene', 'environment'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in vision_keywords)
    
    async def _extract_frames_from_result(self, execution_result: Dict[str, Any]) -> List[Any]:
        """Extract visual frames from execution result."""
        frames = []
        
        if not execution_result['success']:
            return frames
            
        try:
            # If the result contains trajectory IDs, try to get frames from them
            result_data = execution_result.get('result', [])
            
            if isinstance(result_data, list) and result_data:
                # Assume these are trajectory IDs
                sample_trajectories = result_data[:3]  # Limit to first 3 for performance
                
                for traj_id in sample_trajectories:
                    if isinstance(traj_id, str):
                        try:
                            traj_frames = self.robodm.get_trajectory_frames(traj_id)
                            if traj_frames:
                                # Take a few frames from each trajectory
                                sample_frames = traj_frames[:5] if len(traj_frames) > 5 else traj_frames
                                frames.extend(sample_frames)
                        except Exception as e:
                            self.logger.warning(f"Could not get frames from {traj_id}: {e}")
                            continue
                            
        except Exception as e:
            self.logger.warning(f"Error extracting frames: {e}")
            
        return frames
    
    def _generate_text_answer(self, query: str, execution_result: Dict[str, Any]) -> str:
        """Generate text-only answer when vision is not available."""
        if not execution_result['success']:
            return f"Query execution failed: {execution_result.get('error', 'Unknown error')}"
        
        output = execution_result.get('output', '').strip()
        result = execution_result.get('result')
        
        # Combine output and result into a coherent answer
        answer_parts = []
        
        if output:
            answer_parts.append("Execution output:")
            answer_parts.append(output)
        
        if result is not None:
            if isinstance(result, list):
                answer_parts.append(f"\\nFound {len(result)} items matching your query.")
                if result and len(result) <= 10:
                    answer_parts.append("Items: " + ", ".join(str(item) for item in result))
            elif isinstance(result, (int, float)):
                answer_parts.append(f"\\nResult: {result}")
            else:
                answer_parts.append(f"\\nResult: {result}")
        
        if not answer_parts:
            answer_parts.append("Query executed successfully with no specific output.")
        
        return "\\n".join(answer_parts)
    
    async def batch_query(self, queries: List[str]) -> List[QueryResult]:
        """Process multiple queries in parallel.
        
        Args:
            queries: List of natural language queries
            
        Returns:
            List of QueryResult objects
        """
        self.logger.info(f"Processing {len(queries)} queries in batch")
        
        tasks = [self.query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(QueryResult(
                    query=queries[i],
                    generated_code="",
                    execution_result={'success': False, 'error': str(result)},
                    frames=[],
                    answer=f"Batch query failed: {result}",
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def test_setup(self) -> Dict[str, bool]:
        """Test all components to ensure they're working.
        
        Returns:
            Dictionary with test results for each component
        """
        results = {
            'robodm_interface': False,
            'llm_client': False,
            'vlm_client': False,
            'code_executor': False
        }
        
        # Test RoboDM interface
        try:
            trajectories = self.robodm.get_all_trajectories()
            results['robodm_interface'] = len(trajectories) >= 0
        except Exception as e:
            self.logger.error(f"RoboDM interface test failed: {e}")
        
        # Test LLM client
        try:
            results['llm_client'] = await self.llm.test_connection()
        except Exception as e:
            self.logger.error(f"LLM client test failed: {e}")
        
        # Test VLM client
        if self.vlm:
            try:
                results['vlm_client'] = await self.vlm.test_connection()
            except Exception as e:
                self.logger.error(f"VLM client test failed: {e}")
        
        # Test code executor
        try:
            test_code = "result = 2 + 2\\nprint('Test successful')"
            exec_result = await self.executor.execute(test_code)
            results['code_executor'] = exec_result['success'] and exec_result['result'] == 4
        except Exception as e:
            self.logger.error(f"Code executor test failed: {e}")
        
        return results
    
    def close(self):
        """Close all resources."""
        if hasattr(self.robodm, 'close_all'):
            self.robodm.close_all()
