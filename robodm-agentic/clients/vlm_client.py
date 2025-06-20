"""Vision-Language Model client for analyzing trajectory frames."""

import base64
import io
import asyncio
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class VLMClient:
    """Client for vision-language model analysis of trajectory frames."""
    
    def __init__(self, 
                 model: str = "llava:7b", 
                 provider: str = "ollama",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize VLM client.
        
        Args:
            model: Vision-language model name
            provider: "ollama" or "openai" 
            api_key: API key for hosted providers
            base_url: Base URL for API
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
        
        if not HAS_PIL:
            logger.warning("PIL not available. Image processing will be limited.")
        if not HAS_NUMPY:
            logger.warning("NumPy not available. Array processing will be limited.")
    
    async def analyze_and_answer(self, 
                                query: str, 
                                frames: List[Any], 
                                execution_result: Any,
                                max_frames: int = 5) -> str:
        """Analyze frames and answer the original query."""
        
        if not frames:
            return f"Query executed successfully. Result: {execution_result}"
        
        # Limit number of frames to analyze
        frames_to_analyze = frames[:max_frames]
        logger.info(f"Analyzing {len(frames_to_analyze)} frames for query: {query}")
        
        try:
            # Convert frames to base64 for vision model
            encoded_frames = self._encode_frames(frames_to_analyze)
            
            if not encoded_frames:
                return f"Could not process frames. Execution result: {execution_result}"
            
            prompt = self._build_analysis_prompt(query, execution_result, len(frames_to_analyze))
            
            if self.provider == "ollama":
                response = await self._call_ollama_vision(prompt, encoded_frames)
            elif self.provider == "openai":
                response = await self._call_openai_vision(prompt, encoded_frames)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            return response
            
        except Exception as e:
            logger.error(f"Error in vision analysis: {e}")
            return f"Vision analysis failed: {e}. Execution result: {execution_result}"
    
    def _build_analysis_prompt(self, query: str, execution_result: Any, num_frames: int) -> str:
        """Build prompt for vision analysis."""
        return f"""You are analyzing robotics trajectory data to answer a user's question.

Original Query: {query}
Code Execution Result: {execution_result}
Number of frames provided: {num_frames}

Please analyze the provided images/frames and give a comprehensive answer to the original query. 

Instructions:
1. Examine the visual content of the frames carefully
2. Look for patterns, objects, actions, or anomalies relevant to the query
3. Relate your visual observations back to the original question
4. If the query is about finding specific trajectories, explain what visual evidence supports the results
5. If you see robotic actions, describe what the robot appears to be doing
6. Be specific about what you observe in the images

Provide a clear, detailed response that combines the execution results with your visual analysis."""
    
    async def _call_ollama_vision(self, prompt: str, encoded_frames: List[str]) -> str:
        """Call Ollama vision model."""
        try:
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": encoded_frames
                }]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama vision API error: {e}")
            raise
    
    async def _call_openai_vision(self, prompt: str, encoded_frames: List[str]) -> str:
        """Call OpenAI vision model."""
        try:
            # Build content with images for OpenAI format
            content = [{"type": "text", "text": prompt}]
            
            for frame_b64 in encoded_frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}"
                    }
                })
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user", 
                    "content": content
                }]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI vision API error: {e}")
            raise
    
    def _encode_frames(self, frames: List[Any]) -> List[str]:
        """Encode frames as base64 strings."""
        encoded = []
        
        for i, frame in enumerate(frames):
            try:
                encoded_frame = self._frame_to_base64(frame)
                if encoded_frame:
                    encoded.append(encoded_frame)
            except Exception as e:
                logger.warning(f"Failed to encode frame {i}: {e}")
                continue
                
        return encoded
    
    def _frame_to_base64(self, frame: Any) -> Optional[str]:
        """Convert frame to base64 string."""
        try:
            # Handle different frame formats
            if HAS_NUMPY and isinstance(frame, np.ndarray):
                return self._numpy_to_base64(frame)
            elif HAS_PIL and isinstance(frame, Image.Image):
                return self._pil_to_base64(frame)
            elif isinstance(frame, bytes):
                return base64.b64encode(frame).decode('utf-8')
            else:
                # Try to convert to PIL Image if possible
                if HAS_PIL:
                    try:
                        if hasattr(frame, 'save'):  # Looks like an image
                            return self._pil_to_base64(frame)
                        else:
                            # Try to create PIL image from data
                            img = Image.fromarray(frame)
                            return self._pil_to_base64(img)
                    except Exception:
                        pass
                        
                logger.warning(f"Unsupported frame type: {type(frame)}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return None
    
    def _numpy_to_base64(self, array: 'np.ndarray') -> str:
        """Convert numpy array to base64."""
        if not HAS_PIL:
            raise ImportError("PIL required for numpy array conversion")
            
        # Ensure array is in correct format for PIL
        if array.dtype != np.uint8:
            # Normalize to 0-255 range
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        
        # Handle different array shapes
        if len(array.shape) == 3 and array.shape[2] in [1, 3, 4]:
            # RGB/RGBA image
            if array.shape[2] == 1:
                # Grayscale
                array = np.squeeze(array, axis=2)
                img = Image.fromarray(array, mode='L')
            else:
                img = Image.fromarray(array)
        elif len(array.shape) == 2:
            # Grayscale image
            img = Image.fromarray(array, mode='L')
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        
        return self._pil_to_base64(img)
    
    def _pil_to_base64(self, img: 'Image.Image') -> str:
        """Convert PIL Image to base64."""
        buffer = io.BytesIO()
        
        # Convert to RGB if necessary
        if img.mode in ['RGBA', 'LA']:
            img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
            
        img.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    async def test_connection(self) -> bool:
        """Test if the VLM connection is working."""
        try:
            # Create a simple test image
            if HAS_PIL and HAS_NUMPY:
                test_array = np.zeros((64, 64, 3), dtype=np.uint8)
                test_array[20:40, 20:40] = [255, 0, 0]  # Red square
                
                test_response = await self.analyze_and_answer(
                    "What do you see in this test image?",
                    [test_array],
                    "test"
                )
                return len(test_response) > 0
            else:
                logger.warning("Cannot test VLM connection without PIL and NumPy")
                return False
        except Exception as e:
            logger.error(f"VLM connection test failed: {e}")
            return False
