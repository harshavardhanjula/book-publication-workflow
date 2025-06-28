"""AI Writer module for content generation and rewriting."""
import asyncio
import json
from typing import Dict, List, Optional
import aiohttp
from loguru import logger

from src.config import settings


class AIWriter:
    """Handles AI-powered content generation and rewriting."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the AI Writer.
        
        Args:
            model: The model to use for generation
            api_key: OpenRouter API key (uses settings if None)
        """
        self.model = model or settings.DEFAULT_AI_MODEL
        # Temporarily use hardcoded API key for testing
        self.api_key = "api_key"
        self.base_url = "https://openrouter.ai/api/v1"
        
        logger.info(f"Initialized AIWriter with model: {self.model}")
    
    async def _make_api_request(self, messages: List[Dict], **kwargs) -> Dict:
        """Make an API request to OpenRouter with optimized parameters.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dict containing the API response
            
        Raises:
            ValueError: If the request is invalid or the response is malformed
            Exception: For HTTP or network errors
        """
        if not self.api_key:
            raise ValueError("No API key provided")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Automated Book Publication"
        }
        
        # Prepare the payload with optimized parameters
        payload = {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",  # Free model with 256K context
            "messages": messages,
            "temperature": 0.65,             # Creative but grounded
            "top_p": 0.9,                   # Diverse ideas within relevance
            "top_k": 50,                    # Prioritize top 50 likely tokens
            "frequency_penalty": 0.3,       # Reduce repeated phrases
            "presence_penalty": 0.25,       # Encourage novelty
            "repetition_penalty": 1.1,      # Slight push against loops
            "min_p": 0.1,                   # Filter low-probability junk

            **kwargs
        }
        
        # Log the request (without sensitive data)
        safe_payload = payload.copy()
        safe_payload['messages'] = [
            {k: v for k, v in msg.items() if k != 'content' or not isinstance(v, str) or len(v) < 100}
            for msg in safe_payload['messages']
        ]
        logger.debug(f"Sending request to {self.base_url}/chat/completions")
        logger.trace(f"Request payload: {json.dumps(safe_payload, indent=2)}")
        
        # Debug: Print API key for verification (only first 5 characters for security)
        if self.api_key:
            logger.debug(f"Using API key: {self.api_key[:5]}...{self.api_key[-4:]}")
        else:
            logger.error("No API key found in settings or constructor")
            raise ValueError("No OpenRouter API key provided. Please set OPENROUTER_API_KEY in your .env file.")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # Increased timeout to 120 seconds
                ) as response:
                    response_text = await response.text()
                    
                    # Log response time
                    response_time = asyncio.get_event_loop().time() - start_time
                    logger.debug(f"API response received in {response_time:.2f} seconds")
                    
                    # Handle non-200 responses
                    if response.status != 200:
                        error_msg = f"API request failed with status {response.status}"
                        try:
                            error_data = json.loads(response_text)
                            error_msg += f": {error_data.get('error', {}).get('message', 'Unknown error')}"
                            logger.error(f"{error_msg}. Error type: {error_data.get('error', {}).get('type', 'unknown')}")
                        except json.JSONDecodeError:
                            error_msg += f": {response_text}"
                        
                        logger.error(f"Full error response: {response_text}")
                        raise Exception(error_msg)
                    
                    # Parse the response
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse API response: {e}")
                        logger.error(f"Response status: {response.status}")
                        logger.error(f"Response headers: {dict(response.headers)}")
                        logger.error(f"Response text (truncated): {response_text[:500]}..." if len(response_text) > 500 else response_text)
                        raise ValueError("Invalid JSON response from API")
                    
                    # Log the response structure for debugging
                    logger.trace(f"API response structure: {list(response_data.keys())}")
                    
                    # Validate response structure
                    if 'choices' not in response_data or not response_data['choices']:
                        error_msg = f"Unexpected API response format: 'choices' not found"
                        logger.error(f"{error_msg}. Response: {response_data}")
                        raise ValueError(error_msg)
                    
                    if not isinstance(response_data['choices'], list) or not response_data['choices']:
                        error_msg = f"Unexpected API response format: 'choices' is not a list or is empty"
                        logger.error(f"{error_msg}. Choices: {response_data['choices']}")
                        raise ValueError(error_msg)
                    
                    logger.debug(f"API request successful. Model: {response_data.get('model', 'unknown')}")
                    return response_data
                    
        except asyncio.TimeoutError:
            error_msg = f"API request timed out after 120 seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error during API request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    async def rewrite_chapter(
        self,
        content: str,
        style_guide: Dict,
        chapter_title: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """Rewrite a chapter according to the provided style guide.
        
        Args:
            content: The content to rewrite
            style_guide: Style guide to follow
            chapter_title: Optional title of the chapter
            instruction: Optional refinement instruction from the user
            **kwargs: Additional parameters for the AI model
            
        Returns:
            Dict containing the rewritten content and metadata
        """
        if not self.api_key:
            error_msg = "No OpenRouter API key provided. Please set the OPENROUTER_API_KEY environment variable."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "content": content
            }
        
        try:
            # Calculate target length (add 10% buffer for variations)
            target_length = int(len(content) * 1.1)
            
            # Prepare the prompt with strict instructions
            system_prompt = (
                "You are a highly skilled literary editor. Your task is to rewrite the provided content "
                "to match the given style guide while MAINTAINING THE ORIGINAL LENGTH and meaning. "
                "IMPORTANT RULES:\n"
                "1. Preserve all key plot points, descriptions, and dialogue exactly as they are\n"
                "2. Maintain the original structure and scene order\n"
                "3. Do not add new content or story elements\n"
                "4. Do not summarize or shorten the content\n"
                "5. Focus on improving clarity, flow, and style while keeping the original voice\n"
                "6. Ensure the output is approximately the same length as the input\n"
                "7. NEVER repeat phrases or get stuck in loops\n"
                "8. If given a refinement instruction, follow it precisely while respecting these rules"
            )
            
            # Build the user prompt with context
            prompt = self._build_rewrite_prompt(content, style_guide, chapter_title, instruction)
            logger.debug(f"Sending rewrite request for chapter: {chapter_title or 'Untitled'}")
            
            # Prepare messages with enhanced context
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Add refinement instruction as a separate message if provided
            if instruction:
                messages.append({
                    "role": "user",
                    "content": (
                        f"REFINEMENT INSTRUCTION: {instruction}. "
                        "IMPORTANT RULES TO FOLLOW:\n"
                        "1. Maintain the original content length and structure\n"
                        "2. Preserve all key story elements and dialogue\n"
                        "3. Do not add new content or story elements\n"
                        "4. Do not summarize or shorten the content\n"
                        "5. Never repeat phrases or get stuck in loops\n"
                        "6. Ensure the output is coherent and makes sense\n"
                        "7. If you reach the end of the content, stop generating"
                    )
                })
            
            # Calculate max tokens based on content length (1 token ~= 4 chars in English)
            # Add 20% buffer to account for variations in tokenization
            max_tokens = min(4096, int(len(content) * 0.3) + 500)
            
            # Set a hard limit on max tokens to prevent excessive generation
            max_tokens = min(max_tokens, 2048)
            
            # Make the API request with optimized parameters
            response = await self._make_api_request(
                messages=messages,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            
            # Log minimal response info
            logger.debug(f"API Response received. Model: {response.get('model', 'unknown')}, Tokens: {response.get('usage', {}).get('total_tokens', 'unknown')}")
            
            # Extract the generated content with better error handling
            if 'choices' not in response or not response['choices']:
                error_msg = f"Invalid API response: 'choices' not found in response"
                logger.error(f"{error_msg}. Full response: {response}")
                raise ValueError(error_msg)
                
            if not response['choices'] or 'message' not in response['choices'][0]:
                error_msg = f"Invalid API response: 'message' not found in choices"
                logger.error(f"{error_msg}. Choices: {response['choices']}")
                raise ValueError(error_msg)
                
            if 'content' not in response['choices'][0]['message']:
                error_msg = f"Invalid API response: 'content' not found in message"
                logger.error(f"{error_msg}. Message: {response['choices'][0]['message']}")
                raise ValueError(error_msg)
            
            generated_text = response['choices'][0]['message']['content']
            
            if not generated_text or not isinstance(generated_text, str):
                error_msg = "Generated content is empty or invalid"
                logger.error(f"{error_msg}. Content: {generated_text}")
                raise ValueError(error_msg)
                
            # Check for repeating patterns that might indicate a loop
            if self._detect_repetition(generated_text):
                error_msg = "Detected repetitive pattern in generated content"
                logger.error(error_msg)
                # Try to recover by truncating at the last good paragraph
                paragraphs = generated_text.split('\n\n')
                if len(paragraphs) > 1:
                    generated_text = '\n\n'.join(paragraphs[:-1])
                    logger.info("Attempting recovery by truncating the last paragraph")
                else:
                    raise ValueError("Cannot recover from repetition - content is too corrupted")
            
            result = {
                "success": True,
                "content": generated_text,
                "model": self.model,
                "usage": response.get('usage', {})
            }
            
            logger.debug(f"Successfully rewrote chapter. New length: {len(generated_text)} characters")
            return result
            
        except Exception as e:
            error_msg = f"Error in rewrite_chapter: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Include full traceback
            return {
                "success": False,
                "error": error_msg,
                "content": content,
                "model": self.model
            }
    
    def _detect_repetition(self, text: str, min_repeats: int = 3, min_length: int = 20) -> bool:
        """Detect if text contains repeating patterns that might indicate a loop.
        
        Args:
            text: The text to analyze
            min_repeats: Minimum number of repeats to consider it a loop
            min_length: Minimum length of the repeating pattern
            
        Returns:
            bool: True if repetition is detected, False otherwise
        """
        # Check for repeated phrases longer than min_length
        for length in range(min_length, min(100, len(text) // 2)):
            for i in range(len(text) - length * min_repeats):
                chunk = text[i:i+length]
                # If we see the same chunk repeated multiple times in a row
                if chunk * min_repeats in text:
                    logger.warning(f"Detected repetition of chunk: {chunk[:50]}...")
                    return True
        return False
        
    def _build_rewrite_prompt(
        self,
        content: str,
        style_guide: Dict,
        chapter_title: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:
        """Build the prompt for rewriting content.
        
        Args:
            content: The content to be rewritten
            style_guide: Dictionary containing style guidelines
            chapter_title: Optional title of the chapter
            instruction: Optional refinement instruction from the user
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Please rewrite the following chapter to match the provided style guide.",
            "Important: Maintain the original content length and all key story elements.",
            "Do not summarize, shorten, or add a word count.",
            "Focus on enhancing the prose while keeping the original structure and pacing.",
        ]
        
        # Add style guide components if provided
        if style_guide:
            prompt_parts.append("\n=== STYLE GUIDE (MUST FOLLOW) ===")
            if "tone" in style_guide:
                prompt_parts.append(f"TONE: {style_guide['tone'].upper()}")
            if "audience" in style_guide:
                prompt_parts.append(f"TARGET AUDIENCE: {style_guide['audience'].upper()}")
            if "style_rules" in style_guide:
                if isinstance(style_guide['style_rules'], list):
                    prompt_parts.extend([f"- {rule}" for rule in style_guide['style_rules']])
                else:
                    prompt_parts.append(style_guide['style_rules'])
                
                # Add strict length preservation rule
                prompt_parts.append(
                    "- PRESERVE ORIGINAL LENGTH: The output must be approximately the same length as the input. "
                    "Do not add new content or summarize existing content."
                )
        
        # Add chapter title if available
        if chapter_title:
            prompt_parts.append(f"\n=== CHAPTER: {chapter_title.upper()} ===")
        else:
            prompt_parts.append("\n=== ORIGINAL CONTENT (REWRITE THIS) ===")
        
        # Add the content to be rewritten with clear markers
        prompt_parts.append("```")
        prompt_parts.append(content)
        prompt_parts.append("```")
        prompt_parts.append("\n=== BEGIN YOUR REWRITE BELOW ===\n")
        
        # Add refinement instruction if provided
        if instruction:
            prompt_parts.extend([
                "\n=== REFINEMENT INSTRUCTION ===",
                f"Please refine the content according to this instruction: {instruction}",
                "Make sure to maintain the core meaning while applying the requested changes."
            ])
        
        # Add strict instructions for the rewrite
        prompt_parts.extend([
            "\n=== STRICT INSTRUCTIONS (MUST FOLLOW) ===",
            "1. REWRITE THE CONTENT to match the style guide while PRESERVING ALL ORIGINAL MEANING AND LENGTH",
            "2. DO NOT add any new scenes, characters, or plot points",
            "3. DO NOT summarize or shorten the content",
            "4. DO NOT include any commentary, notes, or section headers in your response",
            "5. DO NOT get stuck in loops or repeat phrases",
            "6. If the content contains dialogue, PRESERVE IT EXACTLY as is",
            "7. If the content contains specific terms or names, DO NOT change them",
            "8. Your output should be approximately the same length as the input",
            "9. If you receive a refinement instruction, follow it while respecting these rules",
            "10. Your response should be a clean, direct rewrite with NO additional formatting"
        ])
        
        # Add final instruction
        prompt_parts.append(
            "\n=== YOUR REWRITE ===\n"
            "Rewrite the content EXACTLY as instructed above. Your response must be a clean, direct "
            "rewrite with NO additional commentary, notes, or formatting. Begin your response with the "
            "first word of the rewritten content."
        )
        
        return "\n".join(prompt_parts)
    
    async def generate_chapter(
        self,
        outline: str,
        previous_chapter: Optional[str] = None,
        next_chapter_outline: Optional[str] = None,
        style_guide: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Generate a new chapter based on an outline."""
        if not self.api_key:
            return {
                "success": False,
                "error": "No API key provided",
                "content": ""
            }
        
        try:
            # Prepare the prompt
            prompt = self._build_generation_prompt(
                outline=outline,
                previous_chapter=previous_chapter,
                next_chapter_outline=next_chapter_outline,
                style_guide=style_guide or {}
            )
            
            # Call the AI model
            response = await self._make_api_request(
                messages=[
                    {"role": "system", "content": "You are a professional author."},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            # Extract the generated content
            generated_text = response['choices'][0]['message']['content']
            
            return {
                "success": True,
                "content": generated_text,
                "model": self.model,
                "usage": response.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Error in generate_chapter: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }
    
    def _build_generation_prompt(
        self,
        outline: str,
        previous_chapter: Optional[str] = None,
        next_chapter_outline: Optional[str] = None,
        style_guide: Optional[Dict] = None
    ) -> str:
        """Build the prompt for generating a new chapter."""
        prompt_parts = [
            "Please write a chapter based on the following outline.",
            "Ensure the chapter is well-written and engaging.",
            "\n=== CHAPTER OUTLINE ==="
        ]
        
        # Add the outline
        prompt_parts.append(outline)
        
        # Add context from previous chapter if available
        if previous_chapter:
            prompt_parts.extend([
                "\n=== PREVIOUS CHAPTER (for context) ===",
                previous_chapter[:2000] + "..." if len(previous_chapter) > 2000 else previous_chapter
            ])
        
        # Add next chapter outline if available
        if next_chapter_outline:
            prompt_parts.extend([
                "\n=== NEXT CHAPTER OUTLINE (for foreshadowing) ===",
                next_chapter_outline
            ])
        
        # Add style guide if provided
        if style_guide:
            prompt_parts.append("\n=== STYLE GUIDE ===")
            if "tone" in style_guide:
                prompt_parts.append(f"Tone: {style_guide['tone']}")
            if "audience" in style_guide:
                prompt_parts.append(f"Target Audience: {style_guide['audience']}")
            if "style_rules" in style_guide:
                if isinstance(style_guide['style_rules'], list):
                    prompt_parts.extend([f"- {rule}" for rule in style_guide['style_rules']])
                else:
                    prompt_parts.append(style_guide['style_rules'])
        
        # Add instructions
        prompt_parts.extend([
            "\n=== INSTRUCTIONS ===",
            "Please write the chapter following these guidelines:",
            "1. Follow the outline closely",
            "2. Maintain consistency with the previous chapter (if provided)",
            "3. Include subtle foreshadowing for the next chapter (if outline provided)",
            "4. Ensure the writing is engaging and appropriate for the target audience",
            "5. Follow the style guide (if provided)",
            "\nReturn ONLY the chapter content, without any additional commentary or formatting."
        ])
        
        return "\n".join(prompt_parts)
