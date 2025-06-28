"""AI Reviewer module for content analysis and feedback generation."""
import json
import aiohttp
from typing import Dict, List, Optional
from loguru import logger

from src.config import settings

class AIReviewer:
    """Handles AI-powered content review and feedback generation."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the AI Reviewer.
        
        Args:
            model: The model to use for review
            api_key: OpenRouter API key (uses settings if None)
        """
        self.model = model or "deepseek/deepseek-chat-v3-0324:free"  # Good for analysis
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    async def review_chapter(
        self,
        content: str,
        style_guide: Dict,
        chapter_title: Optional[str] = None
    ) -> Dict:
        """Review chapter content and provide structured feedback.
        
        Args:
            content: The content to review
            style_guide: Style guide to evaluate against
            chapter_title: Optional title of the chapter
            
        Returns:
            Dict containing review results with score, feedback, and suggestions
        """
        # Build the review prompt
        system_prompt = self._build_review_system_prompt(style_guide)
        user_prompt = self._build_review_user_prompt(content, chapter_title)
        
        try:
            # Call the AI API
            response = await self._make_api_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse the response
            review_data = self._parse_review_response(response)
            return {
                "success": True,
                "score": review_data.get("overall_score", 0),
                "feedback": review_data.get("feedback", ""),
                "suggestions": review_data.get("suggestions", []),
                "metadata": {
                    "model": self.model,
                    "review_categories": review_data.get("categories", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error in review_chapter: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "score": 0,
                "feedback": "",
                "suggestions": []
            }
    
    def _build_review_system_prompt(self, style_guide: Dict) -> str:
        """Build the system prompt for the review."""
        return f"""You are an expert literary editor with deep expertise in content analysis. 
Your task is to provide a thorough, constructive review of the provided chapter content.

Review Guidelines:
1. Be specific and provide actionable feedback
2. Highlight both strengths and areas for improvement
3. Reference specific examples from the text
4. Consider the following style guide:
   - Tone: {style_guide.get('tone', 'professional')}
   - Audience: {style_guide.get('audience', 'general')}
   - Additional Rules: {', '.join(style_guide.get('style_rules', []))}

Your response MUST be a valid JSON object with these fields:
- "overall_score" (0-100): Overall quality rating
- "feedback": Detailed analysis (3-5 paragraphs)
- "suggestions": List of specific improvement suggestions (3-5 items)
- "categories": Object with scores (0-10) for different aspects:
  - "readability" (clarity, sentence structure, flow)
  - "style_adherence" (matches requested style)
  - "technical_quality" (grammar, spelling, mechanics)
  - "narrative_flow" (pacing, transitions, coherence)
  - "engagement" (compelling and interesting)

IMPORTANT: Your response must be valid JSON. Do not include any text outside the JSON object."""

    def _build_review_user_prompt(self, content: str, chapter_title: Optional[str] = None) -> str:
        """Build the user prompt for the review."""
        prompt = ["Please review the following chapter content:"]
        if chapter_title:
            prompt.append(f"Chapter Title: {chapter_title}")
        prompt.extend([
            "--- CONTENT STARTS ---",
            content[:15000],  # Limit content length
            "--- CONTENT ENDS ---",
            "\nPlease provide your review as a JSON object following the specified format:"
        ])
        return "\n".join(prompt)
    
    async def _make_api_request(self, messages: List[Dict]) -> Dict:
        """Make an API request to the AI service."""
        if not self.api_key:
            raise ValueError("No API key provided for AI Reviewer")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/your-repo",
            "X-Title": "Automated Book Publication Workflow"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more focused reviews
            "max_tokens": 2000,
            "response_format": { "type": "json_object" }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=60  # 60 seconds timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            raise Exception(f"Failed to get AI review: {str(e)}")
    
    def _parse_review_response(self, response: Dict) -> Dict:
        """Parse the AI response into a structured format."""
        try:
            content = response['choices'][0]['message']['content']
            
            # Handle cases where the model might wrap JSON in markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                # Handle case where language isn't specified
                parts = content.split('```')
                if len(parts) > 1:
                    content = parts[1].strip()
            
            # Parse the JSON content
            review_data = json.loads(content)
            
            # Validate required fields
            required_fields = ['overall_score', 'feedback', 'suggestions', 'categories']
            for field in required_fields:
                if field not in review_data:
                    raise ValueError(f"Missing required field in review: {field}")
            
            # Ensure scores are within valid ranges
            if not (0 <= review_data['overall_score'] <= 100):
                review_data['overall_score'] = max(0, min(100, review_data['overall_score']))
                
            # Ensure suggestions is a list
            if not isinstance(review_data['suggestions'], list):
                review_data['suggestions'] = [str(s) for s in review_data['suggestions']]
                
            return review_data
            
        except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to parse review response: {e}")
            logger.error(f"Original response: {content if 'content' in locals() else response}")
            raise ValueError("Invalid review response format from AI")
