"""
URL Summary Bot - Using FREE APIs
- Hugging Face for LLM (free inference API)
- BeautifulSoup + Requests for web scraping (completely free)
- Jina AI Reader API (free tier) as alternative for better content extraction
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
import os
import time


class URLSummaryBot:
    """
    A bot that fetches content from URLs and generates summaries using
    Hugging Face's free inference API.
    """
    
    def __init__(self, hf_api_key: Optional[str] = None):
        """
        Initialize the bot with Hugging Face API key.
        
        Args:
            hf_api_key: Hugging Face API key (get free at huggingface.co)
        """
        self.hf_api_key = hf_api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.hf_api_url = "https://api-inference.huggingface.co/models/"
        
        # Default to a good free summarization model
        self.model = "facebook/bart-large-cnn"  # Good for summarization
        # Alternative models:
        # "google/pegasus-xsum" - Extreme summarization
        # "philschmid/bart-large-cnn-samsum" - Conversational
        # "Falconsai/text_summarization" - General purpose
        
    def fetch_url_content(self, url: str, use_jina: bool = False) -> str:
        """
        Fetch and extract text content from a given URL.
        
        Args:
            url: The URL to fetch content from
            use_jina: If True, use Jina AI Reader API (free, better extraction)
            
        Returns:
            Extracted text content
        """
        if use_jina:
            return self._fetch_with_jina(url)
        else:
            return self._fetch_with_requests(url)
    
    def _fetch_with_jina(self, url: str) -> str:
        """
        Use Jina AI Reader API for better content extraction (FREE).
        Returns clean markdown content without ads/navigation.
        """
        try:
            # Jina Reader API - completely free, no API key needed
            jina_url = f"https://r.jina.ai/{url}"
            response = requests.get(jina_url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Jina API failed, falling back to requests: {str(e)}")
            return self._fetch_with_requests(url)
    
    def _fetch_with_requests(self, url: str) -> str:
        """
        Basic content extraction using requests + BeautifulSoup (FREE).
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                element.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error fetching URL content: {str(e)}")
    
    def generate_summary(
        self,
        url: str,
        length: str = "medium",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        style_example: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        use_jina: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a summary of content from a URL using Hugging Face.
        
        Args:
            url: The URL to summarize
            length: Predefined length ('short', 'medium', 'long')
            max_length: Maximum length in tokens (overrides length preset)
            min_length: Minimum length in tokens
            style_example: Optional example text for style reference
            custom_instructions: Additional instructions
            use_jina: Use Jina AI for better content extraction
            
        Returns:
            Dictionary containing the summary and metadata
        """
        # Fetch content
        print(f"Fetching content from {url}...")
        content = self.fetch_url_content(url, use_jina=use_jina)
        
        # Truncate content if too long (model limits)
        content = content[:10000]  # Reasonable limit for free API
        
        # Set length parameters
        if not max_length:
            length_map = {
                "short": (30, 80),
                "medium": (100, 200),
                "long": (250, 500)
            }
            min_length, max_length = length_map.get(length, length_map["medium"])
        
        # Build the input text with style instructions if provided
        input_text = content
        if style_example or custom_instructions:
            input_text = self._build_styled_prompt(
                content, style_example, custom_instructions
            )
        
        # Generate summary using Hugging Face
        print(f"Generating summary using Hugging Face ({self.model})...")
        try:
            summary = self._call_huggingface_api(
                input_text,
                max_length=max_length,
                min_length=min_length
            )
            
            return {
                "success": True,
                "url": url,
                "summary": summary,
                "length_target": length,
                "model": self.model,
                "style_applied": style_example is not None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def _call_huggingface_api(
        self,
        text: str,
        max_length: int = 200,
        min_length: int = 50
    ) -> str:
        """
        Call Hugging Face Inference API (FREE).
        """
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}"
        }
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
                "temperature": 0.7
            }
        }
        
        api_url = f"{self.hf_api_url}{self.model}"
        
        # Retry logic for model loading
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("summary_text", result[0].get("generated_text", ""))
                return str(result)
            
            elif response.status_code == 503:
                # Model is loading, wait and retry
                print(f"Model loading... (attempt {attempt + 1}/{max_retries})")
                time.sleep(20)
            else:
                raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
        
        raise Exception("Model failed to load after multiple attempts")
    
    def _build_styled_prompt(
        self,
        content: str,
        style_example: Optional[str],
        custom_instructions: Optional[str]
    ) -> str:
        """
        Build a prompt that includes style guidance.
        Note: Basic summarization models may not follow complex instructions well.
        For better style control, consider using a text-generation model instead.
        """
        prompt_parts = []
        
        if custom_instructions:
            prompt_parts.append(f"Instructions: {custom_instructions}\n")
        
        if style_example:
            prompt_parts.append(
                f"Use this writing style: {style_example[:200]}...\n"
            )
        
        prompt_parts.append(content)
        
        return "\n".join(prompt_parts)
    
    def set_model(self, model_name: str):
        """
        Change the Hugging Face model.
        
        Popular free models for summarization:
        - facebook/bart-large-cnn (default, good quality)
        - google/pegasus-xsum (extreme summarization)
        - philschmid/bart-large-cnn-samsum (conversational)
        - sshleifer/distilbart-cnn-12-6 (faster, smaller)
        - Falconsai/text_summarization
        """
        self.model = model_name


# Alternative: Using Hugging Face for general text generation (better style control)
class URLSummaryBotWithGeneration(URLSummaryBot):
    """
    Extended bot using text generation models for better style control.
    """
    
    def __init__(self, hf_api_key: Optional[str] = None):
        super().__init__(hf_api_key)
        # Use a text generation model for better instruction following
        self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # Other free options:
        # "meta-llama/Llama-2-7b-chat-hf"
        # "HuggingFaceH4/zephyr-7b-beta"
        # "tiiuae/falcon-7b-instruct"
    
    def _call_huggingface_api(self, text: str, max_length: int = 200, min_length: int = 50) -> str:
        """
        Call HF API for text generation with custom prompt.
        """
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        # Build a proper instruction prompt
        prompt = f"""Summarize the following content in approximately {max_length} words. 
Keep it clear, concise, and informative.

Content:
{text[:4000]}

Summary:"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length * 2,  # Roughly tokens = words * 1.3
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        api_url = f"{self.hf_api_url}{self.model}"
        
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get("generated_text", "")
                    # Extract just the summary part (after the prompt)
                    if "Summary:" in generated:
                        return generated.split("Summary:")[-1].strip()
                    return generated
                return str(result)
            
            elif response.status_code == 503:
                print(f"Model loading... (attempt {attempt + 1}/{max_retries})")
                time.sleep(20)
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
        
        raise Exception("Model failed to load")
