from typing import List, Dict, Any
from openai import OpenAI
import google.generativeai as genai
from config import Config
import streamlit as st


class LLMHandler:
    """Handle interactions with different LLM providers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_key = Config.LLM_MODELS.get(model_name)
        
        # Initialize clients based on model
        if "GPT" in model_name:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.provider = "openai"
        elif "Gemini" in model_name:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.client = genai.GenerativeModel(self.model_key)
            self.provider = "gemini"
        elif "DeepSeek" in model_name:
            self.client = OpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
            self.provider = "deepseek"
    
    def create_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Create a prompt with context for the LLM"""
        
        # Combine context chunks
        context = "\n\n".join([
            f"[Chunk {chunk['chunk_index']}]:\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a helpful assistant answering questions based on the provided document context.

Context from the document:
{context}

Question: {query}

Instructions:
- Answer the question based solely on the provided context
- If the answer is not in the context, say "I cannot find this information in the provided document"
- Be concise and accurate
- Cite the relevant chunk numbers when possible

Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a response using the selected LLM"""
        
        prompt = self.create_prompt(query, context_chunks)
        
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt)
            elif self.provider == "gemini":
                return self._generate_gemini(prompt)
            elif self.provider == "deepseek":
                return self._generate_deepseek(prompt)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating the response. Please try again."
        
    def generate_raw(self, prompt: str) -> str:
        """
        Generate a response using this model given a full prompt.
        This bypasses the RAG prompt construction and is used by the LLM council.
        """
        try:
            # Reuse whichever client was set up in __init__
            if "GPT" in self.model_name:
                return self._generate_openai(prompt)
            elif "Gemini" in self.model_name:
                return self._generate_gemini(prompt)
            elif "DeepSeek" in self.model_name:
                return self._generate_deepseek(prompt)
            else:
                raise ValueError(f"Unsupported model for raw generation: {self.model_name}")
        except Exception as e:
            st.error(f"LLM generation error ({self.model_name}): {str(e)}")
            raise
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model_key,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on document context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate response using Google Gemini"""
        response = self.client.generate_content(prompt)
        return response.text
    
    def _generate_deepseek(self, prompt: str) -> str:
        """Generate response using DeepSeek"""
        response = self.client.chat.completions.create(
            model=self.model_key,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on document context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content