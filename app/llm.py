# app/llm.py
import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = """You are a UK HR Policy & Immigration Expert and Forecasting Assistant. 

You help HR professionals, immigration lawyers, and policy analysts understand:
- UK immigration rule changes and their impact on workforce planning
- Skilled Worker visa requirements and sponsor licence obligations  
- Job vacancy trends and labor market forecasts
- Policy event impacts on recruitment and retention

Provide accurate, actionable insights based on official UK government sources.
Always cite specific policies, dates, and sources when available.
Be concise but comprehensive in your explanations."""

def chat(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> str:
    """
    Send messages to OpenAI and return the response.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: OpenAI model to use
        
    Returns:
        The assistant's response as a string
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}. Please check your OpenAI API key."
