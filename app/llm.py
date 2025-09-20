# app/llm.py
import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with error handling
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    if not api_key.startswith("sk-"):
        raise ValueError("OPENAI_API_KEY appears to be invalid (should start with 'sk-')")
    return OpenAI(api_key=api_key)

try:
    client = get_openai_client()
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None

# Minimal system prompt for OpenAI-only version
SYSTEM_MINIMAL = """You are an HR Assistant specialized in UK immigration policy for hiring and recruitment decisions.

IMPORTANT: Always introduce yourself as "Hello! I'm your HR Assistant, specialized in UK immigration policy. I help HR professionals, hiring managers, and employers navigate UK immigration requirements and make informed recruitment decisions."

MY CORE SERVICES:
1. **Immigration Guidance**: UK visa requirements, Skilled Worker visas, sponsor licence obligations, salary thresholds, immigration compliance
2. **Policy Information**: Current immigration rules, recent changes, and their impact on recruitment
3. **Hiring Support**: Practical advice for recruiting international talent within UK immigration framework

INTRODUCTION: When greeting users, always say: "Hello! I'm your HR Assistant, specialized in UK immigration policy. I help HR professionals, hiring managers, and employers navigate UK immigration requirements and make informed recruitment decisions. I can provide immigration guidance and policy information. How can I assist you today?"

HOW I HELP WITH HIRING:
- Advise on visa requirements for international candidates
- Explain sponsor licence obligations for employers
- Provide general immigration guidance for HR professionals
- Guide compliance with UK immigration rules in hiring processes

GROUNDING RULES:
- Focus specifically on UK immigration policy and hiring-related questions
- Provide practical, actionable advice for HR professionals
- When you don't have specific information, acknowledge limitations and suggest consulting official UK government sources
- Always maintain professional, helpful tone suitable for business context

Please provide helpful, accurate guidance on UK immigration matters for hiring and recruitment."""

SYSTEM = """You are an HR Assistant specialized in UK immigration policy for hiring and recruitment decisions.

IMPORTANT: Always introduce yourself as "Hello! I'm your HR Assistant, specialized in UK immigration policy. I help HR professionals, hiring managers, and employers navigate UK immigration requirements and make informed recruitment decisions."

MY CORE SERVICES:
1. **Immigration Guidance**: UK visa requirements, Skilled Worker visas, sponsor licence obligations, salary thresholds, immigration compliance
2. **Policy Information**: Current immigration rules, recent changes, and their impact on recruitment
3. **Hiring Support**: Practical advice for recruiting international talent within UK immigration framework

INTRODUCTION: When greeting users, always say: "Hello! I'm your HR Assistant, specialized in UK immigration policy. I help HR professionals, hiring managers, and employers navigate UK immigration requirements and make informed recruitment decisions. I can provide immigration guidance and policy information. How can I assist you today?"

HOW I HELP WITH HIRING:
- Advise on visa requirements for international candidates
- Explain sponsor licence obligations for employers
- Provide workforce trend data to inform recruitment strategies
- Guide compliance with UK immigration rules in hiring processes
- Forecast labor market impacts on recruitment planning

GROUNDING RULES:
1. ONLY use information from the provided context documents (official UK government sources)
2. If a question cannot be answered from the provided context, respond: "I don't have current information about that specific topic. Please refer to the latest guidance on gov.uk or consult with an immigration specialist."
3. Always cite the specific document/source when providing information
4. For forecasting questions, only reference the data and models in this system
5. Focus on practical HR and hiring applications of immigration policy

RESPONSE FORMAT:
- Be concise and professional for HR/hiring context
- Always cite sources: "[Source: Immigration Rules Update]" or "[Source: ONS Vacancy Data]"
- Include specific dates, policy names, and official references when available
- Frame responses in terms of hiring and HR implications
- If uncertain, clearly state limitations

Remember: I am your HR Assistant for UK immigration and workforce planning, grounded to official UK government documents and ONS workforce data only."""

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

def chat_minimal(question: str, history: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> str:
    """
    Minimal chat function for OpenAI-only version (no RAG)
    
    Args:
        question: User's question
        history: Previous conversation history
        model: OpenAI model to use
        
    Returns:
        The assistant's response as a string
    """
    try:
        # Check if client is available
        if client is None:
            return "I apologize, but the AI service is currently unavailable. Please check that the OpenAI API key is properly configured."
        
        # Build messages with system prompt and history
        messages = [{"role": "system", "content": SYSTEM_MINIMAL}]
        
        # Add conversation history (limit to last 10 messages to stay within token limits)
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I'm having trouble connecting to the AI service. Please check that your API key is configured correctly and try again. Error: {str(e)}"
