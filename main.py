import os
import json
import asyncio
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1") 
API_KEY = os.getenv("API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo") 

if not API_KEY:
    raise ValueError(
        "Please set API_KEY in your environment variables or .env file."
    )
    
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


# --- Model for LinkedIn Post Output ---
class LinkedInPost(BaseModel):
    topic: str
    language: str
    post: str = Field(description="Generated LinkedIn-style post, 2–4 engaging paragraphs")


# --- Resume Writer Agent ---
resume_writer_agent = Agent(
    name="Resume Writer",
    instructions="""
    You are a professional LinkedIn post writer.  
    Your job is to take a given topic and language, and then create a professional, 
    engaging LinkedIn-style post (2–4 short paragraphs) also provide a title for the post and use hashtags at the end of the post.  

    The post should:
    - Be in the requested language
    - Be structured and clear
    - Be professional but also friendly and relatable
    - Avoid bullet lists (use paragraph style)
    - Highlight insights, practical tips, or professional reflections
    """,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    output_type=LinkedInPost,
)


# --- Main Function ---
async def main():
    # Example inputs
    queries = [
        {"topic": "AI in Healthcare", "language": "English"},
        {"topic": "Remote Work Productivity", "language": "Japanese"},
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"TOPIC: {query['topic']} | LANGUAGE: {query['language']}")
        
        # Pass structured input
        result = await Runner.run(
            resume_writer_agent, 
            f"Write a LinkedIn-style post about '{query['topic']}' in {query['language']}."
        )
        
        # Extract and print
        linkedin_post = result.final_output
        print("\n📝 LINKEDIN POST:")
        print(linkedin_post.post)


if __name__ == "__main__":
    asyncio.run(main())
