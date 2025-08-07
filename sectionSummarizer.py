import os
import json
import streamlit as st
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from overall_summary import extract_sections_with_titles

# Load environment variables
load_dotenv()
# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize the LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=GOOGLE_API_KEY,
)

# Define the summarization agent
summarizer_agent = Agent(
    role="Summarizer Agent",
    goal="Summarize research paper sections in bullet points, tailored to section type.",
    llm=llm,
    backstory=(
        "You are an academic summarization expert who understands the structure and purpose of various "
        "research paper sections like Abstract, Introduction, Methodology, Results, and Conclusion."
    ),
    max_iterations=3,
)

# Define function to create a Task
def summarize_section(text, title):
    return Task(
        description=f"""
You are an expert research paper analyst and summarizer.

**Section Title**: {title}

**Content**:
{text}

Extract and return only a bullet-point summary covering all key ideas and technical points. 
If any formula or equation is present, include it in the output.

Style Guide:
- For "Abstract": summarize objective, methods, results.
- For "Introduction": highlight the problem, motivation, and contributions.
- For "Methodology": describe approach, models, and setup.
- For "Results" or "Experiments": explain key outcomes and implications.
- For "Conclusion": summarize findings and future work.

Respond ONLY with a bullet-point summary in **JSON format**. No extra text.
""",
        agent=summarizer_agent,
        expected_output="JSON bullet-point summary",
        input={"text": text}
    )
