import json
import logging
from crewai import Agent, Task, Crew, Process
import arxiv
from typing import List, Dict
from crewai.tools import tool
import os
import streamlit as st
from crewai import LLM
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from google import genai
GOOGLE_API_KEY=st.secrets['GOOGLE_API_KEY']
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

llm=LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)


# Custom arXiv search tool
@tool("ArxivSearchTool")
def arxiv_search(query: str, max_results:int) -> List[Dict]:
    """
    Search arXiv for papers based on a query and return up to max_results papers.
    """
    if max_results <= 0:
        logger.warning("max_results must be positive. Setting to 5.")
        max_results = 5
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        papers: List[Dict] = []
        for result in client.results(search):
            try:
                papers.append({
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "summary": result.summary[:200],
                    "pdf_url": result.pdf_url,
                    "published": result.published.strftime("%Y-%m-%d"),
                })
            except Exception as e:
                logger.error(f"Error processing paper {result.title}: {e}")
                continue
        logger.info(f"Retrieved {len(papers)} papers for query: {query}")
        return papers
    except Exception as e:
        logger.error(f"Error during arXiv search: {e}")
        return []

# ArXiv Researcher Agent
arxiv_researcher_agent = Agent(
    role="Arxiv Researcher Agent",
    goal="Create ArXiv queries and retrieve candidate papers",
    llm=llm,
    tools=[arxiv_search],
    backstory=(
        "You are an expert in scientific literature retrieval. Your role is to identify the most relevant,latest and impactful papers from arXiv. "
        "You specialize in crafting optimal queries based on user-specified topics, overfetching relevant,latest research to ensure thorough coverage, "
        "and filtering to return only the most useful papers for downstream summarization."
    ),
    # verbose=True,
    max_iterations=3
)

# Summarizer Agent
summarizer_agent = Agent(
    role="Summarizer Agent",
    goal="Summarize the results into a structured JSON format",
    llm=llm,
    backstory=(
        "You are a seasoned academic researcher and writer. You are skilled in analyzing scientific papers and synthesizing their insights into clear, "
        "concise, and structured literature reviews. You excel at capturing key contributions and contextualizing them for a research audience."
    ),
    # verbose=True,
    max_iterations=3
)

def run_literature_review_latest(topic: str, num_papers: int):
    """Run the literature review crew for a given topic"""
    researcher_task = Task(
        description=f"""
            Given the topic "{topic}" and desired number of papers {num_papers}, generate an effective arXiv query.
            Use the arXiv search tool to retrieve around three times more papers than needed (e.g., {num_papers * 3}).
            Then, filter and select exactly {num_papers} highly relevant papers based on title, abstract, and relevance to the user query.
            Return these selected papers in a structured JSON format with fields like: title, link, authors, abstract.
        """,
        agent=arxiv_researcher_agent,
        expected_output=f"Choose exactly {num_papers} papers and pass them as concise JSON to the summarizer."
    )

    summarizer_task = Task(
        description=f"""
            Given a JSON list of papers about "{topic}", generate a summary in **valid JSON format only**.
            Your output must follow this exact structure and contain **ONLY** the JSON:
            {{
              "introduction": "Brief overview of the topic.",
              "papers details": [
                {{
                  "title": "Paper title",
                  "link": "https://...",
                  "authors": "Author A, Author B",
                  "problem": "What problem the paper addresses",
                  "contribution": "What the key contribution is",
                  "summary": "Concise summary of the paper."
                }}
              ]
            }}
            DO NOT include Markdown formatting, bullet points, code blocks, or any extra text before or after the JSON.
            Output must be a single valid JSON object.
        """,
        agent=summarizer_agent,
        expected_output="A valid JSON object with 'introduction' and 'papers details'.",
        # output_json=True # Removed as it's causing the ValidationError
    )

    crew = Crew(
        agents=[arxiv_researcher_agent, summarizer_agent],
        tasks=[researcher_task, summarizer_task],
        process=Process.sequential,
        # verbose=True
    )

    result = crew.kickoff()
    try:
        # Assuming the summarizer agent's output is the last task's raw output
        raw_outputs = result.tasks_output[-1].raw
        result_json = json.loads(raw_outputs)
        return result_json
    except (json.JSONDecodeError, IndexError) as e:
        json_str = result.raw.strip().replace('```json', '').replace('```', '')
        json_str=json.loads(json_str)
        return json_str
        