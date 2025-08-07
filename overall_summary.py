import fitz 
import re
from io import BytesIO
import os
import streamlit as st
from langchain_groq import ChatGroq
import arxiv
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import ArxivAPIWrapper
arxiv=ArxivAPIWrapper()

load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
# api_key=st.secrets["GROQ_API_KEY"]


llm=ChatGroq(model="gemma2-9b-it",api_key=api_key)


text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=7000,
    chunk_overlap=200,
    separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
)


def generate_metadata(chunks):
    e_info=[]
    chunk_prompt = f"""
    You are an expert research paper analyst.

    Given the following chunk of a research paper:
    {chunks[0]}

    Extract and return any relevant information for the following structured JSON fields:

    {{
    "title": "<Paper Title>",
    "model": "<Primary model/architecture proposed or used>",
    "dataset": "<Primary dataset used in the experiments>",
    "metrics": {{
        "<metric_name>": <numeric_value>
    }},
    "summary": {{
        "Objective": "<2-line goal of the paper>",
        "Key Methods": "<Key method or technique used>",
        "Dataset used": "<Dataset(s) used in the experiment>",
        "Evaluation Metrics": "<Metrics used for evaluation>",
        "Key Findings": "<2-3 line summary of findings/results>",
        "Tasks":"<where we can use this real life example>",
        "Advantages":"<Advantages of the proposed method>"
    }}
    }}
    If any field is not found in THIS CHUNK, 
    then based on the research paper title add the relevant information and return it.

    Only return the JSON format,whithout json quotation and no explanation.
    """
    response = llm.invoke(chunk_prompt)
    e_info.append(response.content)
    e_info=json.loads(e_info[0])
    return e_info




def extract_sections_with_titles(uploaded_file):
    if uploaded_file is None:
        return None, None, None

    # Read file content and wrap it in BytesIO
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty or corrupted.")
    
    pdf_stream = BytesIO(file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    # Extract text from the first 2 pages for arXiv ID
    first_text = "\n".join([doc[i].get_text() for i in range(min(2, len(doc)))])
    
    # Extract arXiv ID
    ids = []
    arxiv_match = re.search(r"arXiv[: ]\d{4}\.\d{4,5}(v\d+)?", first_text, re.IGNORECASE)
    if arxiv_match:
        ids.append(arxiv_match.group().strip())

    # Full text of the document
    full_text = "\n".join(page.get_text() for page in doc)

    section_titles = [
        "Abstract", "Introduction", "Related Work", "Background", "Methodology",
        "Experiments", "Results", "Discussion", "Conclusion", "Conclusions and future work"
    ]

    sections = {}
    for i, title in enumerate(section_titles):
        pattern = rf"(?i)\b{title}\b\s*[\n:]"
        matches = list(re.finditer(pattern, full_text))

        if matches:
            start = matches[0].end()
            end = len(full_text)
            for j in range(i + 1, len(section_titles)):
                next_pattern = rf"(?i)\b{section_titles[j]}\b\s*[\n:]"
                next_match = re.search(next_pattern, full_text[start:])
                if next_match:
                    end = start + next_match.start()
                    break
            sections[title] = full_text[start:end].strip()
        else:
            sections[title] = ""

    return full_text, ids, sections



