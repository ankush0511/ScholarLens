
import fitz  # PyMuPDF
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Define PDF text extractor
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Define LLM
llm = ChatGroq(model='openai/gpt-oss-120b', api_key=GROQ_API_KEY)

# Define prompt template using ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Some parts of the research papers may be truncated due to length constraints.  
Please use your understanding and reasoning abilities to compare and complete any missing sections if possible.

Your task is to compare the two research papers below across the following dimensions:
- **Title**  
- **Problem Statement**  
- **Methodology**  
- **Datasets Used**  
- **Results**   
- **Model Performance**  
- **Model Accuracy**

Return the output as a **well-formatted Markdown table** for easy comparison.

### Paper 1:
{paper1}

### Paper 2:
{paper2}
""")

# Create chain using LCEL (LangChain Expression Language)
chain = prompt | llm | StrOutputParser()




# Usage example:
# result = chain.invoke({"paper1": paper1_text, "paper2": paper2_text})
# import fitz  # PyMuPDF
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import os
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Define PDF text extractor
# def extract_text_from_pdf(file):
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     return "\n".join([page.get_text() for page in doc])

# # Define LLM chain
# llm = ChatGroq(model='gemma2-9b-it', api_key=GROQ_API_KEY)

# prompt = PromptTemplate(
#     input_variables=["paper1", "paper2"],
#     template="""
# Some parts of the research papers may be truncated due to length constraints.  
# Please use your understanding and reasoning abilities to compare and complete any missing sections if possible.

# Your task is to compare the two research papers below across the following dimensions:

# - **Title**  
# - **Problem Statement**  
# - **Methodology**  
# - **Datasets Used**  
# - **Results**   
# - **Model Performance**  
# - **Model Accuracy**

# Return the output as a **well-formatted Markdown table** for easy comparison.

# ### Paper 1:
# {paper1}

# ### Paper 2:
# {paper2}
# """
# )

# chain = LLMChain(llm=llm, prompt=prompt)
