import patch_sqlite
import json
import tempfile
import os
import streamlit as st
from overall_summary import extract_sections_with_titles, generate_metadata, arxiv, text_splitter
from find_Research_Paper import run_literature_review
from sectionSummarizer import summarize_section, summarizer_agent, LLM, Crew, Process
from ComparePapers import extract_text_from_pdf, chain
from io import BytesIO
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.agent import Agent
from Agentic_Rag import embeddings, vector_db, PDFReader,Agent, Gemini

# Initialize session state
if 'pdf_knowledge_base' not in st.session_state:
    st.session_state.pdf_knowledge_base = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

import streamlit as st
from PIL import Image

# Set page title and favicon
st.set_page_config(
    page_title="ScholarLens",
    page_icon="🧠",
    layout="wide",
)

# Display logo in sidebar
with st.sidebar:
    logo = Image.open("assets/arxiv.jpeg")  # adjust path as needed
    st.image(logo, width=100)
    st.markdown("### **ScholarLens**")
    st.markdown("AI-powered Research Assistant")


# Sidebar navigation
options = st.sidebar.selectbox(
    "Select a page", ["ArXiv Smart Search", "AI Paper Companion","RAG Chatbot", "Compare Research Papers"]
)

# Page 1: ArXiv Literature Review
if options == 'ArXiv Smart Search':
    st.markdown("## 🔍 ArXiv Smart Search")
    st.markdown("Use this tool to search and summarize recent research papers from arXiv based on your topic.")

    with st.sidebar:
        st.header("📌 Configuration")
        topic = st.text_input("Enter research topic", placeholder="e.g. Diffusion Models in Vision")
        num_papers = st.slider("Number of papers to summarize", 1, 10, 4)
        run_button = st.button("🚀 Find Paper")

    if run_button:
        if not topic.strip():
            st.warning("Please enter a valid topic.")
        else:
            with st.spinner("🔎 Searching and summarizing papers..."):
                result = run_literature_review(topic, num_papers=num_papers)
                st.success("✅ Paper Search Complete!")

                st.write("## ✨ Introduction")
                st.write(result.get("introduction"))

                st.markdown("## 📄 Papers Summary")
                papers = result.get("papers details", [])
                if not papers:
                    st.warning("No paper details found.")
                else:
                    for i, paper in enumerate(papers, 1):
                        with st.expander(f"📘 {i}. {paper.get('title', 'Untitled')}", expanded=False):
                            st.markdown(f"**🔗 Title:** [{paper.get('title')}]({paper.get('link')})")
                            st.markdown(f"**👥 Authors:** {paper.get('authors', 'N/A')}")
                            st.markdown(f"**❓ Problem:** {paper.get('problem', 'N/A')}")
                            st.markdown(f"**🚀 Contribution:** {paper.get('contribution', 'N/A')}")
                            st.markdown(f"**📝 Summary:** {paper.get('summary', 'N/A')}")

                st.download_button(
                    label="📥 Download Results as JSON",
                    data=json.dumps(result, indent=2),
                    file_name="literature_review.json",
                    mime="application/json"
                )

# Page 2: PDF Assistant (Summary, Sectional Summary, RAG Chatbot)
elif options == 'AI Paper Companion':
    with st.sidebar:
        st.markdown("## 📎 Upload Paper")
        uploaded_file = st.file_uploader("Upload Paper", type=["pdf"])

    if uploaded_file is not None:
        st.success("PDF uploaded successfully!")
        text, index, sections = extract_sections_with_titles(uploaded_file)
        arxiv_id = index[0].replace("arXiv:", "").strip()

        tab1, tab2 = st.tabs(['⚡ Instant Glance', "📝Summary (Guru Mode)"])

        with tab1:
            st.markdown("## ⚡ Instant Glance")
            chunks = text_splitter.split_text(text)
            paper_data = generate_metadata(chunks)
            st.subheader(f"🧾 Title: {paper_data['title']}")
            st.markdown(f"**🧠 Model:** {paper_data['model']}")
            st.markdown(f"**📊 Dataset:** {paper_data['dataset']}")
            st.markdown("### 📈 Evaluation Metrics:")
            for metric, value in paper_data["metrics"].items():
                st.markdown(f"- **{metric}**: {value}")

            summary = paper_data["summary"]
            st.markdown("### 📝 Summary")
            for key, val in summary.items():
                st.markdown(f"**{key}:** {val}")

        with tab2:
            st.title('📚 Paper Summarizer (Guru Mode)')
            section_titles = [
                "Abstract", "Introduction", "Related Work", "Background", "Methodology",
                "Experiments", "Results", "Discussion", "Conclusion", "Conclusions and future work"
            ]
            available_sections = [title for title in section_titles if sections.get(title)]
            if not available_sections:
                st.warning("No recognized sections found in the uploaded PDF.")
            else:
                selected_section = st.selectbox("📌 Choose a section to summarize", available_sections)
                task = summarize_section(sections[selected_section], selected_section)
                crew = Crew(agents=[summarizer_agent], tasks=[task], process=Process.sequential)

                with st.spinner("🔍 Summarizing..."):
                    result = crew.kickoff()
                    raw_output = result.raw.strip().strip('`')
                    if raw_output.startswith("json"):
                        raw_output = raw_output[4:].strip()
                    try:
                        summary_data = json.loads(raw_output)
                        st.subheader("📝 Bullet-Point Summary:")
                        for value in summary_data.values():
                            st.write(f"- {value}")
                    except json.JSONDecodeError:
                        st.error("⚠️ Failed to parse JSON summary. Showing raw output:")
                        st.code(raw_output)

        
# elif options == 'RAG Chatbot':

#     with st.sidebar:
#         st.markdown("## 📎 Upload Paper")
#         uploaded_file = st.file_uploader("Upload Paper", type=["pdf"])

#     tmp_path = None

#     if uploaded_file is not None:
#         st.success("PDF uploaded successfully!")
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             tmp_path = tmp_file.name

#     if st.button("Load Knowledge Base"):
#         if not tmp_path:
#             st.warning("Please upload a PDF first.")
#         else:
#             try:
#                 pdf_knowledge_base = PDFKnowledgeBase(
#                     path=tmp_path,
#                     vector_db=vector_db,
#                     reader=PDFReader(chunk=True)
#                 )

#                 pdf_knowledge_base.load(recreate=True, upsert=True)

#                 agent = Agent(
#                     knowledge=pdf_knowledge_base,
#                     show_tool_calls=True,
#                     model=Gemini(id="gemini-2.0-flash-lite"),
#                     markdown=True,
#                     read_chat_history=True
#                 )

#                 st.session_state.pdf_knowledge_base = pdf_knowledge_base
#                 st.session_state.agent = agent
#                 st.success("✅ Knowledge base loaded!")

            
#                 if st.session_state.get('agent'):
#                     query = st.text_input("🤖 Ask Me Anything About This Paper")
#                     if query:
#                         response = st.session_state.agent.run(query)
#                         st.markdown("### Response")
#                         st.write(response.content)

#             except Exception as e:
#                 st.error(f"❌ Failed to load knowledge base: {e}")

#             finally:
#                 os.unlink(tmp_path)  # Always clean up if created


elif options == 'RAG Chatbot':
    with st.sidebar:
        st.markdown("## 📎 Upload Paper")
        uploaded_file = st.file_uploader("Upload Paper", type=["pdf"])

    if uploaded_file is not None:
        st.success("PDF uploaded successfully!")

        if 'pdf_temp_path' not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.pdf_temp_path = tmp_file.name

    if st.button("Load Knowledge Base"):
        if 'pdf_temp_path' not in st.session_state:
            st.warning("Please upload a PDF first.")
        else:
            try:
                pdf_knowledge_base = PDFKnowledgeBase(
                    path=st.session_state.pdf_temp_path,
                    vector_db=vector_db,
                    reader=PDFReader(chunk=True)
                )
                pdf_knowledge_base.load(recreate=False, upsert=True)

                agent = Agent(
                    knowledge=pdf_knowledge_base,
                    show_tool_calls=True,
                    model=Gemini(id="gemini-2.0-flash-lite"),
                    markdown=True,
                    read_chat_history=True
                )

                st.session_state.pdf_knowledge_base = pdf_knowledge_base
                st.session_state.agent = agent
                st.success("✅ Knowledge base loaded!")

            except Exception as e:
                st.error(f"❌ Failed to load knowledge base: {e}")

    if st.session_state.get('agent'):
        query = st.text_input("🤖 Ask Me Anything About This Paper")
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(query)
                st.markdown("### Response")
                st.write(response.content)
    
    if st.button("Clear Session"):
        st.session_state.clear()
        if 'pdf_temp_path' in st.session_state:
            os.unlink(st.session_state.pdf_temp_path)
        st.rerun()



# Page 3: Paper Comparison
else:
    uploaded_files = st.file_uploader("Upload 2 research papers (PDF)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("Please upload at least 2 PDFs.")
        elif len(uploaded_files) > 3:
            st.warning("Please upload no more than 3 PDFs.")
        else:
            with st.spinner("🔍 Extracting text from PDFs..."):
                pdf_texts = [extract_text_from_pdf(f) for f in uploaded_files]
            st.success("✅ Text extracted! Click below to compare.")

            if st.button("🧠 Compare Papers"):
                with st.spinner("🤖 Comparing research papers using LLM..."):
                    try:
                        response = chain.run({
                            "paper1": pdf_texts[0][:6000],
                            "paper2": pdf_texts[1][:6000]
                        })
                        st.markdown("### 📊 Comparison Table")
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    else:
        st.info("Awaiting uploads...")
