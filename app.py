import streamlit as st
from typing import List
from config import Config
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.llm_handler import LLMHandler
from utils.council import CouncilConfig, CouncilMember, CouncilOrchestrator
from utils.job_search import JobSearchAPI
from utils.job_matcher import JobMatcher
from utils.snowflake_agent import SnowflakeAgent
import uuid
from datetime import datetime
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="RAG Chat Application",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = None
if 'current_namespace' not in st.session_state:
    st.session_state.current_namespace = None
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'job_search_results' not in st.session_state:
    st.session_state.job_search_results = None
if 'selected_job_details' not in st.session_state:
    st.session_state.selected_job_details = None
if 'snowflake_agent' not in st.session_state:
    st.session_state.snowflake_agent = None
if 'h1b_query_result' not in st.session_state:
    st.session_state.h1b_query_result = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 3  # Default to H1-B tab (index 3)
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0


def initialize_components(namespace: str = None):
    """Initialize PDF processor and vector store with namespace"""
    try:
        Config.validate()
        st.session_state.pdf_processor = PDFProcessor()
        
        # Generate a unique namespace for this session if not provided
        if namespace is None:
            namespace = f"session_{uuid.uuid4().hex[:8]}"
        
        st.session_state.current_namespace = namespace
        st.session_state.vector_store = VectorStore(namespace=namespace)
        
        return True
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return False


def process_uploaded_pdf(pdf_file, selected_model):
    """Process the uploaded PDF and store in Pinecone"""
    with st.spinner("Processing PDF..."):
        try:
            # Store document name
            st.session_state.document_name = pdf_file.name
            
            # Clear existing vectors in current namespace
            st.info(f"Clearing existing vectors from namespace '{st.session_state.current_namespace}'...")
            st.session_state.vector_store.clear_index()
            
            # Process PDF
            st.info("Extracting and chunking document...")
            chunks = st.session_state.pdf_processor.process_pdf(pdf_file)
            
            # Upload to Pinecone
            st.info("Uploading to vector database...")
            source_id = f"{pdf_file.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            num_chunks = st.session_state.vector_store.upsert_documents(chunks, source=source_id)
            
            # Update state
            st.session_state.document_uploaded = True
            st.session_state.chat_history = []
            
            st.success(f"✅ Document '{pdf_file.name}' processed successfully! ({num_chunks} chunks)")
            
            # Show index stats
            stats = st.session_state.vector_store.get_index_stats()
            st.info(f"Namespace '{st.session_state.current_namespace}' contains {stats.get('namespace_vector_count', 0)} vectors")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")


def handle_query(query: str, selected_model: str):
    """Handle user query and generate response"""
    
    # Retrieve relevant chunks
    with st.spinner("Searching document..."):
        relevant_chunks = st.session_state.vector_store.query(query, top_k=Config.TOP_K)
    
    if not relevant_chunks:
        response = "I couldn't find relevant information in the document to answer your question."
    else:
        # Generate response using selected LLM
        with st.spinner(f"Generating response using {selected_model}..."):
            llm_handler = LLMHandler(selected_model)
            response = llm_handler.generate_response(query, relevant_chunks)
    
    # Add to chat history
    st.session_state.chat_history.append({
        "query": query,
        "response": response,
        "model": selected_model,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    return response


def clear_chat_session():
    """Clear the current chat session and Pinecone namespace"""
    with st.spinner("Ending chat session..."):
        try:
            # Clear Pinecone namespace
            st.session_state.vector_store.clear_index()
            
            # Reset state
            st.session_state.document_uploaded = False
            st.session_state.chat_history = []
            st.session_state.document_name = None
            
            # Generate new namespace for next session
            st.session_state.current_namespace = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.vector_store = VectorStore(namespace=st.session_state.current_namespace)
            
            st.success("✅ Chat session ended. You can upload a new document.")
        except Exception as e:
            st.error(f"Error ending session: {str(e)}")


def render_rag_chat(selected_model: str):
    """Main RAG chat interface (document Q&A)."""
    if not st.session_state.document_uploaded:
        st.info("👈 Please upload a PDF document from the sidebar to start chatting!")
        
        # Show instructions
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload a PDF** document using the file uploader in the sidebar  
            2. **Click "Process Document"** to extract and index the content  
            3. **Choose an LLM model** from the dropdown (GPT-4, GPT-3.5, Gemini Pro, or DeepSeek)  
            4. **Ask questions** about your document in the chat interface  
            5. **End the session** when done to clear the document and start fresh  
            
            **Features:**  
            - Multiple questions on the same document  
            - Switch between different AI models  
            - View retrieval sources and confidence scores  
            - Isolated namespaces for privacy  
            """)
    else:
        # Display chat history
        st.header("💬 Chat History")
        
        if len(st.session_state.chat_history) == 0:
            st.info("No messages yet. Ask a question below!")
        
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"**You:** {chat['query']}")
                with col2:
                    st.caption(f"{chat['timestamp']}")
                
                st.markdown(f"**Assistant ({chat['model']}):** {chat['response']}")
                st.divider()
        
        # Query input
        st.header("❓ Ask a Question")
        
        with st.form(key="query_form", clear_on_submit=True):
            user_query = st.text_area(
                "Enter your question about the document:",
                height=100,
                placeholder="What is this document about?"
            )
            
            col1, col2 = st.columns([4, 1])
            with col1:
                submit_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
            with col2:
                st.caption(f"Using: {selected_model}")
            
            if submit_button and user_query:
                handle_query(user_query, selected_model)
                st.rerun()


def render_llm_council():
    """Resume vs JD LLM Council interface."""
    st.header("🧑‍💼 Resume vs JD – LLM Council")
    st.markdown(
        "Upload a resume and a job description to get a panel of LLMs "
        "to score the match, highlight strengths/gaps, and suggest concrete edits."
    )

    col_r, col_j = st.columns(2)
    with col_r:
        resume_file = st.file_uploader(
            "Upload your Resume (PDF)",
            type=["pdf"],
            key="council_resume"
        )
    with col_j:
        jd_file = st.file_uploader(
            "Upload Job Description (PDF or text)",
            type=["pdf", "txt"],
            key="council_jd"
        )

    # Optional: JD as plain text if they don't have a PDF
    jd_text_manual = st.text_area(
        "Or paste JD text here (used if no JD file is uploaded):",
        height=200
    )

    st.subheader("Council Configuration")

    # Use same model keys as for RAG
    model_options = list(Config.LLM_MODELS.keys())

    # Choose some sensible defaults: first three models (if present)
    default_members = model_options[:3] if len(model_options) >= 3 else model_options

    selected_member_models = st.multiselect(
        "Select council members (LLMs)",
        options=model_options,
        default=default_members,
        help="These models will independently score and critique your resume."
    )

    judge_model_name = st.selectbox(
        "Select judge model",
        options=model_options,
        index=0,
        help="This model will combine all opinions into one final verdict."
    )

    peer_review = st.checkbox(
        "Enable peer-review round between LLMs",
        value=True,
        help="Each LLM sees anonymised opinions from others and can revise its scores."
    )

    run_council = st.button("Run LLM Council on Resume + JD", type="primary")

    if run_council:
        if not resume_file:
            st.error("Please upload a Resume PDF.")
            return

        # Make sure we have a PDFProcessor instance
        if st.session_state.pdf_processor is None:
            st.session_state.pdf_processor = PDFProcessor()
        pdf_proc: PDFProcessor = st.session_state.pdf_processor

        # 1) Get resume text
        with st.spinner("Extracting text from resume..."):
            resume_text = pdf_proc.extract_text_from_pdf(resume_file)
            st.session_state.resume_text = resume_text  # Store for job search

        # 2) Get JD text – either from file or textarea
        jd_text = ""
        if jd_file is not None:
            if jd_file.type == "application/pdf":
                with st.spinner("Extracting text from JD PDF..."):
                    jd_text = pdf_proc.extract_text_from_pdf(jd_file)
            elif jd_file.type.startswith("text/"):
                jd_text = jd_file.read().decode("utf-8", errors="ignore")

        if not jd_text.strip():
            jd_text = jd_text_manual

        if not jd_text.strip():
            st.error("Please upload a JD file or paste JD text.")
            return

        # 3) Build council config
        if not selected_member_models:
            st.error("Please select at least one council member model.")
            return

        members = [
            CouncilMember(name=model_name, model_name=model_name)
            for model_name in selected_member_models
        ]
        config = CouncilConfig(
            members=members,
            judge_model_name=judge_model_name
        )
        council = CouncilOrchestrator(config)

        # 4) Run council
        result = council.run(
            resume_text=resume_text,
            jd_text=jd_text,
            peer_review=peer_review
        )

        member_opinions = result["member_opinions"]
        verdict = result["verdict"]

        # 5) Display verdict
        st.subheader("🎓 Final Verdict (Judge)")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Match Score (Resume vs JD)",
                f"{verdict.get('final_match_score', 0)} / 100"
            )
        with col_b:
            st.metric(
                "Overall Resume Quality",
                f"{verdict.get('final_resume_score', 0)} / 100"
            )

        st.markdown("**Key Strengths:**")
        for s in verdict.get("key_strengths", []):
            st.write(f"- {s}")

        st.markdown("**Key Gaps:**")
        for g in verdict.get("key_gaps", []):
            st.write(f"- {g}")

        st.markdown("**Top 5 Edits to Make Today:**")
        for i, tip in enumerate(verdict.get("top_5_edits_to_make_today", []), start=1):
            st.write(f"{i}. {tip}")

        st.markdown("**Explanation (for the student):**")
        st.write(verdict.get("explanation", ""))

        # 6) Show council opinions
        with st.expander("📝 See individual LLM opinions"):
            for op in member_opinions:
                st.markdown(f"### {op.get('model', 'Model')}")
                st.write("Match Score:", op.get("resume_jd_match_score", 0))
                st.write("Resume Score:", op.get("resume_overall_score", 0))

                st.markdown("**Strengths:**")
                for s in op.get("strengths", []):
                    st.write(f"- {s}")

                st.markdown("**Gaps:**")
                for g in op.get("gaps", []):
                    st.write(f"- {g}")

                st.markdown("**Specific Rewrite Suggestions:**")
                for sug in op.get("specific_rewrite_suggestions", []):
                    st.write(f"- {sug}")

                st.markdown("**Summary:**")
                st.write(op.get("short_summary", ""))
                st.markdown("---")


def render_job_search():
    """Job Search interface using resume and JSearch API"""
    st.header("🔍 Find Relevant Jobs")
    st.markdown("Upload your resume and we'll find jobs that match your profile using AI.")
    
    # Resume upload
    resume_upload = st.file_uploader(
        "Upload your Resume (PDF)",
        type=["pdf"],
        key="job_search_resume"
    )
    
    # If resume uploaded, extract text
    if resume_upload:
        if st.session_state.pdf_processor is None:
            st.session_state.pdf_processor = PDFProcessor()
        
        if st.button("📄 Extract Resume Info", type="secondary"):
            with st.spinner("Extracting text from resume..."):
                resume_text = st.session_state.pdf_processor.extract_text_from_pdf(resume_upload)
                st.session_state.resume_text = resume_text
                st.success("✅ Resume text extracted!")
    
    # Manual search parameters (optional override)
    st.subheader("Search Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Auto-populate if resume exists
        if st.session_state.resume_text and st.button("🤖 Auto-fill from Resume"):
            with st.spinner("Analyzing resume with AI..."):
                matcher = JobMatcher()
                params = matcher.extract_job_search_params(st.session_state.resume_text)
                
                st.session_state.job_titles = ", ".join(params.get("job_titles", []))
                st.session_state.locations = ", ".join(params.get("locations", []))
                st.session_state.key_skills = ", ".join(params.get("key_skills", []))
                st.success("✅ Parameters extracted!")
        
        job_query = st.text_input(
            "Job Title / Keywords",
            value=st.session_state.get("job_titles", ""),
            placeholder="e.g., Data Scientist, ML Engineer"
        )
        
        location = st.text_input(
            "Location",
            value=st.session_state.get("locations", "Remote"),
            placeholder="e.g., Boston, MA or Remote"
        )
        
        employment_type = st.multiselect(
            "Employment Type",
            options=["FULLTIME", "PARTTIME", "CONTRACTOR", "INTERN"],
            default=["FULLTIME"]
        )
    
    with col2:
        remote_only = st.checkbox("Remote jobs only", value=False)
        
        date_posted = st.selectbox(
            "Date Posted",
            options=["all", "today", "3days", "week", "month"],
            index=3
        )
        
        num_results = st.slider(
            "Number of results",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        calculate_match = st.checkbox(
            "Calculate AI match scores (slower)",
            value=True,
            help="Uses DeepSeek to score each job against your resume"
        )
    
    # Search button
    if st.button("🔍 Search Jobs", type="primary", use_container_width=True):
        if not job_query:
            st.error("Please enter a job title or keywords")
            return
        
        # Search for jobs
        with st.spinner("Searching for jobs..."):
            api = JobSearchAPI()
            results = api.search_jobs(
                query=job_query,
                location=location if location != "Remote" else None,
                num_pages=(num_results // 10) + 1,
                employment_types=employment_type if employment_type else None,
                remote_jobs_only=remote_only,
                date_posted=date_posted
            )
        
        if "error" in results:
            st.error(f"Error: {results['error']}")
            return
        
        jobs = results.get("data", [])
        
        if not jobs:
            st.warning("No jobs found. Try adjusting your search parameters.")
            return
        
        st.success(f"✅ Found {len(jobs)} jobs!")
        
        # Calculate match scores if requested and resume exists
        if calculate_match and st.session_state.resume_text:
            with st.spinner("Calculating match scores with AI..."):
                matcher = JobMatcher()
                jobs = matcher.rank_jobs_by_fit(
                    st.session_state.resume_text,
                    jobs,
                    max_jobs=min(num_results, 20)
                )
        
        st.session_state.job_search_results = jobs
    
    # Display results
    if st.session_state.job_search_results:
        st.divider()
        st.subheader(f"📋 {len(st.session_state.job_search_results)} Jobs Found")
        
        # Sorting options
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=["Match Score", "Date Posted", "Salary (High to Low)"],
                index=0 if calculate_match else 1
            )
        
        # Sort jobs
        jobs = st.session_state.job_search_results.copy()
        if sort_by == "Match Score":
            jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        elif sort_by == "Date Posted":
            jobs.sort(key=lambda x: x.get("job_posted_at_timestamp", 0), reverse=True)
        elif sort_by == "Salary (High to Low)":
            jobs.sort(key=lambda x: x.get("job_max_salary", 0), reverse=True)
        
        # Display job cards
        api = JobSearchAPI()
        
        for i, job in enumerate(jobs):
            render_job_card(job, api, i)


def render_job_card(job: dict, api: JobSearchAPI, index: int):
    """Render a single job card"""
    
    with st.container():
        # Border styling
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            # Company logo
            logo_url = job.get("employer_logo")
            if logo_url:
                st.image(logo_url, width=80)
            else:
                st.write("🏢")
        
        with col2:
            # Job title and company
            st.markdown(f"### {job.get('job_title', 'Unknown Title')}")
            st.markdown(f"**{job.get('employer_name', 'Unknown Company')}**")
            
            # Location and type badges
            location = f"{job.get('job_city', '')}, {job.get('job_state', '')}"
            if not location.strip(", "):
                location = job.get('job_country', 'Location N/A')
            
            badges = []
            badges.append(f"📍 {location}")
            
            if job.get('job_is_remote'):
                badges.append("🏠 Remote")
            
            employment_type = job.get('job_employment_type', '').replace('FULLTIME', 'Full-time')
            employment_type = employment_type.replace('PARTTIME', 'Part-time')
            if employment_type:
                badges.append(f"⏱️ {employment_type}")
            
            st.markdown(" | ".join(badges))
        
        with col3:
            # Match score (if available)
            match_score = job.get("match_score")
            if match_score is not None:
                st.metric("🎯 Match", f"{match_score}/100")
            
            # Salary
            salary = api.format_salary(job)
            st.markdown(f"**💰 {salary}**")
            
            # Posted date
            posted = api.format_posted_date(job.get('job_posted_at_datetime_utc', ''))
            st.caption(f"📅 {posted}")
        
        # Skills (if available)
        skills = job.get('job_required_skills', [])
        if skills:
            st.markdown("**Skills:** " + " • ".join([f"`{s}`" for s in skills[:6]]))
        
        # Description preview
        description = job.get('job_description', '')
        if description:
            preview = description[:200] + "..." if len(description) > 200 else description
            with st.expander("📋 Job Description Preview"):
                st.write(preview)
        
        # Action buttons (2 buttons now instead of 3)
        col_a, col_b = st.columns(2)
        
        with col_a:
            apply_link = job.get('job_apply_link')
            if apply_link:
                st.link_button("Apply Now →", apply_link, use_container_width=True, type="primary")
        
        with col_b:
            if st.session_state.resume_text:
                if st.button("Compare w/ Resume", key=f"compare_{index}", use_container_width=True):
                    render_job_comparison_modal(job, index)


def render_job_comparison_modal(job: dict, index: int):
    """Render detailed comparison between resume and job in a centered, full-width scrollable container"""
    
    # Create unique key for this comparison
    comparison_key = f"comparison_{index}"
    
    # Check if comparison already exists in session state
    if comparison_key not in st.session_state:
        with st.spinner("Analyzing match with DeepSeek..."):
            matcher = JobMatcher()
            comparison = matcher.generate_job_comparison(
                st.session_state.resume_text,
                job
            )
            st.session_state[comparison_key] = comparison
    else:
        comparison = st.session_state[comparison_key]
    
    # Full-width centered container
    st.markdown("---")
    
    # Title section - centered and full width
    st.markdown(f"""
    <div style="
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        margin: 20px auto;
        text-align: center;
    ">
        <h2 style="color: #fff; margin: 0;">
            🎯 Resume vs Job Analysis: {job.get('job_title', 'Unknown')}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Use expander for clean collapsible view - full width
    with st.expander("📊 View Detailed Analysis", expanded=True):
        # Add custom styling for better readability
        st.markdown("""
        <style>
        .strength-item { color: #4ade80; margin: 8px 0; }
        .gap-item { color: #fb923c; margin: 8px 0; }
        .recommendation-item { color: #60a5fa; margin: 8px 0; }
        div[data-testid="column"] {
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Two-column layout for strengths and gaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Your Strengths")
            strengths = comparison.get("strengths", [])
            if strengths:
                for strength in strengths:
                    st.markdown(f'<div class="strength-item">• {strength}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific strengths identified")
        
        with col2:
            st.markdown("### ⚠️ Gaps to Address")
            gaps = comparison.get("gaps", [])
            if gaps:
                for gap in gaps:
                    st.markdown(f'<div class="gap-item">• {gap}</div>', unsafe_allow_html=True)
            else:
                st.success("No major gaps identified!")
        
        st.markdown("---")
        
        # Recommendations - full width
        st.markdown("### 💡 Recommendations")
        recommendations = comparison.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-item">• {rec}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific recommendations")
        
        st.markdown("---")
        
        # Overall fit summary - full width
        overall_fit = comparison.get('overall_fit', 'N/A')
        st.info(f"**Overall Fit:** {overall_fit}")
        
        # Close button - centered
        col_empty1, col_btn, col_empty2 = st.columns([2, 1, 2])
        with col_btn:
            if st.button("Close Analysis", key=f"close_{index}", use_container_width=True):
                # Remove from session state to allow re-analysis
                if comparison_key in st.session_state:
                    del st.session_state[comparison_key]
                st.rerun()
    
    st.markdown("---")


def render_h1b_sponsorship():
    """H1-B Sponsorship Analysis using Snowflake Agent"""
    st.header("🇺🇸 H1-B Sponsorship Analysis")
    st.markdown("Explore H1-B visa sponsorship data with AI-powered insights and visualizations.")
    
    # Connection status and retry button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.session_state.snowflake_agent = None
            st.session_state.h1b_query_result = None
            st.rerun()
    with col3:
        if st.button("🔍 Check IP", use_container_width=True, help="Check your current IP address for whitelisting"):
            import requests
            try:
                ip = requests.get('https://api.ipify.org', timeout=5).text
                st.info(f"**Your Current IP:** `{ip}`\n\n**Verify in Snowflake:**\n```sql\nSHOW USERS LIKE 'Vemana30';\nSELECT \"name\", \"network_policy\" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));\n```")
            except:
                st.warning("Could not fetch IP address")
    
    # Initialize Snowflake agent
    if st.session_state.snowflake_agent is None:
        try:
            with st.spinner("Connecting to Snowflake..."):
                st.session_state.snowflake_agent = SnowflakeAgent()
                st.success("✅ Connected to Snowflake!")
        except ValueError as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            with st.expander("📋 Configuration Checklist"):
                st.markdown("""
                **Required Environment Variables in .env file:**
                - `SNOWFLAKE_ACCOUNT` - Your Snowflake account identifier (e.g., `xy12345` or `xy12345.us-east-1`)
                - `SNOWFLAKE_USER` - Your Snowflake username
                - `SNOWFLAKE_PASSWORD` - Your Snowflake password
                - `SNOWFLAKE_WAREHOUSE` - Your warehouse name
                - `SNOWFLAKE_DATABASE` - Your database name
                - `SNOWFLAKE_SCHEMA` - Your schema name
                - `SNOWFLAKE_ROLE` - (Optional) Your role, defaults to PUBLIC
                - `SNOWFLAKE_TABLE` - (Optional) Table name, defaults to H1B_DATA
                """)
            return
        except Exception as e:
            error_str = str(e)
            st.error(f"❌ Connection Failed: {error_str}")
            
            # Add a note about waiting
            st.info("⏳ **If you just whitelisted your IP, network policies can take 2-5 minutes to propagate. Please wait and try again.**")
            
            with st.expander("🔧 Troubleshooting Guide"):
                st.markdown("""
                **Error 250001 - IP Whitelisting Required (Most Common Fix)**
                
                This error means your IP address is not whitelisted in Snowflake. Follow these steps:
                
                **Step 1: Get Your IP Address**
                - Click the "🔍 Check IP" button above to see your current IP
                - Or visit: https://api.ipify.org
                
                **Step 2: Log into Snowflake Web UI**
                - Go to: https://di87692.snowflakecomputing.com
                - Log in as ACCOUNTADMIN (or ask your admin)
                
                **Step 3: Run These SQL Commands**
                ```sql
                -- Create network policy (allows all IPs for testing)
                CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
                  ALLOWED_IP_LIST = ('0.0.0.0/0')
                  BLOCKED_IP_LIST = ();
                
                -- Apply to your user
                ALTER USER Vemana30 SET NETWORK_POLICY = ALLOW_CURRENT_IP;
                ```
                
                **Step 4: Restart and Retry**
                - After running the SQL, click "🔄 Retry Connection" above
                - The connection should work now!
                
                **Alternative: Allow Only Your IP (More Secure)**
                ```sql
                CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
                  ALLOWED_IP_LIST = ('YOUR_IP_HERE/32')  -- Replace with your IP
                  BLOCKED_IP_LIST = ();
                
                ALTER USER Vemana30 SET NETWORK_POLICY = ALLOW_CURRENT_IP;
                ```
                
                **Other Possible Issues:**
                - Corporate firewall blocking Snowflake
                - VPN required for your Snowflake account
                - Private Link configuration needed
                - Account format: Try `SNOWFLAKE_ACCOUNT=di87692` in .env
                
                **SQL File Available:** See `whitelist_ip.sql` in the project root for complete SQL commands.
                """)
            return
    
    agent = st.session_state.snowflake_agent
    
    # Ensure we stay on H1-B tab when interacting with this section
    st.session_state.active_tab = 3
    
    # Question selection dropdown
    st.subheader("📊 Select Analysis Question")
    question_options = list(agent.QUESTIONS.keys())
    
    # Initialize selected question in session state if not exists
    if 'selected_h1b_question' not in st.session_state:
        st.session_state.selected_h1b_question = question_options[0] if question_options else None
    
    selected_question = st.selectbox(
        "Choose a question to analyze:",
        options=question_options,
        index=question_options.index(st.session_state.selected_h1b_question) if st.session_state.selected_h1b_question in question_options else 0,
        help="Select a question to query H1-B data and generate insights",
        key="h1b_question_selectbox"
    )
    
    # Update session state
    st.session_state.selected_h1b_question = selected_question
    # Keep tab active
    st.session_state.active_tab = 3
    
    # Show question description
    if selected_question:
        question_desc = agent.QUESTIONS[selected_question]["description"]
        st.info(f"📝 **Description:** {question_desc}")
    
    # Execute button
    if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Executing query and generating insights..."):
            try:
                result = agent.process_question(selected_question)
                st.session_state.h1b_query_result = result
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                return
    
    # Display results if available
    if st.session_state.h1b_query_result:
        result = st.session_state.h1b_query_result
        
        # AI Summary Section
        st.divider()
        st.subheader("🤖 AI-Generated Summary")
        st.markdown(result["summary"])
        
        # Data Table Section
        st.divider()
        st.subheader("📋 Data Results")
        st.dataframe(result["data"], use_container_width=True, height=400)
        
        # Download button for data
        csv = result["data"].to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"h1b_{selected_question.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
        
        # Visualizations Section
        st.divider()
        st.subheader("📈 Visualizations")
        
        if result["visualizations"]:
            # Display visualizations in a grid
            num_viz = len(result["visualizations"])
            cols_per_row = 2
            
            for i in range(0, num_viz, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < num_viz:
                        fig, title = result["visualizations"][i + j]
                        with col:
                            st.markdown(f"**{title}**")
                            st.pyplot(fig)
                            plt.close(fig)  # Close to free memory
        else:
            st.warning("No visualizations could be generated for this data.")
        
        # Query Information (Expandable)
        with st.expander("🔍 View SQL Query"):
            st.code(result["query"], language="sql")
        
        # Statistics Section
        st.divider()
        st.subheader("📊 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        df = result["data"]
        question = result["question"]
        question_lower = question.lower()
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            # Calculate appropriate metric based on question type
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_cols_filtered = [col for col in numeric_cols if col.upper() not in ['FISCAL_YEAR', 'YEAR', 'INDEX']]
            
            if len(numeric_cols_filtered) > 0:
                metric_value = None
                metric_label = None
                
                # For fiscal year queries
                if "fiscal year" in question_lower:
                    total_col = [col for col in numeric_cols_filtered if "TOTAL_APPROVALS" in col.upper()]
                    if total_col:
                        metric_value = df[total_col[0]].sum()
                        metric_label = "Total Approvals"
                
                # For company/industry/city queries
                elif any(x in question_lower for x in ["companies", "industries", "cities"]):
                    total_col = [col for col in numeric_cols_filtered if "TOTAL_APPROVALS" in col.upper()]
                    if total_col:
                        metric_value = df[total_col[0]].sum()
                        metric_label = "Total Approvals"
                    elif len(numeric_cols_filtered) > 0:
                        metric_value = df[numeric_cols_filtered[0]].sum()
                        metric_label = numeric_cols_filtered[0].replace("_", " ").title()
                
                # For state queries
                elif "state" in question_lower:
                    total_col = [col for col in numeric_cols_filtered if "TOTAL_APPROVALS" in col.upper()]
                    if total_col:
                        metric_value = df[total_col[0]].sum()
                        metric_label = "Total Approvals"
                    else:
                        rate_col = [col for col in numeric_cols_filtered if "RATE" in col.upper()]
                        if rate_col:
                            metric_value = df[rate_col[0]].mean()
                            metric_label = f"Avg {rate_col[0].replace('_', ' ').title()}"
                
                # For approval/denial trends
                elif "approval" in question_lower and "denial" in question_lower:
                    total_col = [col for col in numeric_cols_filtered if "TOTAL_APPROVED" in col.upper() or "TOTAL_APPROVALS" in col.upper()]
                    if total_col:
                        metric_value = df[total_col[0]].sum()
                        metric_label = "Total Approved"
                
                # For growth queries
                elif "growth" in question_lower:
                    total_col = [col for col in numeric_cols_filtered if "TOTAL_APPROVALS" in col.upper()]
                    if total_col:
                        metric_value = df[total_col[0]].sum()
                        metric_label = "Total Approvals"
                
                # Default: use first numeric column
                if metric_value is None and len(numeric_cols_filtered) > 0:
                    metric_value = df[numeric_cols_filtered[0]].sum()
                    metric_label = numeric_cols_filtered[0].replace("_", " ").title()
                
                if metric_value is not None:
                    if isinstance(metric_value, (int, float)) and metric_value >= 1000:
                        st.metric(metric_label, f"{metric_value:,.0f}")
                    elif isinstance(metric_value, (int, float)):
                        st.metric(metric_label, f"{metric_value:.2f}")
                    else:
                        st.metric(metric_label, str(metric_value))
                else:
                    st.metric("N/A", "-")
            else:
                st.metric("N/A", "-")
        with col4:
            st.metric("Visualizations", len(result["visualizations"]))


def main():
    # Title and description
    st.title("AtlasAI: An Intelligent Career and Immigration Insight Platform")
    st.markdown("Upload a PDF and chat with it, evaluate a resume vs JD, find relevant jobs, or analyze H1-B sponsorship data.")

    # Initialize components
    if st.session_state.pdf_processor is None or st.session_state.vector_store is None:
        if not initialize_components():
            st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Select LLM Model for RAG Chat",
            options=list(Config.LLM_MODELS.keys()),
            help="Choose the language model for generating responses in the document chat"
        )
        
        st.divider()
        
        # File upload
        st.header("📄 Document Upload (for RAG Chat)")
        
        if st.session_state.document_uploaded:
            st.success(f"📄 Current: {st.session_state.document_name}")
        
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to chat with",
            key="rag_pdf"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary", use_container_width=True):
                process_uploaded_pdf(uploaded_file, selected_model)
        
        st.divider()
        
        # Session controls
        st.header("🔧 Session Controls")
        
        if st.session_state.document_uploaded:
            if st.button("End Chat & Clear Document", type="secondary", use_container_width=True):
                clear_chat_session()
                st.rerun()
        else:
            st.info("No document loaded")
        
        # Show current namespace
        if st.session_state.current_namespace:
            st.caption(f"Namespace: `{st.session_state.current_namespace}`")
        
        st.divider()
        
        # Index stats
        st.header("📊 Index Statistics")
        if st.session_state.vector_store:
            if st.button("Refresh Stats", use_container_width=True):
                st.rerun()
            
            stats = st.session_state.vector_store.get_index_stats()
            st.metric("Total Vectors", stats.get('total_vector_count', 0))
            st.metric("Current Namespace", stats.get('namespace_vector_count', 0))
            st.metric("Dimension", stats.get('dimension', 'N/A'))
            st.caption(f"Model: {Config.EMBEDDING_MODEL}")

    # Main layout: four tabs
    # Check if we should stay on H1-B tab (if user was interacting with H1-B section)
    if 'selected_h1b_question' in st.session_state or 'h1b_query_result' in st.session_state:
        st.session_state.active_tab = 3
    
    # Initialize active_tab if not set
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    tab_names = [
        "📄 Document RAG Chat",
        "🧑‍💼 Resume–JD LLM Council",
        "🔍 Job Search",
        "🇺🇸 H1-B Sponsorship"
    ]
    
    # Use radio buttons styled as tabs for better control
    # This allows us to preserve tab selection across reruns
    selected_tab_index = st.radio(
        "Navigation",
        options=list(range(len(tab_names))),
        format_func=lambda x: tab_names[x],
        horizontal=True,
        label_visibility="collapsed",
        index=st.session_state.active_tab,
        key="main_tab_selector"
    )
    
    # Update session state
    st.session_state.active_tab = selected_tab_index
    
    # Render content based on selected tab
    if selected_tab_index == 0:
        render_rag_chat(selected_model)
    elif selected_tab_index == 1:
        render_llm_council()
    elif selected_tab_index == 2:
        render_job_search()
    elif selected_tab_index == 3:
        render_h1b_sponsorship()


if __name__ == "__main__":
    main()