# AtlasAI – Intelligent Career & Immigration Insight Platform

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An AI-powered career platform for resume optimization, intelligent job search, document chat, and H1-B visa analytics.**

</div>

🚀 **Live Demo:**
[https://job-board-ai-jwgzdkxcadeqqsqdct9wsn.streamlit.app/](https://job-board-ai-jwgzdkxcadeqqsqdct9wsn.streamlit.app/)

🚀 **Walkthrough Video:**
[https://youtu.be/hH0rN9RoZSk](https://youtu.be/hH0rN9RoZSk)

---

## 🌟 What is AtlasAI?

AtlasAI is an all-in-one career intelligence platform that combines **multiple AI models** (GPT-4o, Gemini, DeepSeek) to help job seekers make **data-driven decisions**.

Instead of guessing which jobs fit, paying for generic resume reviews, or manually researching visa trends, AtlasAI provides **personalized insights at scale**.

---

## 🎯 Core Features

### 📄 RAG-Based Document Chat

* Chat with any PDF using Retrieval-Augmented Generation (RAG)
* Grounded answers with citations
* Switch AI models mid-chat
* Private, isolated document namespaces

**Use cases:** textbooks, research papers, contracts, technical docs

---

### 🧑‍💼 Resume vs Job Description (LLM Council)

* Multi-AI resume evaluation (GPT-4o, Gemini, DeepSeek)
* Peer-review + judge synthesis
* Match score, resume quality score, strengths, gaps
* Concrete, actionable resume edits

**Why it matters:** reduces single-model bias and improves reliability

---

### 🔍 AI-Powered Job Search

* Resume-aware job search across 50+ job boards (via JSearch)
* Personalized 0–100 match score per job
* Deep resume-vs-job comparison on demand
* Extremely low cost (≈ $0.002 for 20 jobs)

**Ideal for:** students, professionals, career switchers, international applicants

---

### 🇺🇸 H1-B Sponsorship Analytics

* Built on real USCIS H1-B approval data
* 10 pre-built analyses (companies, states, industries, trends)
* Interactive charts, AI summaries, CSV export

**Perfect for:** international students & workers planning US careers

---

## 🧠 Tech Highlights

* **Multi-model AI strategy** (quality + cost optimization)
* **RAG + vector embeddings** (hallucination-resistant)
* **LLM Council w/ peer review**
* **Pinecone namespace isolation**
* **Snowflake-backed analytics**

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/kiranss777/job-board-ai.git
cd job-board-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
# OpenAI
OPENAI_API_KEY=

# Google Gemini
GOOGLE_API_KEY=

# DeepSeek
DEEPSEEK_API_KEY=

# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_HOST=

# Mistral
MISTRAL_API_KEY=

# JSearch
JSEARCH_API_KEY=

# Snowflake
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_ROLE=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DATABASE=JOB_ASSISTANT_DATABASE
SNOWFLAKE_SCHEMA=PUBLIC
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 👥 Who Is This For?

* **Students** – internships, resume tuning, H1-B research
* **Job Seekers** – personalized job ranking & resume feedback
* **Career Changers** – skill gap analysis & role matching
* **International Workers** – visa-friendly employer insights

---

## 💡 Why AtlasAI?

* ✅ Multi-AI evaluations (not single-model bias)
* ✅ Resume-aware job matching (not keyword spam)
* ✅ Real immigration data (not anecdotes)
* ✅ One platform instead of 5 tools
* ✅ Pennies per search

---

<div align="center">

[Live Demo](https://job-board-ai-jwgzdkxcadeqqsqdct9wsn.streamlit.app/) •
[GitHub](https://github.com/kiranss777/job-board-ai)

⭐ Star the repo if you find it useful!

</div>

---
