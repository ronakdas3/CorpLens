# 🚀 ResearchLens

**ResearchLens** is an AI-powered research assistant that allows users to interact with research papers using natural language.

Upload a paper or choose from default datasets, ask questions, and get context-aware answers powered by Retrieval-Augmented Generation (RAG).

---

## 🌐 Live Demo

👉 https://researchlenss.streamlit.app/

---

## ✨ Features

* 📄 Ask questions about research papers
* 📤 Upload your own PDF and query it instantly
* 📚 Switch between multiple research papers
* 🔍 Semantic search using FAISS
* 🧠 Cross-encoder reranking for better relevance
* 🤖 LLM-based answer generation
* 💬 Chat-style interface (like ChatGPT)
* ⚙️ Adjustable answer length

---

## 🧠 How It Works (RAG Pipeline)

```
PDF → Text Extraction → Chunking → Embeddings → FAISS Index  
→ Retrieval → Reranking → LLM → Answer
```

---

## 🏗️ Architecture

* **Text Extraction**: Extracts raw text from PDFs
* **Chunking**: Splits text into meaningful segments
* **Embeddings**: Converts text into vector representations
* **FAISS Index**: Enables fast similarity search
* **Retriever**: Finds relevant chunks for a query
* **Reranker**: Improves relevance using cross-encoder
* **LLM**: Generates final answer from retrieved context

---

## 🖥️ Demo Use Cases

Try asking:

* What is the main contribution of the paper?
* What architecture does the paper propose?
* What dataset was used?
* How does the model work?

---

## ⚙️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **Vector DB**: FAISS
* **Embeddings**: Sentence Transformers
* **LLM**: FLAN-T5 / TinyLlama (local)
* **Reranking**: Cross-encoder

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/researchlens.git
cd researchlens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build index (optional)

```bash
python -m src.indexing.build_index
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## 📂 Project Structure

```
app/
  └── streamlit_app.py

src/
  ├── data/
  ├── embeddings/
  ├── indexing/
  ├── models/
  ├── retrieval/

data/
  └── papers/
```

---

## ⚠️ Deployment Notes

* Lightweight model used for cloud deployment
* Larger models (TinyLlama) used locally for better performance
* Handles CPU-only environments

---

## 🔮 Future Improvements

* Inline citations in answers
* Hybrid search (BM25 + embeddings)
* Multi-document memory
* Improved UI/UX
* Better LLM integration

---

## 💡 Key Learnings

* Built a full RAG pipeline from scratch
* Solved real-world deployment issues (CPU, memory limits)
* Designed scalable multi-document architecture
* Balanced model performance vs resource constraints

---

## 👨‍💻 Author

Ronak Das

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
