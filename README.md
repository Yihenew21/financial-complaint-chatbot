# Financial Complaint Chatbot

This repository contains an **intelligent complaint analysis chatbot** built as part of the **KAIM5 10 Academy project** for CreditTrust Financial, a digital finance company serving East African markets. The chatbot leverages **Retrieval-Augmented Generation (RAG)** to analyze customer complaints across multiple financial products and provides an interactive interface for querying insights.

---

## Project Overview

CreditTrust Financial, with over **500,000 users across three countries**, receives thousands of monthly complaints. This tool transforms **raw, unstructured complaint data into actionable insights** for product managers, support teams, and compliance officers, enabling a shift from reactive to proactive problem-solving.

---

## Business Objective

✅ **Reduce analysis time** from days to minutes for identifying major complaint trends.\
✅ **Empower non-technical teams** with direct semantic search insights.\
✅ **Enhance customer experience** by leveraging real-time feedback.

---

## Key Features

- **Semantic search** through complaint narratives using ChromaDB and SentenceTransformers.
- **RAG pipeline** with HuggingFace LLM (`flan-t5-large`) for informed natural language responses.
- **Multi-product analysis** across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.
- **Interactive chat interface** implemented with Streamlit (Gradio attempt documented but switched for stability).
- **Source attribution** for transparency (planned for production).

---

## Project Structure

```
financial-complaint-chatbot/
│
├── app.py                      # Streamlit-based interactive chat interface
├── requirements.txt            # Project dependencies
├── README.md                   # This file
│
├── data/
│   ├── raw/                    # Original CFPB dataset (e.g. complaints.csv)
│   ├── processed/              # Cleaned dataset (e.g. filtered_complaints.csv)
│   ├── embeddings/             # ChromaDB vector store (excluded from Git due to 3GB size)
│   └── evaluation_results.csv  # Evaluation outputs
│
├── notebooks/
│   ├── 01_eda_analysis.ipynb   # Exploratory data analysis and preprocessing
│   ├── 02_embedding_indexing.ipynb # Chunking, embedding, vector store creation
│   ├── 03_rag_pipline.ipynb # RAG pipeline: retrieval + generation
│   └── 04_evaluation.ipynb      # Interactive UI (pending)
│
└── src/
    ├── data_loader.py          # Chunked data loading and category mapping
    └── embedding_utils.py      # Utility functions for embedding (optional)
```

---

## Tasks Progress

✅ **Task 1: EDA & Data Preprocessing**

- Completed using `01_eda_analysis.ipynb`.
- Processed **382,819 complaints**, mapped to five product categories, saved as `filtered_complaints.csv`.

✅ **Task 2: Text Chunking & Vector Store**

- Implemented chunking with LangChain’s `RecursiveCharacterTextSplitter` and embedding with `all-MiniLM-L6-v2`.
- Indexed chunks into a **3GB ChromaDB vector store** in `data/embeddings/`.

🟡 **Task 3: RAG Pipeline & Evaluation**

- Developed in `03_rag_pipline.ipynb`.
- Built retrieval + generation pipeline using **ChromaDB** for embeddings and **HuggingFace flan-t5-large** for response generation.
- **Challenges**:
  - Inconsistent context retrieval for certain categories (e.g., worked for Personal Loans but not Credit Cards).
  - Sparse data for BNPL delays.
  - Initial model loading issues resolved by switching to free models.
- **Evaluation pending** due to current integration limitations.

🟡 **Task 4: Interactive Chat Interface**

- Implemented with **Streamlit** after Gradio issues:
  - Gradio faced async generator errors, no UI response despite debug prints.
  - Streamlit implementation uses `st.chat_input` and `st.chat_message`, with fallback responses and collection health checks.
- **Current status**:
  - Streamlit UI runs but ** response displayed**. Requires further debugging of `collection.count()` and terminal outputs.

---

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <https://github.com/Yihenew21/financial-complaint-chatbot.git>
   cd financial-complaint-chatbot
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download CFPB dataset**

   - Place in `data/raw/complaints.csv`.

5. **Run notebooks in order**

   - `notebooks/01_eda_analysis.ipynb`: Data exploration & preprocessing.
   - `notebooks/02_embedding_indexing.ipynb`: Chunking, embedding, vector store creation.
   - `notebooks/03_query_response.ipynb`: RAG system pipeline and evaluation.

6. **Launch the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   - Access at: [http://localhost:8501](http://localhost:8501)

---

## Recent Work

- **July 8, 2025**:
  - Debugged Gradio async issues, switched to Streamlit.
  - Implemented session state, spinner, and source expander in Streamlit.
  - Added test fallback response and collection health checks to diagnose persistent no-response issue.

---

## Next Steps

- Verify ChromaDB collection data using:
  ```python
  collection.count()
  ```
- Debug LLM response pipeline outputs.
- Capture UI screenshots and GIFs for the final `report.md` upon resolution.
- Implement **source attribution display** for trust-building.

---

## Dependencies

- `sentence-transformers`
- `chromadb`
- `langchain`
- `streamlit`
- `huggingface_hub`

*(See **`requirements.txt`** for complete versions)*

---

## Key Dates

- **Final Submission**: Tuesday, July 8, 2025 (8:00 PM UTC)\
  Submit GitHub `main` branch and final `report.md` in Medium blog format.

---

## Additional Notes

- `data/embeddings/` (3GB) is excluded via `.gitignore` but **regenerable** by rerunning `02_embedding_indexing.ipynb`.
- Detailed evaluation and screenshots are pending final pipeline resolution.

---

*End of README.*

---

Let me know if you need this committed to your repository with a tailored commit message for your upcoming final submission.

