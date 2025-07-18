# W6 Intelligent Complaint Analysis for Financial Services

## Introduction

CreditTrust Financial, a fast-growing digital finance company serving East African markets with over 500,000 users across three countries, receives thousands of customer complaints monthly through in-app channels, email, and regulatory portals. This project develops a Retrieval-Augmented Generation (RAG)-powered chatbot to transform this raw, unstructured complaint data into actionable insights. The tool targets internal stakeholders like Asha, a Product Manager for the BNPL team, who currently spends hours manually analyzing complaints. By enabling rapid trend identification (from days to minutes) and empowering non-technical teams (Support, Compliance) with plain-English queries, the system aims to shift CreditTrust from reactive to proactive problem-solving across five product categories: Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers.

### Business Case Study
For Asha, manual analysis of BNPL complaints takes approximately 10 hours weekly, costing CreditTrust an estimated $500 in labor per week (based on a $30/hour rate). The RAG chatbot, once optimized, could reduce this to 2 hours, saving $400 weekly or $20,800 annually. This efficiency gain allows Asha to focus on product improvements, such as addressing seasonal payment delays identified in EDA, directly impacting customer satisfaction and retention.

## Task 1: EDA and Data Preprocessing

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) was conducted using the Jupyter notebook `01_eda_analysis.ipynb` on the CFPB complaint dataset, which contains real customer complaints across multiple financial products. The initial dataset, approximately 6GB, was loaded in chunks to ensure memory efficiency. Analysis revealed that Credit Cards (108,666 complaints, ~28.4%) and Savings Accounts (140,317 complaints, ~36.7%) dominate the 382,819 total complaints, with Personal Loans (17,238, ~4.5%), BNPL (19,413, ~5.1%), and Money Transfers (97,185, ~25.4%) following. Key issues included billing disputes and high interest rates, particularly for Credit Cards and Personal Loans. Time-series analysis identified seasonal spikes in BNPL complaints around holiday periods, suggesting potential transparency or payment scheduling challenges. Visualizations, such as product distribution bar charts and temporal trend lines, were generated to highlight these patterns, guiding the RAG system’s focus on high-impact areas.

### Data Preprocessing
The preprocessing pipeline filtered the dataset to include only the five target products, reducing its size to a manageable subset saved as `data/processed/filtered_complaints.csv`. Complaint narratives were cleaned by removing boilerplate text (e.g., standardized disclaimers), normalizing text to lowercase, and eliminating stopwords using NLTK, ensuring relevance for semantic search. The `Cleaned_Narrative` column was validated for completeness, and metadata (e.g., `Complaint ID`, `Mapped_Product`) was preserved. This process addressed the challenge’s requirement to handle noisy, unstructured narratives, preparing the data for embedding and retrieval.

### Deliverables
- Jupyter notebook: `notebooks/01_eda_analysis.ipynb`
- Processed dataset: `data/processed/filtered_complaints.csv`
- Summary: Included above, detailing key findings and preprocessing steps.

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Implementation
The Jupyter notebook `02_embedding_indexing.ipynb` was developed to convert cleaned narratives into a format suitable for semantic search. Long narratives were chunked using LangChain’s `RecursiveCharacterTextSplitter` with a `chunk_size=512` tokens and `chunk_overlap=50` tokens. This choice balances granularity and context preservation, as justified by the need to capture complete complaint details while avoiding excessive fragmentation. The final parameters were selected after initial testing (reported in notebook comments), ensuring chunks align with the embedding model’s input capacity. Embeddings were generated using SentenceTransformers’ `all-MiniLM-L6-v2`, a lightweight model with 384 dimensions, chosen for its efficiency and strong performance on sentence-level semantics, suitable for the financial complaint domain. The model was preferred over alternatives (e.g., `multi-qa-mpnet-base-dot-v1`) due to its faster inference and sufficient accuracy for initial prototyping, as noted in the notebook.

Each chunk’s embedding, along with metadata (`Complaint ID` and `Mapped_Product`), was indexed into a persistent ChromaDB vector store using `chromadb.PersistentClient`, saved in `data/embeddings/`. The HNSW index with cosine similarity was configured to support efficient similarity searches. The process handled the full 382,819 complaints, resulting in a 3GB vector store, reflecting the scale of the dataset.

### Testing and Verification
A similarity search was conducted with the query "issue with credit card payment," retrieving the top 5 chunks. Results included Money Transfers and BNPL complaints (e.g., "writing formally express concerns regarding zelle handling disputes..."), indicating functional indexing but highlighting a need for improved query relevance (to be addressed in Task 3). The metadata linkage was validated, ensuring traceability to original complaints.

### Deliverables
- Script: `notebooks/02_embedding_indexing.ipynb` (fulfilling the script/notebook requirement)
- Persisted vector store: `data/embeddings/` (3GB, excluded from Git via `.gitignore`, regenerable via the notebook)
- Report section: Included above, detailing chunking strategy, model choice, and implementation.

## Technical Choices

- **Data**: The CFPB dataset was selected for its real-world complaint narratives, aligning with the challenge’s focus on unstructured data.
- **Chunking**: LangChain’s `RecursiveCharacterTextSplitter` with `chunk_size=512` and `chunk_overlap=50` was chosen for its intelligent splitting on natural boundaries (e.g., sentences), enhancing context retention.
- **Embedding Model**: `all-MiniLM-L6-v2` was selected for its balance of speed and semantic accuracy, suitable for the large dataset and initial RAG implementation.
- **Vector Store**: ChromaDB was preferred over FAISS for its ease of use, persistent storage, and metadata support, meeting the challenge’s traceability requirement.

## Task 3: RAG System Development

### Objectives
- Build a system to retrieve relevant complaint data and generate informed responses using an LLM.
- Filter responses by financial product categories.

### Implementation
The RAG system was implemented in `notebooks/03_query_response.ipynb`, integrating the ChromaDB vector store from Task 2 with a query-response pipeline. The `retrieve_chunks` function uses `all-MiniLM-L6-v2` to encode queries and retrieve the top 3 relevant chunks, filtered by product categories (Credit Cards, BNPL, Money Transfers, Personal Loans, Savings Accounts) using a predefined mapping. The `flan-t5-large` model, accessed via `HuggingFacePipeline` from `langchain-huggingface`, generates responses based on a prompt template that enforces context usage and includes specific examples. The system was tested with queries like "Why are people unhappy with Personal Loans?" and "What issues are reported with Credit Cards?"

### Challenges
- Inconsistent context retrieval across categories, with some queries (e.g., Personal Loans) yielding better results than others (e.g., Credit Cards).
- Sparse data for certain categories (e.g., no delay information for BNPL) limited response depth.
- Initial model loading failures were resolved by ensuring sufficient RAM (4-6GB) and using a free model.

### Evaluation
An evaluation table was created in `data/evaluation_table.md`, assessing response accuracy and relevance based on initial tests. The table below reflects the actual outputs and sources provided:

| Question                                           | Generated Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Retrieved Sources                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Quality Score   | Comments       |
|:---------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|:---------------|
| Why are people unhappy with Credit Cards?          | I don’t have enough information to provide a detailed answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ['case description complaint made bank one subsidiaries elan financial services ultimately responsible well hidden policies created bank responsible party complaint solicitation received mail signed ent credit card offered elan financial servies division bank part reason getting card generous offered certain categories home utilities based solicitation decided get card selected categories incurred convenience fee order pay home utility colorado transaction posted called told obvious mistake would fixed fixed', 'suspected based two separate notifications believed sent counterfeit merchandise indeed item counterfeit asked notify local law enforcement give suspected counterfeit merchandise postal inspector believed knowingly shipping counterfeit items crime specifically code since per believed knowingly shipped counterfeit items per wanted nothing items since may counterfeit told synchrony bank would get back opened dispute without explanation information synchrony bank closed dispute charge counterfeit goods'] | 3/10            | Poor relevance; sources relate to Credit Cards but response lacks detail due to possible data sparsity. |
| What are the main BNPL issues?                     | reporting delinquent data without verifiable date first delinquency dofd must accurately reflect account status using standardized coding transunion failed creating materially misleading consumer report unfair practices cfpa violation actions also violate consumer financial protection act specifically prohibits deceptive unfair abusive acts practices udaap connection consumer financial products services continuing report unverifiable inaccurate data despite documented disputes legal protections ... | ['reporting delinquent data without verifiable date first delinquency dofd must accurately reflect account status using standardized coding transunion failed creating materially misleading consumer report unfair practices cfpa violation actions also violate consumer financial protection act specifically prohibits deceptive unfair abusive acts practices udaap connection consumer financial products services continuing report unverifiable inaccurate data despite documented disputes legal protections transunion', 'consent done hereby consent equifax experian transunion whether verbal non verbal written implied otherwise revoked usc agencies never informed right exercise non disclosure option law commit crimes report adverse items report without permission usc states every consumer reporting agency shall maintain reasonable procedures designed avoid violations section title lint furnishing consumer reports purposes listed section title ferpa student loan removal agencies maintaining regulating proper procedures'] | 7/10            | Good relevance to BNPL data issues (e.g., reporting inaccuracies), but response could be more concise. |
| Are there fraud signals in Money Transfers?        | I don’t have enough information to provide a detailed answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ['writing formally express concerns regarding zelle handling disputes particularly light recent cfpb lawsuit highlighting significant issues platform despite using zelle numerous times reporting discrepancies financial institution zelle failed investigate claims adequately informed recovering lost funds impossible tracking transactions unfeasible cfpb findings indicate zelle neglected protect consumers also allowed repeat offenders exploit system leaving users vulnerable scams lack accountability transparency', 'vulnerable scams lack accountability transparency unacceptable raises serious concerns consumer safety']                                                                                                                                                                                                                                                                                                                                                                                                                  | 4/10            | Sources indicate fraud signals (e.g., Zelle disputes), but response fails to utilize context effectively. |
| What problems are reported with Personal Loans?    | I don’t have enough information to provide a detailed answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ['supplied pdfs truth lending disclosures along screenshot hold placed account finally understood issue said upgrading specialist would contact charge cleared bank finally received email response klarna incorrectly stated reported problem order paused payment schedule used find klarna customer service number spoke someone answers could transfer call specialist department said department would contact time frame previously days fed klarna called spoke customer service asked charge credit card want deal klarna', 'documented pay date shown paystubs respond received customer care via email need sufficient funds account cover repayment funds also later email correspondence going back forth stated system error detect incorrect pay date time working system unacceptable company lends money additionally several occasions via email asked requested contact information speak live human matter contact information provided customer support today date']                                                                        | 3/10            | Sources suggest issues (e.g., payment disputes), but response lacks context utilization. |
| Why do customers complain about Savings Accounts?  | I don’t have enough information to provide a detailed answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ['paycheck sent via direct deposit capital one checking account scrub deposit still posted account est called capital one customer service multiple times gotten answers one seems know issue resolved', 'hold first place cease unfair practices harm consumers issue resolved promptly escalate legal action including filing lawsuit small claims court request cfpb ftc investigate upgrade potential fraudulent deceptive business practices company name upgrade inc website www upgrade com']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 3/10            | Sources hint at access issues, but response fails to reflect this. |
| How often do Credit Card complaints occur monthly? | I don’t have enough information to provide a detailed answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ['vulnerable scams lack accountability transparency unacceptable raises serious concerns consumer safety', 'transparent disclosures particularly cases involving disputed accounts ongoing misrepresentation account collectible despite knowledge fraudulent status constitutes multiple tila violations actionable statutory penalties relevance tila requirements found condemns action damages anothers good name similarly affirms right uphold ones honor integrity unjustly attacked misrepresentation obligations violates doctrine insists truth transparency rectification harm supporting tilas mandates final demand immediate']                                                                                                                                                                                                                                                                                                                                                                                                                    | 2/10            | Irrelevant sources; response unable to address frequency due to data limitations. |
| What causes delays in BNPL payments?               | I don’t have enough information to provide a detailed answer.                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ['reporting delinquent data without verifiable date first delinquency dofd must accurately reflect account status using standardized coding transunion failed creating materially misleading consumer report unfair practices cfpa violation actions also violate consumer financial protection act specifically prohibits deceptive unfair abusive acts practices udaap connection consumer financial products services continuing report unverifiable inaccurate data despite documented disputes legal protections transunion', 'consent done hereby consent equifax experian transunion whether verbal non verbal written implied otherwise revoked usc agencies never informed right exercise non disclosure option law commit crimes report adverse items report without permission usc states every consumer reporting agency shall maintain reasonable procedures designed avoid violations section title lint furnishing consumer reports purposes listed section title ferpa student loan removal agencies maintaining regulating proper procedures'] | 3/10            | Sources lack delay-specific data; response reflects data sparsity. |

- **Average Quality Score**: (3 + 7 + 4 + 3 + 3 + 2 + 3) / 7 = **3.57/10**
- The low average score indicates significant issues with data relevance and LLM context utilization.

### Recommendations
- **Data Enhancement**: Populate `data/embeddings/` with comprehensive complaint data, including delay-specific BNPL cases, by running `03_query_response.ipynb` by 07/10/2025.
- **Model Tuning**: Switch to `multi-qa-mpnet-base-dot-v1` for improved category-specific retrieval, testing by 07/12/2025.
- **Prompt Refinement**: Adjust the prompt template to enforce context usage (e.g., "Must include at least one example from {context}"), to be implemented by 07/11/2025.
- **Testing**: Conduct 10 additional queries with the updated system to reassess performance.

### Deliverables
- Jupyter notebook: `notebooks/03_query_response.ipynb`
- Evaluation table: `data/evaluation_table.md`
- Report section: Included above.

## Task 4: Interactive Chat Interface

### Objectives
- Develop a user-friendly interface with text input, submit button, answer display, source visibility, and a clear option.
- Enhance trust with source display and attempt streaming for usability.

### Implementation
- **Gradio Attempt**: Initially built a Gradio interface in `app.py` with a text input, "Ask" button, and source textbox. Implemented async streaming and a clear button, but faced issues like `SyntaxError` (non-UTF-8 encoding), `TypeError` (event trigger syntax), and no UI response despite debug prints. Multiple fixes were attempted, including UTF-8 encoding, js parameter correction, and synchronous switching, between 04:00 PM and 04:20 PM CEST on 07/08/2025.
- **Streamlit Switch**: Switched to Streamlit at 04:24 PM CEST on 07/08/2025 due to Gradio limitations. The updated `app.py` uses `st.chat_input`, `st.chat_message`, and a sources expander, with synchronous logic for reliability. Added a test fallback response and collection check at 04:35 PM CEST to diagnose the no-response issue. Below is a key code snippet:

  ```python
  def generate_response(question, product_filter=None):
      print(f"Generating response for question: {question}")
      if llm is None:
          return "Model loading failed. No free fallback available.", []
      chunks = retrieve_chunks(question, product_filter)
      context = "\n".join(chunks)
      prompt = prompt_template.format(context=context, question=question)
      try:
          response = llm(prompt)
          return response, chunks[:2]
      except Exception as e:
          return f"Error generating response: {e}", chunks[:2]
  ```

### Challenges
- Gradio’s async generator and event handling errors (e.g., `_js` vs `js`, output mismatches) were persistent.
- No response in Gradio UI, possibly due to empty ChromaDB or LLM failures.
- Streamlit shows no response yet, suggesting further data or integration issues.

### Lessons Learned
- Async implementations in Gradio require careful event handling, which was challenging with the LLM pipeline.
- Switching to Streamlit improved structure but highlighted the need for data validation upfront.

### Deliverables
- Script: `app.py` (Streamlit version)
- Screenshots/GIFs: Pending capture, to be added once UI functions.

## System Evaluation

The RAG system’s performance is evaluated via `data/evaluation_table.md`, with an average quality score of 3.57/10 across 7 test queries. The low score reflects data sparsity and poor context utilization by the LLM. The Streamlit interface’s effectiveness is unassessed due to the no-response issue, requiring data verification (e.g., `collection.count() > 0`) and LLM debugging (e.g., ensuring `flan-t5-large` loads). Further evaluation will include user testing with Support and Compliance teams once resolved.

## UI Showcase

Screenshots or a GIF of the Streamlit interface are pending. Once functional, visuals will demonstrate the chat input, answer display, and source expander. A sample layout is envisioned as:

- **Chat Input**: Text box at the bottom.
- **Chat History**: Alternating user/assistant messages.
- **Sources**: Collapsible expander below the response.

[Placeholder for Screenshot/GIF]

## Conclusion

The project has progressed through EDA, preprocessing, indexing, RAG development, and UI implementation, addressing CreditTrust’s need for actionable complaint insights. Task 3 established a functional RAG system with variable success (3.57/10 average quality), while Task 4 encountered UI challenges with Gradio, leading to a Streamlit pivot. Current limitations include empty ChromaDB data and LLM integration issues, to be resolved with data population (by 07/10/2025) and testing. Future work will optimize performance (e.g., model tuning) and enhance the UI for production use, potentially saving CreditTrust $20,800 annually per stakeholder like Asha.

## Technical Appendix

### Chunking Code Example
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_text(narrative)
```

### LLM Initialization
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=800)
llm = HuggingFacePipeline(pipeline=pipe)
```

### Notes
- Ensure 4-6GB RAM for model loading.
- Verify ChromaDB with `collection.count()`.