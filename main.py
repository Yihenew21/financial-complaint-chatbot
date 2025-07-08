# -*- coding: utf-8 -*-
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Initialize models and client
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="data/embeddings/")
collection = client.get_collection(name="complaint_embeddings")

print(f"Collection count: {collection.count()}")
if collection.count() == 0:
    print("Warning: ChromaDB collection is empty. Please populate it with data.")

try:
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=800, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=pipe)
    print("LLM initialized successfully.")
except Exception as e:
    print(f"Error loading flan-t5-large: {e}. Using fallback message.")
    llm = None

def retrieve_chunks(query, product_filter=None, k=3):
    query_embedding = embedding_model.encode([query])
    product_map = {
        "credit cards": "Credit Cards",
        "bnpl": "Buy Now, Pay Later (BNPL)",
        "money transfers": "Money Transfers",
        "personal loans": "Personal Loans",
        "savings accounts": "Savings Accounts"
    }
    if product_filter:
        filter_product = product_map.get(product_filter.lower(), product_filter)
        print(f"Filtering for product: {filter_product}")
        results = collection.query(
            query_embeddings=query_embedding,
            where={"mapped_product": filter_product},
            n_results=k
        )
    else:
        print("No product filter applied")
        results = collection.query(query_embeddings=query_embedding, n_results=k)
    print(f"Retrieved chunks: {results['documents'][0]}")
    return results['documents'][0]

prompt_template = """
You are a financial analyst assistant for CreditTrust. Your task is to analyze customer complaints based on the provided context.
You MUST use ALL relevant details from the context to summarize the main issues or reasons for complaints in a clear, concise paragraph. Include at least one specific example (e.g., late fees, fraud disputes, payment timing issues) directly from the context. If the context is empty or completely irrelevant to the question, and only then, state: 'I don’t have enough information to provide a detailed answer.'
Context: {context}
Question: {question}
Answer:
"""

def generate_response(question, product_filter=None):
    print(f"Generating response for question: {question}")
    if llm is None:
        print("Fallback triggered due to no LLM")
        return "Model loading failed. No free fallback available.", []
    chunks = retrieve_chunks(question, product_filter)
    context = "\n".join(chunks)
    prompt = prompt_template.format(context=context, question=question)
    print(f"Prompt generated: {prompt[:100]}...")
    try:
        response = llm(prompt)
        print(f"Full response: {response}")
        return response, chunks[:2]
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}", chunks[:2]
    if not response or "I don’t have enough information" in response:
        print("Falling back to test response due to empty or default output")
        return "Test response: Please check data or model setup.", ["Test source"]

# Streamlit UI
st.title("CreditTrust Complaint Analyzer")
st.caption("Ask about customer complaints related to financial products.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you with complaint analysis today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Enter your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response, sources = generate_response(prompt)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    with st.expander("Sources"):
        st.write("\n".join(sources) if sources else "No sources available.")