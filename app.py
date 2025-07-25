# -*- coding: utf-8 -*-
import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize models and client
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="data/embeddings/")
collection = client.get_collection(name="complaint_embeddings")

# Check collection data
print(f"Collection count: {collection.count()}")
if collection.count() == 0:
    print("Warning: ChromaDB collection is empty. Please populate it with data.")

try:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
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
    print(f"Retrieved chunks: {results['documents'][0]}")  # Debug retrieved data
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
    print(f"Prompt generated: {prompt[:100]}...")  # Debug prompt
    try:
        response = llm(prompt)
        print(f"Full response: {response}")  # Debug full response
        return response, chunks[:2]
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {e}", chunks[:2]
    # Test fallback response
    if not response or "I don’t have enough information" in response:
        print("Falling back to test response due to empty or default output")
        return "Test response: Please check data or model setup.", ["Test source"]

# Gradio interface
with gr.Blocks(title="CreditTrust Complaint Analyzer") as demo:
    gr.Markdown("# CreditTrust Complaint Analyzer")
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Enter your question", placeholder="e.g., Why are people unhappy with Credit Cards?")
            submit_btn = gr.Button("Ask")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            output = gr.Textbox(label="Answer", interactive=False)
            sources = gr.Textbox(label="Sources", interactive=False)
    
    submit_btn.click(
        fn=generate_response,
        inputs=[question_input, gr.State(value=None)],
        outputs=[output, sources],
        js="() => [document.querySelector('gradio-textbox input').value, null]",
        api_name="submit"
    )

    clear_btn.click(
        fn=lambda: ("", []),
        inputs=[],
        outputs=[output, sources]
    )

demo.launch()