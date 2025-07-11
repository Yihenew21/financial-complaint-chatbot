"""
Utility functions for text chunking and embedding generation.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained model (e.g., 'all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')


def chunk_text(text: str, max_length: int = 512,
               overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input text to chunk
        max_length (int): Maximum length of each chunk (in tokens)
        overlap (int): Number of overlapping tokens between chunks

    Returns:
        List[str]: List of text chunks
    """
    if not text or not isinstance(text, str):
        return []

    words = text.split()
    chunks = []
    for i in range(0, len(words) - overlap, max_length - overlap):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)

    if len(words) > max_length:
        last_chunk = ' '.join(words[len(words) - max_length:])
        chunks.append(last_chunk)

    return chunks


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using SentenceTransformers.

    Args:
        texts (List[str]): List of text chunks

    Returns:
        np.ndarray: Array of embeddings
    """
    if not texts:
        return np.array([])

    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def process_complaints(csv_path: str,
                       output_dir: str = "data/embeddings/") -> Tuple[
                           List[str], np.ndarray]:
    """
    Process complaint narratives, chunk them, and generate embeddings.

    Args:
        csv_path (str): Path to the filtered complaints CSV
        output_dir (str): Directory to save embeddings

    Returns:
        Tuple[List[str], np.ndarray]: Chunks and their embeddings
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)
    if 'Cleaned_Narrative' not in df.columns:
        logger.error("Cleaned_Narrative column not found in CSV")
        return [], np.array([])

    # Chunk narratives
    all_chunks = []
    for narrative in df['Cleaned_Narrative']:
        chunks = chunk_text(narrative)
        all_chunks.extend(chunks)

    # Generate embeddings
    embeddings = generate_embeddings(all_chunks)

    # Save embeddings (optional, for debugging or later use)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    chunks_file = os.path.join(output_dir, "chunks.txt")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_chunks))

    logger.info(f"Processed {len(all_chunks)} chunks with "
                f"{embeddings.shape[0]} embeddings")
    return all_chunks, embeddings
