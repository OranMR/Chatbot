#!/usr/bin/env python3
import os
import glob
import argparse
from argparse import Namespace
import numpy as np
import faiss
import openai
import csv

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_indices(embedding_dir):
    """Load all FAISS indices from the specified directory."""
    index_files = glob.glob(os.path.join(embedding_dir, "*.index"))
    indices = {}
    for path in index_files:
        name = os.path.splitext(os.path.basename(path))[0]
        indices[name] = faiss.read_index(path)
    return indices


def load_text_chunks(embedding_dir):
    """Load all text chunk files from the specified directory."""
    text_files = glob.glob(os.path.join(embedding_dir, "*.txt"))
    chunks = {}
    for path in text_files:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().split("\n----\n")
            chunks[name] = content
    return chunks


def embed_query(query, model="text-embedding-3-small"):
    """Get the embedding for a single query using OpenAI Embeddings."""
    response = openai.embeddings.create(model=model, input=[query])
    emb = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
    return emb


def retrieve_chunks(queries, index, text_chunks, k, model):
    """Retrieve top-k chunks for each query."""
    results = []
    for query in queries:
        emb = embed_query(query, model)
        distances, indices = index.search(emb, k)
        idxs = indices[0]
        retrieved = [text_chunks[i] for i in idxs if i < len(text_chunks)]
        results.append((query, retrieved))
    return results


def read_queries(file_path):
    """Read queries from a text file, one per line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def write_csv(results, output_path, k):
    """Write the queries and their retrieved chunks to a CSV file."""
    headers = ["query"] + [f"chunk_{i+1}" for i in range(k)]
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for query, chunks in results:
            row = [query] + chunks + [''] * (k - len(chunks))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve FAISS chunks for queries and save to CSV"
    )
    parser.add_argument(
        "--questions-file", required=True,
        help="Path to file with one question per line"
    )
    parser.add_argument(
        "--embedding-dir", default="embeddings",
        help="Directory containing .index and .txt files"
    )
    parser.add_argument(
        "--index-name", required=True,
        help="Name of FAISS index (basename without extension)"
    )
    parser.add_argument(
        "--text-chunks-name", required=True,
        help="Name of text chunks file (basename without extension)"
    )
    parser.add_argument(
        "--output-csv", default="retrieved_chunks.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of neighbors to retrieve"
    )
    parser.add_argument(
        "--model", default="text-embedding-3-small",
        help="OpenAI embedding model"
    )
    args = Namespace(
    questions_file="C:/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/questions.txt",
    embedding_dir="C:/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/embeddings",  # Adjust this
    index_name="Test papers",
    text_chunks_name="Test papers",
    output_csv="output.csv",
    k=10,
    model="text-embedding-3-small"
)
# Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the output CSV path to be in the same directory as the script
    output_csv_path = os.path.join(script_dir, args.output_csv)
    
    indices = load_indices(args.embedding_dir)
    print("Indices loaded:", list(indices.keys()))

    if args.index_name not in indices:
        raise ValueError(f"Index '{args.index_name}' not found in {args.embedding_dir}")
    index = indices[args.index_name]

    text_chunks_dict = load_text_chunks(args.embedding_dir)
    print("Text chunks loaded:", list(text_chunks_dict.keys()))

    if args.text_chunks_name not in text_chunks_dict:
        raise ValueError(f"Text chunks '{args.text_chunks_name}' not found in {args.embedding_dir}")
    text_chunks = text_chunks_dict[args.text_chunks_name]

# 💬 Read queries and run retrieval
    queries = read_queries(args.questions_file)
    results = retrieve_chunks(queries, index, text_chunks, args.k, args.model)

# 💾 Write to CSV
    write_csv(results, output_csv_path, args.k)
    print(f"Saved retrieved chunks to {args.output_csv}")


if __name__ == "__main__":
    main()