import os
import pandas as pd
import numpy as np
import faiss
import openai
from pdfminer.high_level import extract_text
import itertools
import time

# Set your OpenAI API key (or ensure it is set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using PDFMiner."""
    pdf_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            try:
                pdf_texts[filename] = extract_text(filepath)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return pdf_texts

def chunk_text_overlap(text, chunk_size=500, overlap=50):
    """Chunk text into pieces of approximately chunk_size characters with a given overlap."""
    chunks = []
    text = text.strip()
    while len(text) > chunk_size:
        split_at = text.rfind(" ", 0, chunk_size) #prevents splitting in the middle of a word
        if split_at == -1:  # No space found, force split at chunk size
            split_at = chunk_size
        chunks.append(text[:split_at]) #adds new chunks to list
        text = text[split_at:].strip()  # Strip leading spaces for next chunk
    chunks.append(text)
    return chunks


def generate_embeddings(chunks, model_name):
    """Generate embeddings for each text chunk using the specified OpenAI embedding model.
    Returns a list of embeddings."""
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            response = openai.embeddings.create(
                model=model_name,
                input=[chunk]
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            # (Optional) add a short sleep to be courteous with rate limits
            time.sleep(0.1)
            print(f"Embedding for {i} generated")
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
            embeddings.append([0.0] * 768)  # fallback (adjust dimensionality as needed)
    return embeddings


def build_library_index(library_dir, chunk_size, overlap, embedding_model):
    """Loads PDFs from the given library directory, chunks each PDF with the given parameters,
    computes embeddings with the chosen embedding model, and builds a FAISS index.
    Returns:
        index      : the FAISS index built on the chunk embeddings.
        chunks     : list of all text chunks.
        metadata   : list of metadata tuples (e.g. (filename, chunk_index))."""
    pdf_texts = load_pdfs_from_directory(library_dir)
    all_chunks = []
    metadata = []  # e.g., (source filename, chunk number)
    for filename, text in pdf_texts.items():
        chunks = chunk_text_overlap(text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        metadata.extend([(filename, i) for i in range(len(chunks))])
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Generate embeddings
    embed_list = generate_embeddings(all_chunks, embedding_model)
    # Convert embeddings to a numpy float32 array (storage format)
    embeddings = np.array(embed_list).astype('float32') 
    print("embeddings converted")
    
    # Build FAISS index (using L2 distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("library constructed")
    return index, all_chunks, metadata


def run_experiment(library_dir, queries_csv, chunk_size, overlap, embedding_model, k):
    """Runs one experiment with the given parameters.
    Loads and indexes the library, then for each query, checks if the correct answer snippet
    is retrieved in the top k results.
    Returns the success rate (proportion of queries for which the answer was found)."""
   
    print(f"\nExperiment: chunk_size={chunk_size}, overlap={overlap}, "
          f"embedding_model={embedding_model}, k={k}")
    # Build library index
    index, chunks, metadata = build_library_index(library_dir, chunk_size, overlap, embedding_model)
    
    # Load queries CSV. Assumes CSV has columns "query" and "answer".
    df = pd.read_csv(queries_csv)
    total_queries = len(df)
    success_count = 0

    for idx, row in df.iterrows():
        query = row["query"]
        correct_answer = str(row["answer"]).strip()  # ground truth snippet 

        try:
            # Embed the query
            response = openai.embeddings.create(
                model=embedding_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
        except Exception as e:
            print(f"Error embedding query {idx}: {e}")
            continue

        # Search FAISS index for the top k nearest neighbours
        distances, indices_result = index.search(query_embedding, k)
        retrieved_chunks = [chunks[i] for i in indices_result[0] if i < len(chunks)]
        
        # Check if any retrieved chunk contains the correct answer (using a substring check)
        found = any(correct_answer.lower() in chunk.lower() or chunk.lower() in correct_answer.lower()
                    for chunk in retrieved_chunks)
        if found:
            success_count += 1

    success_rate = success_count / total_queries if total_queries > 0 else 0
    print(f"Success rate: {success_rate:.3f} ({success_count}/{total_queries})")
    return success_rate

def main():
    # Define your paths
    library_dir = "Experiment Test Papers"   # update this path to your PDFs
    queries_csv = "Experiment Questions.csv"          # update this path to your queries CSV
    
    # Define parameter ranges for experiments
    chunk_sizes = [200, 500, 1000]          # e.g., chunk sizes in characters
    overlaps = [0, 50, 100]                 # e.g., overlap in characters
    embedding_models = ["text-embedding-3-small", "text-embedding-3-large"]
    k_values = [1, 5, 10]                   # number of nearest neighbours to retrieve
    
    # List to hold results
    experiment_results = []
    
    # Iterate over all parameter combinations
    for chunk_size, overlap, model_name, k in itertools.product(chunk_sizes, overlaps, embedding_models, k_values):
        try:
            rate = run_experiment(library_dir, queries_csv, chunk_size, overlap, model_name, k)
            experiment_results.append({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "embedding_model": model_name,
                "k": k,
                "success_rate": rate
            })
        except Exception as e:
            print(f"Experiment failed for parameters "
                  f"chunk_size={chunk_size}, overlap={overlap}, model={model_name}, k={k}: {e}")
    
    # Convert results to DataFrame and print/save
    results_df = pd.DataFrame(experiment_results)
    print("\nExperiment Results:")
    print(results_df)
    # Optionally, save to CSV
    results_df.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    main() 