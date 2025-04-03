import os
import pandas as pd
import numpy as np
import faiss
import openai
from pdfminer.high_level import extract_text
import itertools
from tqdm import tqdm  # For progress bars
import re
from difflib import SequenceMatcher  # For fuzzy matching

# Set your OpenAI API key (or ensure it is set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using PDFMiner."""
    pdf_texts = {}
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"Error: No PDF files found in {directory}")
        return pdf_texts
        
    for filename in tqdm(pdf_files, desc="Loading PDFs"): 
        filepath = os.path.join(directory, filename)
        try:
            text = extract_text(filepath)
            if not text.strip():
                print(f"WARNING: No text extracted from {filename}")
                continue
            # text.strip() = normalize whitespace (converts spaces, tabs, enters etc into 1 space)
            text = re.sub(r'\s+', ' ', text).strip()
            pdf_texts[filename] = text
            print(f"Loaded {filename}: {len(text)} characters")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return pdf_texts

def chunk_text_overlap(text, chunk_size, overlap):
    """Chunk text into pieces of approximately chunk_size characters with a given overlap. Includes function to split at full stops"""
    chunks = []
    start = 0
    text = text.strip()
    # Use larger chunk size to avoid splitting answers
    if len(text) <= chunk_size:
        return [text]
    while start < len(text):
        # Find end position for this chunk
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to find a sentence end (period followed by space) near the chunk_size
        sentence_end = text.rfind('. ', start + chunk_size - 100, start + chunk_size + 100)
        if sentence_end != -1:
            end = sentence_end + 1  # Include the period
        else:
            # If no sentence end, find a space
            space = text.rfind(' ', start + chunk_size - 100, start + chunk_size + 100)
            if space != -1:
                end = space      
        chunks.append(text[start:end])
        # Move start position for next chunk, accounting for overlap
        start = end - overlap
    return chunks

def generate_embeddings(chunks, model_name):
    """Generate embeddings for each text chunk using the specified OpenAI embedding model. Returns a list of embeddings."""
    embeddings = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        if not chunk.strip():
            print(f"WARNING: Empty chunk at index {i}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 1536)  # Using 1536 for text-embedding-3-large dimension. 3-large is usually 3072, we can specify. 
            continue
        try:
            response = openai.embeddings.create(
                model=model_name,
                input=[chunk]
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error embedding chunk {i+1}/{total_chunks}: {e}")
            # Use dimension appropriate for the model
            dimension = 3072 if "large" in model_name else 1536
            embeddings.append([0.0] * dimension)
    return embeddings

def build_library_index(library_dir, chunk_size, overlap, embedding_model):
    """Loads PDFs from the given library directory, chunks each PDF with the given parameters,
    computes embeddings with the chosen embedding model, and builds a FAISS index."""
    pdf_texts = load_pdfs_from_directory(library_dir)
    if not pdf_texts:
        raise ValueError(f"No valid PDF texts found in {library_dir}")
    all_chunks = []
    metadata = []  # e.g., (source filename, chunk number, chunk_text)
    print("Chunking documents...")
    for filename, text in pdf_texts.items(): #processes and chunks text
        chunks = chunk_text_overlap(text, chunk_size=chunk_size, overlap=overlap)
        chunk_start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        for i, chunk in enumerate(chunks): # Store more metadata including the actual text
            metadata.append({
                "filename": filename, 
                "chunk_index": i + chunk_start_idx,
                "text": chunk   
            })   
    print(f"Total chunks created: {len(all_chunks)}")
    if len(all_chunks) == 0:
        raise ValueError("No chunks were created from the PDFs")
    
    # Generate embeddings
    embed_list = generate_embeddings(all_chunks, embedding_model)
    # Convert embeddings to a numpy float32 array (storage format)
    embeddings = np.array(embed_list).astype('float32')
    # Validate embeddings
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings were generated")
    if np.isnan(embeddings).any():
        print("WARNING: NaN values found in embeddings")
        # Replace NaN values with zeros
        embeddings = np.nan_to_num(embeddings)
    
    # Build FAISS index (using cosine similarity instead of L2 for better semantic matching)
    dimension = embeddings.shape[1]
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    # Use an inner product index (equivalent to cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, all_chunks, metadata

def similarity_score(text1, text2):
    """Calculate fuzzy matching similarity between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def run_experiment(library_dir, queries_csv, chunk_size, overlap, embedding_model, k):
    """Runs one experiment with the given parameters with improved matching logic."""
    print(f"\nRunning experiment: chunk_size={chunk_size}, overlap={overlap}, "
          f"embedding_model={embedding_model}, k={k}")
          
    # Build library index
    index, chunks, metadata = build_library_index(library_dir, chunk_size, overlap, embedding_model)
    
    # Load queries CSV. Assumes CSV has columns "query" and "answer".
    try:
        df = pd.read_csv(queries_csv)
        if "query" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV file must contain 'query' and 'answer' columns")
    except Exception as e:
        print(f"Error loading queries CSV: {e}")
        return 0
        
    total_queries = len(df)
    success_count = 0
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        query = row["query"]
        correct_answer = str(row["answer"]).strip()
        if not correct_answer:
            print(f"WARNING: Empty answer for query {idx}: '{query}'")
            continue
            
        try:
            # Embed the query
            response = openai.embeddings.create(
                model=embedding_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            print(f"Error embedding query {idx}: {e}")
            continue

        # Search FAISS index for the top k nearest neighbours
        scores, indices_result = index.search(query_embedding, k)
        best_score = 0
        best_chunk = ""
        found = False
        
        for i, idx in enumerate(indices_result[0]):
            if idx >= len(chunks):
                continue
                
            chunk = chunks[idx]
            chunk_norm = re.sub(r'\s+', ' ', chunk).lower()
            
            ''' Tries several matching approaches:'''
            # 1. Direct substring match
            if correct_answer in chunk_norm or chunk_norm in correct_answer:
                found = True
                best_chunk = chunk
                best_score = 1.0
                break
                
            # 2. Fuzzy matching (for minor differences)
            sim_score = similarity_score(correct_answer, chunk_norm)
            if sim_score > 0.8:  # 80% similarity threshold, can set otherwise
                found = True
                if sim_score > best_score:
                    best_score = sim_score
                    best_chunk = chunk
                    
            # 3. Word overlap ratio
            answer_words = set(correct_answer.split())
            chunk_words = set(chunk_norm.split())
            if len(answer_words) > 0:
                overlap_ratio = len(answer_words.intersection(chunk_words)) / len(answer_words)
                if overlap_ratio > 0.8:  # 80% word overlap
                    found = True
                    if overlap_ratio > best_score:
                        best_score = overlap_ratio
                        best_chunk = chunk
        
        if found:
            success_count += 1
            
        # Store detailed results for debugging
        results.append({
            "query": query,
            "answer": correct_answer,
            "found": found,
            "best_score": best_score,
            "best_chunk": best_chunk if best_chunk else ""
        })

    success_rate = success_count / total_queries if total_queries > 0 else 0
    print(f"Success rate: {success_rate:.3f} ({success_count}/{total_queries})")
    # Save detailed results for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"Results for cs{chunk_size}_ol{overlap}_k{k}.csv", index=False)
    return success_rate

def main():
    # Define your paths
    library_dir = "Experiment Test Papers"
    queries_csv = "Experiment Questions.csv"
    
    # Define parameter ranges for experiments
    chunk_sizes = [2000]  # Try larger chunk sizes
    overlaps = [50, 100]             # Try larger overlaps
    embedding_models = ["text-embedding-3-large"]
    k_values = [5]            # Try more neighbors
    
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
    # Save to CSV
    results_df.to_csv("experiment_results.csv", index=False)
    
    # Show best performing configuration
    if not results_df.empty:
        best_row = results_df.loc[results_df['success_rate'].idxmax()]
        print("\nBest configuration:")
        print(f"Chunk size: {best_row['chunk_size']}")
        print(f"Overlap: {best_row['overlap']}")
        print(f"Model: {best_row['embedding_model']}")
        print(f"k: {best_row['k']}")
        print(f"Success rate: {best_row['success_rate']:.3f}")

if __name__ == "__main__":
    main()