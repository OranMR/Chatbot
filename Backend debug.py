import os
import re
import numpy as np
import faiss
import openai
from flask import Flask, render_template, request, jsonify
from pdfminer.high_level import extract_text
from tqdm import tqdm

# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Flask app initialization
app = Flask(__name__, template_folder='frontend')

# System prompts dictionary
SYSTEM_PROMPTS = {
    '1': "You are speaking to a child, be as basic as possible and use many brainrotted (for example skibidi toilet references) and gamer terms as possible, ideally at least 1 reference or term per sentence",
    '2': "You are a chatbot representing an organisation conducting research into infectious diseases, and describing a scientific concept to people familiar with the topic.\
    Answer the user's question based primarily on the following context (studies conducted in the city of Edinburgh).\
    Ask the user for clarification if you can't fully answer based on the context. \
    Make sure your answers are detailed and refer to specfic studies listed in the context if appropriate.\
    For each part of your response, reference the corresponding study that provided the information, with the study title \
    as it would be referred to in a scientific paper, for example (Study title, 2015).\
    Try to keep answers concise, but prioritise detail over concision. Be professional but passionate about the work being done"
}

# Display names for the prompt styles
PROMPT_NAMES = {
    '1': "Basic",
    '2': "Advanced"
}

# Original Functions from My EPDFs.py
def preprocess_text(text):
    """Clean and normalize extracted PDF text"""
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\b\d+\s*\|\s*Page\b', '', text)
    text = re.sub(r'\bPage\s*\d+\s*of\s*\d+\b', '', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\f', ' ', text)  # Form feed characters
    
    # Remove headers/footers (customize based on your PDFs)
    text = re.sub(r'(?i)confidential|draft|internal use only', '', text)
    
    # Fix hyphenated words that span line breaks (common in PDFs)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Clean up quotes and apostrophes
    text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
    
    # Remove URLs and emails if needed
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # Strip extra spaces and normalize spacing
    text = text.strip()
    return text

def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using pdfMiner."""
    pdf_texts = {}
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    # Progress bar for PDF loading
    for filename in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        filepath = os.path.join(directory, filename)
        raw_text = extract_text(filepath)
        pdf_texts[filename] = preprocess_text(raw_text)  # Apply preprocessing
    return pdf_texts

def chunk_text(text, chunk_size=500, overlap=150):
    """Chunk text into pieces with overlap between chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end position for this chunk
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            # Try to find a sentence boundary (., !, ?) near the end of chunk
            sentence_end = -1
            for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                pos = text.rfind(punct, start, end + 50)
                if pos > sentence_end and pos <= end + 50:
                    sentence_end = pos + 1  # Include the punctuation
            if sentence_end > start:
                end = sentence_end
            else:
                # No sentence boundary found, try to find a space
                space = text.rfind(' ', end - 50, end + 50)
                if space > start:
                    end = space + 1
        
        # Add the chunk to our list
        chunks.append(text[start:end].strip())
        
        # Move start position for next chunk, accounting for overlap
        start = end - overlap if end - overlap > start else end
    return chunks

def generate_embeddings(chunks, batch_size=20):
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings", unit="batch"):
        batch = chunks[i:i + batch_size]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings, dtype=np.float32)

def store_embeddings_and_chunks(file, chunks, embeddings, filenames, output_dir='embeddings'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving chunks and embeddings for {file}...")

    # Save chunks with optional metadata
    chunks_filepath = os.path.join(output_dir, f"{file}.txt")
    with open(chunks_filepath, 'w') as f:
        for i, (chunk, filename) in enumerate(zip(chunks, filenames)):
            f.write(f"Source: {filename} | \n{chunk}\n----\n")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create and save FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_filepath = os.path.join(output_dir, f"{file}.index")
    faiss.write_index(index, index_filepath)

    print(f"Successfully saved {len(chunks)} chunks and embeddings for {file}")

def processpdfs(base_directory):
    """Process all PDFs."""
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    # Main progress bar for overall subdirectories
    for file in tqdm(subdirectories, desc="Processing directories", unit="dir"):  
        file_dir = os.path.join(base_directory, f"{file}")
        if os.path.exists(file_dir):
            print(f"\nProcessing directory: {file}")
            # Load PDFs and extract text
            pdf_texts = load_pdfs_from_directory(file_dir)
            
            # Process each PDF's text
            all_chunks = []
            all_filenames = []
            
            # Progress bar for processing individual PDFs
            for filename, text in tqdm(pdf_texts.items(), desc="Processing PDFs", unit="pdf"):
                chunks = chunk_text(text, chunk_size=500, overlap=150)
                all_chunks.extend(chunks)
                all_filenames.extend([filename] * len(chunks))
            
            # Generate embeddings
            embeddings = generate_embeddings(all_chunks)
            
            # Store the embeddings and chunks
            store_embeddings_and_chunks(file, all_chunks, embeddings, all_filenames)
        else:
            print(f"{file} directory not found. Skipping.")





# Testing Functions
def check_text_chunks_format(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = content.split("\n----\n")
            print(f"Found {len(chunks)} chunks")
            
            # Check first few chunks
            for i, chunk in enumerate(chunks[:5]):
                if i < len(chunks) - 1:  # Skip last chunk if it's empty
                    print(f"\nChunk {i} (length: {len(chunk)}):")
                    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    
                    # Check if chunk has expected format
                    if not chunk.startswith("Source:"):
                        print(f"WARNING: Chunk {i} doesn't start with 'Source:'")
                    
                    # Check for binary data (simple heuristic)
                    non_ascii = len([c for c in chunk if ord(c) > 127])
                    if non_ascii > len(chunk) * 0.1:  # More than 10% non-ASCII
                        print(f"WARNING: Chunk {i} may contain binary data ({non_ascii} non-ASCII chars)")
    except UnicodeDecodeError:
        print("File contains binary data that cannot be decoded as UTF-8")

def check_embedding_dimensions(index_path):
    index = faiss.read_index(index_path)
    dimension = index.d
    vector_count = index.ntotal
    
    print(f"Index dimensions: {dimension}")
    print(f"Number of vectors: {vector_count}")
    
    # Test query embedding dimensions
    test_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=["This is a test query"]
    )
    test_dimension = len(test_response.data[0].embedding)
    print(f"Test query embedding dimension: {test_dimension}")
    
    if dimension != test_dimension:
        print("DIMENSION MISMATCH: Index and query embeddings have different dimensions")
    else:
        print("Dimensions match between index and query embeddings")

def test_normalization(index_path, query):
    # Load index
    index = faiss.read_index(index_path)
    
    # Get embedding for query
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    # Test with and without normalization
    print("Without normalization:")
    distances1, indices1 = index.search(query_embedding, 3)
    print(f"Top distances: {distances1}")
    print(f"Top indices: {indices1}")
    
    # Normalize and test again
    faiss.normalize_L2(query_embedding)
    print("\nWith normalization:")
    distances2, indices2 = index.search(query_embedding, 3)
    print(f"Top distances: {distances2}")
    print(f"Top indices: {indices2}")

def validate_index_and_chunks(index_path, text_chunks_path):
    # Load index
    index = faiss.read_index(index_path)
    vector_count = index.ntotal
    
    # Count text chunks
    with open(text_chunks_path, 'r', encoding='utf-8') as f:
        content = f.read()
        chunks = content.split("\n----\n")
        # Remove empty chunks
        chunks = [c for c in chunks if c.strip()]
        chunk_count = len(chunks)
    
    print(f"Vector count in index: {vector_count}")
    print(f"Chunk count in text file: {chunk_count}")
    
    if vector_count != chunk_count:
        print("COUNT MISMATCH: Number of vectors doesn't match number of text chunks")
    else:
        print("Counts match between vectors and text chunks")

def test_retrieval(index_path, text_chunks_path, query):
    # Load index
    index = faiss.read_index(index_path)
    
    # Load text chunks
    with open(text_chunks_path, 'r', encoding='utf-8') as f:
        text_chunks = f.read().split("\n----\n")
        # Remove empty chunks
        text_chunks = [c for c in text_chunks if c.strip()]
    
    # Get embedding for query
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    # Normalize (as done in your original code)
    faiss.normalize_L2(query_embedding)
    
    # Search
    k = 3  # Get top 3 results
    distances, indices = index.search(query_embedding, k)
    
    print(f"Query: {query}")
    print(f"Top distances: {distances[0]}")
    print(f"Top indices: {indices[0]}")
    
    # Show retrieved chunks
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(text_chunks):
            print(f"\nResult {i+1} (index {idx}, distance {distances[0][i]}):")
            chunk = text_chunks[idx]
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        else:
            print(f"\nResult {i+1}: Index {idx} out of bounds (text chunks length: {len(text_chunks)})")

def fix_corrupted_text_chunks(text_chunks_path, output_path):
    """Attempt to fix corrupted text chunks by removing binary data"""
    try:
        with open(text_chunks_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            chunks = content.split("\n----\n")
            
            # Filter out chunks that don't start with "Source:"
            valid_chunks = []
            for chunk in chunks:
                if chunk.strip() and chunk.strip().startswith("Source:"):
                    # Remove any non-printable characters
                    clean_chunk = ''.join(c if c.isprintable() or c.isspace() else ' ' for c in chunk)
                    valid_chunks.append(clean_chunk)
            
            print(f"Original chunks: {len(chunks)}")
            print(f"Valid chunks after filtering: {len(valid_chunks)}")
            
            # Write clean chunks to new file
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write("\n----\n".join(valid_chunks))
            
            print(f"Cleaned text chunks saved to {output_path}")
    except Exception as e:
        print(f"Error fixing text chunks: {str(e)}")

def run_all_tests(embedding_dir, index_file, text_file, query):
    """Run all tests on the given embedding files"""
    index_path = os.path.join(embedding_dir, index_file)
    text_path = os.path.join(embedding_dir, text_file)
    
    print("===== Testing Text Chunks Format =====")
    check_text_chunks_format(text_path)
    
    print("\n===== Testing Embedding Dimensions =====")
    check_embedding_dimensions(index_path)
    
    print("\n===== Validating Index and Chunk Counts =====")
    validate_index_and_chunks(index_path, text_path)
    
    print("\n===== Testing Normalization =====")
    test_normalization(index_path, query)
    
    print("\n===== Testing Complete Retrieval Pipeline =====")
    test_retrieval(index_path, text_path, query)
    
    # If issues were found, offer to fix text chunks
    clean_text_path = os.path.join(embedding_dir, f"clean_{text_file}")
    print("\n===== Attempting to Fix Text Chunks =====")
    fix_corrupted_text_chunks(text_path, clean_text_path)
    
    # Test with fixed chunks
    print("\n===== Testing Retrieval with Cleaned Chunks =====")
    test_retrieval(index_path, clean_text_path, query)

if __name__ == "__main__":
    embedding_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
    index_file = input("Enter name of FAISS index file (e.g., example.index): ")
    text_file = input("Enter name of text chunks file (e.g., example.txt): ")
    test_query = input("Enter a test query: ")
    
    run_all_tests(embedding_dir, index_file, text_file, test_query)
    print("\n✅ Testing complete!")