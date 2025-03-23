import os
from pdfminer.high_level import extract_text #library, takes pdf and strips text so python can use it. Extract text means we can use that as shorthand
import numpy as np #library for numeric operations
import faiss #database allowing fast vector calculation
import openai
#for pipinstall, install pdfminer.six, faiss-cpu

# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using pdfMiner."""
    pdf_texts = {} #creates empty space for pdfs
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename) #links file name with location in directory
            pdf_texts[filename] = extract_text(filepath)  # Extract text with PDFMiner
    return pdf_texts


def chunk_text(text, chunk_size=500):                                                                         #change, add overlaps
    """Chunk text into pieces of approximately chunk_size characters."""
    chunks = [] #empty list to put chunks in
    while len(text) > chunk_size:
        split_at = text.rfind(" ", 0, chunk_size) #prevents splitting in the middle of a word
        if split_at == -1:  # No space found, force split at chunk size
            split_at = chunk_size
        chunks.append(text[:split_at]) #adds new chunks to list
        text = text[split_at:].strip()  # Strip leading spaces for next chunk
    chunks.append(text)
    return chunks


def generate_embeddings(chunks): #takes each chunk and makes a mathematical embedding for it
    """Generate embeddings for each text chunk using OpenAI's text-embedding-3-small model.""" #can try 3-large as well
    embeddings = []#creates empty list for embeddings
    for chunk in chunks: #loops through chunks and embeds
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[chunk]  # Ensure input is a list
        )
        embedding = response.data[0].embedding # Access the embedding via response.data[0].embedding attribute
        embeddings.append(embedding) #adds embedded chunks to embeddings
    return np.array(embeddings) 


def store_embeddings_and_chunks(file, chunks, embeddings, filenames, output_dir='embeddings'):
    """Store embeddings and their corresponding chunks."""
    os.makedirs(f"{output_dir}/{file}", exist_ok=True)  #makes a dirctory structure with the format embeddings/file
   #Will allow the bot to store and call info in different places, allowing you to make the responses variable (e.g. to user)
   
   # Save chunks with filenames
    """Allows bot to return files with the output, meaning you can effectively reference your answer"""
    chunks_filepath = os.path.join(output_dir, f"{file}/text_chunks_{file}.txt") #makes filepath to store chunks in
    with open(chunks_filepath, 'w') as f: #opens the file for writing (w)
        for chunk, filename in zip(chunks, filenames):
            f.write(f"Source: {filename}\n{chunk}\n----\n")
    # Save embeddings
    embeddings_filepath = os.path.join(output_dir, f"{file}/faiss_{file}.index")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, embeddings_filepath)


def processpdfs(base_directory): #The culmination. Adds all pdfs to directory
    """Process all PDFs."""
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    for file in subdirectories:  
        file_dir = os.path.join(base_directory, f"{file}") 
        if os.path.exists(file_dir):
            print(f"Processing {file}...")
            # Load PDFs and extract text
            pdf_texts = load_pdfs_from_directory(file_dir)
            # Process each PDF's text
            all_chunks = []
            all_filenames = []
            for filename, text in pdf_texts.items():
                print (f"Processing {file} {filename}")
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                all_filenames.extend([filename] * len(chunks))
            
            # Generate embeddings
            embeddings = generate_embeddings(all_chunks)
            
            # Store the embeddings and chunks
            store_embeddings_and_chunks(file, all_chunks, embeddings, all_filenames)
        else:
            print(f"{file} directory not found. Skipping.")


# Run the processing
processpdfs('/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Test Papers') #test folder
print ("Done") #message to let you know code has finished running
