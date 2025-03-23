import os
from pdfminer.high_level import extract_text #library, takes pdf and strips text so python can use it. Extract text means we can use that as shorthand
import numpy as np #library for numeric operations
import faiss #database allowing fast vector calculation
import openai
#for pipinstall, install pdfminer.six, faiss-cpu


# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using PDFMiner."""
    pdf_texts = {} #creates empty space for pdfs
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            pdf_texts[filename] = extract_text(filepath)  # Extract text with PDFMiner
    return pdf_texts

def chunk_text(text, chunk_size=500):
    """Chunk text into pieces of approximately chunk_size characters."""
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind(" ", 0, chunk_size) #prevents splitting in the middle of a word
        if split_at == -1:  # No space found, force split
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:].strip()  # Strip leading spaces for next chunk
    chunks.append(text)
    return chunks

def generate_embeddings(chunks):
    """Generate embeddings for each text chunk using OpenAI's text-embedding-3-small model.""" #can try 3-large as well
    embeddings = []#creates empty space for embeddings
    for chunk in chunks: #loops through chunks and embeds
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[chunk]  # Ensure input is a list
        )
        # Access the embedding via response.data[0].embedding attribute
        embedding = response.data[0].embedding
        embeddings.append(embedding) #adds embedded chinks to embeddings
    return np.array(embeddings) 

def store_embeddings_and_chunks(week, chunks, embeddings, filenames, output_dir='embeddings'):
    """Store embeddings and their corresponding chunks."""
    os.makedirs(f"{output_dir}/week{week}", exist_ok=True)  
   #Will have to majorly change this for mine
   #Will allow the bot to store and call info in different places, allowing you to make the responses variable (e.g. to user)
    
    # Save chunks with filenames
    """Allows bot to return files with the output, meaning you can effectively reference your answer"""
    chunks_filepath = os.path.join(output_dir, f"week{week}/text_chunks_week_{week}.txt")
    with open(chunks_filepath, 'w') as f:
        for chunk, filename in zip(chunks, filenames):
            f.write(f"Source: {filename}\n{chunk}\n----\n")
    
    # Save embeddings
    embeddings_filepath = os.path.join(output_dir, f"week{week}/faiss_week_{week}.index")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, embeddings_filepath)

def process_weekly_pdfs(base_directory): #The culmination. Adds all pdfs to directory, here in week-specific folders
    """Process all PDFs for each week."""
    for week in range(1, 11):  # Assuming 10 weeks
        week_dir = os.path.join(base_directory, f"week{week}")
        if os.path.exists(week_dir):
            print(f"Processing Week {week}...")
            
            # Load PDFs and extract text
            pdf_texts = load_pdfs_from_directory(week_dir)
            
            # Process each PDF's text
            all_chunks = []
            all_filenames = []
            for filename, text in pdf_texts.items():
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                all_filenames.extend([filename] * len(chunks))
            
            # Generate embeddings
            embeddings = generate_embeddings(all_chunks)
            
            # Store the embeddings and chunks
            store_embeddings_and_chunks(week, all_chunks, embeddings, all_filenames)
        else:
            print(f"Week {week} directory not found. Skipping.")

# Run the processing
process_weekly_pdfs('/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Test papers1') #test folder
