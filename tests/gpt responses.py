import numpy as np
import faiss
import openai
import os

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load FAISS indices and corresponding text chunks for each file
embedding_dir = "embeddings"
indices = {}  # Dictionary to store FAISS indices for each file
text_chunks = {}  # Dictionary to store text chunks for each file

for file in os.listdir(embedding_dir): #Detects named directories (modified from original)
    folder_path = os.path.join(embedding_dir, file)
    if os.path.isdir(folder_path):  # Ensure it's a directory
        try:   
            index_path = os.path.join(folder_path, f"faiss_{file}.index")
            indices[file] = faiss.read_index(index_path)
            # Load text chunks
            text_chunk_path = os.path.join(folder_path, f"text_chunks_{file}.txt")
            with open(text_chunk_path, 'r') as f:
                text_chunks[file] = f.read().split("\n----\n")
            print(f"Loaded embeddings for {file}")      
        except Exception as e:
            # Print error message if loading fails
            print(f"Error loading data for file {file}: {e}")
        
user_query = "In 300 words, describe antimicrobial resistance and it's relationship with One Health"  # set the user query
selected_file = 'Papers'  # set the file

history = []

# Step 1: Embed the user's query using OpenAI's text-embedding-3-small
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=[user_query]
)
# Extract the embedding vector from the response
query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

# Step 2: Search for relevant text chunks using FAISS for the selected file
index = indices[selected_file]  # Get the FAISS index for the selected file
chunks = text_chunks[selected_file]  # Get the text chunks for the selected file

k = 10  # Number of nearest neighbors to retrieve
# Perform the search on the FAISS index to find the most similar text chunks
distances, indices_result = index.search(query_embedding, k)


# Gather the most relevant text chunks and their sources
relevant_texts = []  # List to store relevant text chunks
sources = []  # List to store corresponding source filenames
for idx in indices_result[0]:
    if idx < len(chunks):
        relevant_texts.append(chunks[idx])  # Add the text chunk to the list
        # Extract the source filename from the chunk metadata
        source = chunks[idx].split('\n')[0].replace('Source: ', '')
        sources.append(source)  # Add the source to the list
        
# If no relevant texts are found, return an error
if not relevant_texts:
    print ("No relevant text found")
    
# Combine the relevant texts into a single context
context = "\n".join(relevant_texts)  # Concatenate all relevant text chunks
source_info = "\n".join([f"Source: {source}" for source in sources])  # Concatenate source information

# Step 3: Prepare the full conversation for GPT-4o, including history
messages = history + [
    # Add system prompt to instruct GPT-4o to answer based on the provided context
    {"role": "system", "content": "You are describing a scientific concept to experts. Answer the user’s question based only on the following context. Ask the user for clarification if you can't fully answer based on the context. Make sure your answers are detailed. For each part of your response, reference the corresponding source file that provided the information, with the study title as it would be referred to in a scientific paper (e.g. Murray et al, 2015)."},
    {"role": "system", "content": f"Context: {context}\n{source_info}"},  # Provide context and sources
    {"role": "user", "content": user_query},  # User's query
]

# Call GPT-4o with the conversation history and context
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

# Extract the response from GPT-4o
gpt_response = response.choices[0].message.content

print(gpt_response)

