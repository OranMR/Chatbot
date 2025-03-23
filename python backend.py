from flask import Flask, render_template, request, jsonify
import numpy as np
import faiss
import openai
from openai import OpenAIError
import os

# Initialize Flask app
app = Flask(__name__) # Start Flask application

#Set users (source prompt codes)
Person1="Basic"
Person2="Advanced"

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Raise an error if the API key is not found
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Load FAISS indices and corresponding text chunks for each file
embedding_dir = "embeddings"
indices = {}  # Dictionary to store FAISS indices for each file
text_chunks = {}  # Dictionary to store text chunks for each file

for file in os.listdir(embedding_dir): #Detects named directories (modified from original)
    folder_path = os.path.join(embedding_dir, file) 
    if os.path.isdir(folder_path):  # Ensure it's a directory
        try:   
            index_path = os.path.join(folder_path, f"faiss_{file}.index") #links faiss file to directory path
            indices[file] = faiss.read_index(index_path)
            # Load text chunks
            text_chunk_path = os.path.join(folder_path, f"text_chunks_{file}.txt") #links chunks to directory path
            with open(text_chunk_path, 'r') as f:
                 text_chunks[file] = f.read().split("\n----\n") #creates file containing all text chunks
            print(f"Loaded embeddings for {file}")      
        except Exception as e:
            # Print error message if loading fails
            print(f"Error loading data for file {file}: {e}")

# Route to render the frontend (index.html)
@app.route('/')#sends the html upon opening
def index():
    # Render the main HTML page
    return render_template('index.html') #html must be called index.html

# Route to handle chat API requests
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Parse JSON data from the POST request
        data = request.json
        user_query = data.get('message', '').strip()  # Extract the user query
        selected_user = data.get('file')  # Extract the selected file from the frontend             may change, may be redundant
        history = data.get('history', [])  # Extract conversation history. History is stored in users' browser, we do not need to define it

    # Convert file selection to string matching directory name. May simplify later, will likely require coding in front and back ends
        file_mapping = {
             '1': Person1,
             '2': Person2
         }
        selected_user = file_mapping.get(selected_user)
        print(f"Mapped file: {selected_user}")


        # Validate the user query and selected user, basically a troubleshooter for if the search engine geeks out
        if not user_query or not selected_user:
            return jsonify({"error": "Empty query or file not selected"}), 400

        # Ensure the selected file is found in the chunks/file dictionary
        if selected_user not in indices:
            return jsonify({"error": "Invalid file selected"}), 400

        # Step 1: Embed the user's query using OpenAI's text embedding models (3-small, 3-large, ada-002),     tweak 
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        # Extract the embedding vector from the response
        query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
        
        # Step 2: Search for relevant text chunks using FAISS for the selected file
        index = indices[selected_user]  # Get the FAISS index for the selected file
        chunks = text_chunks[selected_user]  # Get the text chunks for the selected file

        k = 10  # Number of nearest neighbors to retrieve                                    tweak
        # Perform the search on the FAISS index to find the most similar text chunks
        distances, indices_result = index.search(query_embedding, k)

        # Check if any relevant text chunks were found, error checking to ensure chunks are found
        if len(indices_result[0]) == 0:
            return jsonify({"error": "No relevant text found"}), 404

        # Gather the most relevant text chunks and their sources,       can be modified to include links to original texts
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
            return jsonify({"error": "No relevant texts found"}), 404

        # Combine the relevant texts into a single context
        context = "\n".join(relevant_texts)  # Concatenate all relevant text chunks
        source_info = "\n".join([f"Source: {source}" for source in sources])  # Concatenate source information

        # Step 3: Prepare the full conversation for GPT-4o, including history                         tweak
        if selected_user==Person1:
            messages = history + [
                # Add system prompt to instruct GPT-4o to answer based on the provided context
                {"role": "system", "content": "You are speaking to a child, be as basic as possible and use many brainrotted (for example skibidi toilet references) and gamer terms as possible, ideally at least 1 reference or term per sentence"},
                {"role": "system", "content": f"Context: {context}\n{source_info}"},  # Provide context and sources
                {"role": "user", "content": user_query},  # User's query
            ]
        elif selected_user==Person2:
            messages = history + [
                # Add system prompt to instruct GPT-4o to answer based on the provided context
                {"role": "system", "content": "You are describing a scientific concept to experts. Answer the user’s question based only on the following context. Ask the user for clarification if you can't fully answer based on the context. Make sure your answers are detailed. For each part of your response, reference the corresponding study that provided the information, with the study title as it would be referred to in a scientific paper (e.g. Murray et al, 2015)."},
                {"role": "system", "content": f"Context: {context}\n{source_info}"},  # Provide context and sources
                {"role": "user", "content": user_query},  # User's query
                ]
#Add in other source prompts as necessary
            
        # Call GPT-4o with the conversation history and context
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # Extract the response from GPT-4o
        gpt_response = response.choices[0].message.content

        # Step 4: Return the response to the frontend
        return jsonify({"response": gpt_response})

    except OpenAIError as e:
        # Handle errors from the OpenAI API (or specific to GPT-4o)
        return jsonify({"error": f"Error communicating with GPT-4o: {str(e)}"}), 500

    except Exception as e: # Lord pray we never see this
        # Handle any other unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    

# Run the Flask app
if __name__ == "__main__":
    # Run the app on the specified host and port
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    #works for prototype, deployed on hiroku. Final production version may be deployed someplace else