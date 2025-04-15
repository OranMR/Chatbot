from flask import Flask, render_template, request, jsonify
import numpy as np
import faiss
import openai
from openai import OpenAIError
import os

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

# Debug info, prints path to pythond backend location
print(f"Current working directory: {os.getcwd()}")

# System prompts dictionary
SYSTEM_PROMPTS = {
    '1': "You are speaking to a child, be as basic as possible and use many brainrotted (for example skibidi toilet references) and gamer terms as possible, ideally at least 1 reference or term per sentence",
    '2': "You are describing a scientific concept to experts. Answer the user's question based only on the following context. Ask the user for clarification if you can't fully answer based on the context. Make sure your answers are detailed. Refer to specfic studies if appropriate. For each part of your response, reference the corresponding study that provided the information, with the study title as it would be referred to in a scientific paper (e.g. Murray et al, 2015)."
}

# Display names for the prompt styles
PROMPT_NAMES = {
    '1': "Basic",
    '2': "Advanced"
}

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Raise an error if the API key is not found
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Load FAISS index and text chunks from their (single) directory. Currently must be named faiss_index.index etc. Change MyEPDFs code to name smth like 'saved papers'
embedding_dir = "embeddings"
index_path = os.path.join(embedding_dir, "faiss_index.index")
text_chunks_path = os.path.join(embedding_dir, "text_chunks.txt")

# Print debug info
print(f"Embeddings directory: {embedding_dir}") #see line 34
print(f"Index path: {index_path}") 
print(f"Text chunks path: {text_chunks_path}")

# Load the FAISS index and text chunks. Print statements ensure they are all loaded correctly
try:
    if os.path.exists(index_path):
        embeddings = faiss.read_index(index_path)
        print("FAISS index loaded successfully")
    else:
        print(f"Index file not found at {index_path}")
        embeddings = None
        
    if os.path.exists(text_chunks_path):
        with open(text_chunks_path, 'r') as f:
            text_chunks = f.read().split("\n----\n")
        print(f"Loaded {len(text_chunks)} text chunks successfully")
    else:
        print(f"Text chunks file not found at {text_chunks_path}")
        text_chunks = []
        
except Exception as e:
    print(f"Error loading embeddings: {str(e)}")
    embeddings = None
    text_chunks = []

# Route to render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle chat API requests
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Parse JSON data from the POST request
        data = request.json
        user_query = data.get('message', '').strip()
        prompt_style = data.get('file')  # Now this is just the prompt style ID
        history = data.get('history', [])

        # Validate the user query and prompt style
        if not user_query or not prompt_style:
            return jsonify({"error": "Empty query or style not selected"}), 400

        # Make sure we have a valid prompt style
        if prompt_style not in SYSTEM_PROMPTS:
            return jsonify({"error": "Invalid style selected"}), 400
            
        # Check if embeddings were loaded successfully
        if embeddings is None:
            return jsonify({"error": "Embeddings not loaded. Please check your index file."}), 500

        # Step 1: Embed the user's query
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
        
        # Step 2: Search for relevant text chunks using FAISS
        k = 10  # Number of nearest neighbors
        distances, indices_result = embeddings.search(query_embedding, k)

        # Check if any relevant text chunks were found
        if len(indices_result[0]) == 0:
            return jsonify({"error": "No relevant text found"}), 404

        # Gather the most relevant text chunks and their sources
        relevant_texts = []
        sources = []
        for idx in indices_result[0]:
            if idx < len(text_chunks):
                relevant_texts.append(text_chunks[idx])
                source = text_chunks[idx].split('\n')[0].replace('Source: ', '')
                sources.append(source)

        # Combine the relevant texts into a single context
        context = "\n".join(relevant_texts)
        source_info = "\n".join([f"Source: {source}" for source in sources])

        # Step 3: Get the system prompt based on the selected style
        system_prompt = SYSTEM_PROMPTS[prompt_style]
        
        # Prepare the conversation for GPT-4o
        messages = history + [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context: {context}\n{source_info}"},
            {"role": "user", "content": user_query},
        ]
            
        # Call GPT-4o with the conversation history and context
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # Extract the response
        gpt_response = response.choices[0].message.content

        # Return the response to the frontend
        return jsonify({"response": gpt_response})

    except OpenAIError as e:
        return jsonify({"error": f"Error communicating with GPT-4o: {str(e)}"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))