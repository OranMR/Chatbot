from flask import Flask, render_template, request, jsonify
import numpy as np
import faiss
import openai
from openai import OpenAIError
import os
import glob

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

gpt_model="gpt-4o"

# Display names for the prompt styles
PROMPT_NAMES = {
    '1': "Basic",
    '2': "Advanced"
}

# System prompts dictionary
SYSTEM_PROMPTS = {
    '1': "",
    '2': "You are a chatbot representing Edinburgh Infectious Diseases, an organisation connecting scientists who conduct\
    research into infectious diseases who are based in Edinburgh. You will be hosted on their website\
    The users of the chatbot are familiar with the general biology of disease and related topics.\
    Answer the user's question based primarily on the following context (studies conducted in the city of Edinburgh).\
    Ask the user for clarification if you can't fully answer based on the context. \
    Make sure your answers are detailed and refer to specfic studies listed in the context if appropriate.\
    For each part of your response, reference the study that provided the information, with the study title \
    as it would be referred to in a scientific paper, for example (Study title, 2015).\
    Keep answers concise, but if detailed answers are warranted give them.\
    For short simple questions, such as definitional queries, give short simple responses.\
    Be professional but passionate about the work being done. As you are a chatbot, keep messages conversational and digestible."
}

#     '2': "Persona: You are an RAG chatbot for the Edinburgh Infectious Diseases organization. You are professional, friendly, and passionate \
#         about the research being conducted in Edinburgh. Your main goal is to facilitate collaboration among researchers by \
#         providing accurate and insightful answers to their inquiries regarding ongoing research, concepts, themes, and networking\
#         opportunities.\
#     Rules:\
#     Always maintain a professional yet approachable tone\
#     Provide references for info provided using studies and data from the following context whenever possible\
#     Be concise but thorough in your responses\
#     Encourage further questions or networking among users\
#     Directly address user needs and inquiries with relevant information\
#     Research Focus:\
# \
#     Stay updated on current research themes and studies happening in Edinburgh related to infectious diseases\
#     Highlight notable collaborations and networking opportunities within the organization\
# "

    
# Debug info, prints path to python backend location
print(f"Current working directory: {os.getcwd()}")

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Raise an error if the API key is not found
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Get embedding directory from environment variable or use default
embedding_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
print(f"Embeddings directory: {embedding_dir}")

# Find all available FAISS index files and text chunk files
index_files = glob.glob(os.path.join(embedding_dir, "*.index"))
text_files = glob.glob(os.path.join(embedding_dir, "*.txt"))

# Create dictionaries to store loaded indices and text chunks
loaded_indices = {}
loaded_text_chunks = {}

# Load all available FAISS indices
for index_path in index_files:
    index_name = os.path.basename(index_path).split('.')[0]
    try:
        loaded_indices[index_name] = faiss.read_index(index_path)
        print(f"FAISS index loaded successfully: {index_name}")
    except Exception as e:
        print(f"Error loading FAISS index {index_name}: {str(e)}")

# Load all available text chunk files
for text_path in text_files:
    text_name = os.path.basename(text_path).split('.')[0]
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read().split("\n----\n")
            loaded_text_chunks[text_name] = text_content
            print(f"Loaded {len(text_content)} text chunks successfully from {text_name}")
    except Exception as e:
        print(f"Error loading text chunks {text_name}: {str(e)}")

# Print available datasets
print(f"Available indices: {list(loaded_indices.keys())}")
print(f"Available text chunks: {list(loaded_text_chunks.keys())}")

# Route to render the frontend
@app.route('/')
def index():
    # Pass the available datasets to the frontend
    return render_template('index.html', 
                          indices=list(loaded_indices.keys()), 
                          text_chunks=list(loaded_text_chunks.keys()),
                          prompt_names=PROMPT_NAMES)

# API endpoint to get available datasets
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    return jsonify({
        "indices": list(loaded_indices.keys()),
        "text_chunks": list(loaded_text_chunks.keys()),
        "prompt_styles": PROMPT_NAMES
    })

# Route to handle chat API requests
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Parse JSON data from the POST request
        data = request.json
        user_query = data.get('message', '').strip()
        prompt_style = data.get('file')  # Now this is just the prompt style ID
        history = data.get('history', [])
        
        # Get dataset names from request
        index_name = data.get('index_name')
        text_chunks_name = data.get('text_chunks_name')
        
        # If no specific datasets are provided, use the first available ones
        if not index_name and loaded_indices:
            index_name = list(loaded_indices.keys())[0]
        
        if not text_chunks_name and loaded_text_chunks:
            text_chunks_name = list(loaded_text_chunks.keys())[0]

        # Validate the user query and prompt style
        if not user_query or not prompt_style:
            return jsonify({"error": "Empty query or style not selected"}), 400

        # Make sure we have a valid prompt style
        if prompt_style not in SYSTEM_PROMPTS:
            return jsonify({"error": "Invalid style selected"}), 400
            
        # Check if the requested index and text chunks exist
        if index_name not in loaded_indices:
            return jsonify({"error": f"Index '{index_name}' not found"}), 404
            
        if text_chunks_name not in loaded_text_chunks:
            return jsonify({"error": f"Text chunks '{text_chunks_name}' not found"}), 404
            
        # Get the selected index and text chunks
        embeddings = loaded_indices[index_name]
        text_chunks = loaded_text_chunks[text_chunks_name]

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
            model=gpt_model,
            messages=messages
        )

        # Extract the response
        gpt_response = response.choices[0].message.content

        # Return the response to the frontend
        return jsonify({
            "response": gpt_response,
            "used_index": index_name,
            "used_text_chunks": text_chunks_name
        })

    except OpenAIError as e:
        return jsonify({"error": f"Error communicating with {gpt_model}: {str(e)}"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))