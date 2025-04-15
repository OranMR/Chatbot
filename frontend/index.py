<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edinburgh Infectious Diseases Chatbot</title>

    <!-- Import Montserrat Font from Google Fonts -->
    <link href="index2_files/css2.css" rel="stylesheet">

    <style>
        body {
            font-family: 'Avenir', sans-serif;
            background-color: #f4f4f4;
            color: #333;
            margin: 0;
            padding: 20px;
            transition: background-color 0.3s, color 0.3s;
            display: flex;
            justify-content: space-between;
        }

        body.dark-mode {
            background-color: #121212;
            color: #ffffff;
        }

        .left-side {
            width: 30%;
            text-align: left;
        }

        .left-side img {
            width: 400px; /* Increased size */
            height: auto;
            margin-bottom: 20px;
        }

        .left-side h1 {
            color: #333; /* Changed to match text color */
            margin-bottom: 10px;
        }

        .left-side p {
            margin-bottom: 20px;
            font-size: 1rem;
            line-height: 1.5;
            color: #555;
        }

        .dark-mode .left-side h1 {
            color: #ffffff; /* Matches text color in dark mode */
        }

        .dark-mode .left-side p {
            color: #bbbbbb;
        }

        #dark-mode-toggle {
            cursor: pointer;
            padding: 10px 20px;
            background-color: #cccccc; /* Gray background */
            color: #333;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            font-family: 'Avenir', sans-serif;
            transition: background-color 0.3s ease;
        }

        #dark-mode-toggle:hover {
            background-color: #aaaaaa; /* Darker gray on hover */
        }

        .dark-mode #dark-mode-toggle {
            background-color: #444444; /* Gray for dark mode */
            color: #ffffff;
        }

        .dark-mode #dark-mode-toggle:hover {
            background-color: #666666; /* Darker gray for dark mode */
        }

        .right-side {
            width: 65%;
            text-align: left;
        }

        #file-select {
            padding: 10px;
            font-size: 1rem;
            font-family: 'Avenir', sans-serif;
            border: 1px solid #cccccc;
            border-radius: 5px;
            background-color: #f0f0f0;
            margin-bottom: 20px;
        }

        #file-select:hover {
            background-color: #e0e0e0;
        }

        #file-label {
            font-weight: bold;
        }

        .dark-mode #file-select {
            background-color: #3a3a3a;
            color: #ffffff;
            border-color: #555;
        }

        .dark-mode #file-select:hover {
            background-color: #555;
        }


        #chat-container {
            display: none;
            margin-top: 20px;
            max-width: 100%;
        }

        #chat-box {
            width: 100%;
            max-width: 800px;
            height: 60vh;
            border: 1px solid #cccccc;
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f0f0; /* Gray background */
            overflow-y: auto;
            margin-top: 20px;
            transition: background-color 0.3s, border-color 0.3s;
            font-family: 'Avenir', sans-serif;
        }

        .dark-mode #chat-box {
            background-color: #2c2c2c;
            border-color: #444;
        }

        #user-input {
            width: 100%;
            max-width: 800px;
            padding: 10px;
            margin-top: 10px;
            font-size: 16px;
            border: 1px solid #cccccc;
            border-radius: 5px;
            background-color: #f0f0f0; /* Gray background */
            color: #333;
            transition: background-color 0.3s, border-color 0.3s, color 0.3s;
            margin-bottom: 20px;
            font-family: 'Avenir', sans-serif;
        }

        .dark-mode #user-input {
            background-color: #3a3a3a;
            border-color: #555;
            color: #ffffff;
        }

        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            line-height: 1.3;
            font-family: 'Avenir', sans-serif;
        }

        .user-message {
            background-color: #e0e0e0;
            text-align: right;
            width: 66%;
            margin-left: auto;
            font-family: 'Avenir', sans-serif;
        }

        .dark-mode .user-message {
            background-color: #4a4a4a;
        }

        .bot-message {
            background-color: #d1e7dd;
            text-align: left;
            font-family: 'Avenir', sans-serif;
        }

        .dark-mode .bot-message {
            background-color: #005d54;
        }

        @media (max-width: 900px) {
            body {
                flex-direction: column;
                align-items: center;
            }

            .left-side,
            .right-side {
                width: 100%;
                text-align: center;
            }

            .file-buttons {
                justify-content: center;
            }

            #chat-box,
            #user-input {
                max-width: 100%;
            }

        }
    </style>
</head>
<body>

    <div class="left-side">
        <img src="{{ url_for('static', filename='img/background.png') }}" alt="Course Logo">
        <h1>Edinburgh Infectious Diseases Chatbot</h1>
        <p>Select a file to get started! The Chatbot will answer your 
questions based on a curated library of literature. Note that your chat 
will be lost if you navigate away from the page, refresh the page, or 
click on a different file. </p>
        <button id="dark-mode-toggle" onclick="toggleDarkMode()">Toggle Dark Mode</button>
    </div>

    <div class="right-side">
        <!-- file Dropdown Menu -->
        <label for="file-select" id="file-label">Select file: </label>
        <select id="file-select" onchange="selectfileDropdown()">
	    <option value="">Select a file</option>
            <option value="1">Basic<option>
            <option value="2">Advanced</option>
        </select>

<!-- Placeholder for displaying the selected file's title -->
<div id="file-title" style="margin-top: 20px; font-size: 1.2rem; font-weight: bold;"></div>


        <div id="chat-container" style="display: block;">
            <div id="chat-box"></div>
            <input type="text" id="user-input" placeholder="Type your question here..." onkeydown="if(event.key === 'Enter') sendMessage()">
        </div>
    </div>

    <script>    let chatHistory = [];
    let selectedfile = null;

    // Array of file titles
    const fileTitles = [
        "Basic",
        "Advanced",
    ];

    function toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
    }

    function selectfileDropdown() {
    const fileDropdown = document.getElementById('file-select');
    selectedfile = fileDropdown.value;  // Get the selected prompt style
    console.log("Selected style: ", selectedfile);

    document.getElementById('chat-container').style.display = 'block';
    document.getElementById('chat-box').innerHTML = '';  // Clear previous chat
    chatHistory = [];  // Clear chat history when selecting a new style
    document.getElementById('user-input').focus();

    // Display the name of the selected prompt style
    const fileTitleElement = document.getElementById('file-title');
    const promptNames = {
        '1': "Basic (Child-friendly with gaming terms)",
        '2': "Advanced (Scientific expert mode)"
    };

    // Update the title display
    fileTitleElement.innerText = `Style: ${promptNames[selectedfile]}`;
}


    function sendMessage() {
        const inputBox = document.getElementById('user-input');
        const userMessage = inputBox.value.trim();

        if (userMessage === '') return;

	if (selectedfile === null) {
        	alert('Please select a file before sending a message.');
       	 return;
    	}

    
        console.log("Sending query for file: ", selectedfile);  // Verify this before sending

        const chatBox = document.getElementById('chat-box');

        const processedMessage = userMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Display user's message
        const userMsgElement = document.createElement('div');
        userMsgElement.className = 'message user-message';
        userMsgElement.innerText = userMessage;
        chatBox.appendChild(userMsgElement);

        // Add to chat history
        chatHistory.push({role: "user", content: userMessage});

        inputBox.value = '';

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;

        // Add loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.className = 'message bot-message';
        loadingElement.innerText = 'Loading...';
        chatBox.appendChild(loadingElement);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Send the message along with the chat history and selected file to the backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                history: chatHistory, 
                message: userMessage, 
                file: String(selectedfile) // Send selected file
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            chatBox.removeChild(loadingElement);

            if (data.response) {
                let botMessage = data.response
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold text
                    .replace(/\n/g, '<br>'); // Preserve line breaks

                const botMsgElement = document.createElement('div');
                botMsgElement.className = 'message bot-message';
                botMsgElement.innerHTML = botMessage;
                chatBox.appendChild(botMsgElement);

                // Add bot response to chat history
                chatHistory.push({role: "assistant", content: data.response});
            } else if (data.error) {
                const errorMsgElement = document.createElement('div');
                errorMsgElement.className = 'message bot-message';
                errorMsgElement.innerText = 'An error occurred: ' + data.error + '. Please try again.';
                chatBox.appendChild(errorMsgElement);
            }

            // Scroll to the bottom of the chat box
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => {
            // Remove loading indicator
            chatBox.removeChild(loadingElement);

            // Display error message in chatbox
            const errorMsgElement = document.createElement('div');
            errorMsgElement.className = 'message bot-message';
            errorMsgElement.innerText = 'An error occurred: ' + error + '. Please try again.';
            chatBox.appendChild(errorMsgElement);

            // Scroll to the bottom of the chat box
            chatBox.scrollTop = chatBox.scrollHeight;

            console.error('Error:', error);
        });
    }
</script>


</body></html>