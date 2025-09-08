package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// Base URL for the Ollama API
const ollamaBaseURL = "http://localhost:11434"
const ollamaGenerateAPI = ollamaBaseURL + "/api/generate"
const ollamaChatAPI = ollamaBaseURL + "/api/chat"
const ollamaTagsAPI = ollamaBaseURL + "/api/tags"
const ollamaPullAPI = ollamaBaseURL + "/api/pull"
const ollamaDeleteAPI = ollamaBaseURL + "/api/delete"

// --- API Request/Response Structures ---

// OllamaGenerateRequestPayload for /api/generate
type OllamaGenerateRequestPayload struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// OllamaChatRequestPayload for /api/chat
type OllamaChatRequestPayload struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

// Message structure for chat API
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OllamaModelActionPayload for /api/pull and /api/delete
type OllamaModelActionPayload struct {
	Model string `json:"name"` // Ollama uses 'name' for model actions
}

// OllamaResponseChunk for streaming responses (generate and chat)
type OllamaResponseChunk struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"` // For generate API
	Message   *Message `json:"message"`  // For chat API
	Done      bool   `json:"done"`
}

// ClientRequest from frontend to Go backend
type ClientRequest struct {
	ActionType string    `json:"actionType"` // "generate", "chat", "pull", "delete"
	Model      string    `json:"model"`
	Prompt     string    `json:"prompt"`     // For generate API
	Messages   []Message `json:"messages"` // For chat API
}

// OllamaModel represents a single model returned by the /api/tags endpoint.
type OllamaModel struct {
	Name string `json:"name"`
}

// OllamaTagsResponse defines the structure of the JSON response from the /api/tags endpoint.
type OllamaTagsResponse struct {
	Models []OllamaModel `json:"models"`
}

// --- Main Server Logic ---

func main() {
	http.HandleFunc("/", serveHTML)
	http.HandleFunc("/api/ollama-action", handleOllamaAction) // Unified endpoint for all actions
	http.HandleFunc("/api/models", handleListModels)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on http://localhost:%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// serveHTML serves the main HTML page for the web UI.
func serveHTML(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama Go Web UI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
        }
        .container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        textarea {
            resize: vertical;
            min-height: 120px;
        }
        button {
            transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        }
        button:hover {
            transform: translateY(-1px);
        }
        button:active {
            transform: translateY(0);
        }
        #loading-indicator {
            display: none;
            color: #4f46e5;
            font-weight: 500;
            margin-top: 1rem;
        }
        #custom-alert-modal {
            z-index: 1000;
        }
        .chat-message {
            margin-bottom: 0.75rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            max-width: 80%;
            word-wrap: break-word; /* Ensure long words wrap */
        }
        .chat-message.user {
            background-color: #e0e7ff; /* Indigo-100 */
            text-align: right;
            margin-left: auto;
        }
        .chat-message.assistant {
            background-color: #e5e7eb; /* Gray-200 */
            text-align: left;
            margin-right: auto;
        }
        .api-section {
            border: 1px solid #e5e7eb; /* Light gray border */
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: #f9fafb; /* Lighter gray background */
        }
        .api-section h2 {
            font-size: 1.25rem; /* text-xl */
            font-weight: 600; /* font-semibold */
            color: #374151; /* gray-700 */
            margin-bottom: 1rem;
        }
        #thinking-output {
            background-color: #fffbeb; /* Amber-50 */
            border: 1px dashed #fcd34d; /* Amber-300 */
            padding: 0.75rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-style: italic;
            color: #d97706; /* Amber-700 */
            max-height: 100px;
            overflow-y: auto;
            word-wrap: break-word;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="container w-full">
        <h1 class="text-4xl font-extrabold text-center text-gray-900 mb-4">Ollama Go Web UI</h1>
        <p class="text-center text-gray-600 mb-8">Interact with your local Ollama instance for text generation, chat, and model management.</p>
        <p class="text-center text-gray-500 text-sm mb-8">Make sure Ollama is running on <code class="bg-gray-200 px-1 py-0.5 rounded">http://localhost:11434</code> and you have downloaded models (e.g., <code class="bg-gray-200 px-1 py-0.5 rounded">ollama pull llama2</code>).</p>


        <div class="mb-6">
            <label for="api-type-select" class="block text-gray-700 text-sm font-medium mb-2">Select API Type:</label>
            <select id="api-type-select" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent">
                <option value="generate">Generate Text</option>
                <option value="chat">Chat</option>
                <option value="model-management">Model Management</option>
            </select>
        </div>

        <div class="mb-6" id="common-model-select-container">
            <label for="model-select" class="block text-gray-700 text-sm font-medium mb-2">Choose Ollama Model:</label>
            <select id="model-select" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent">
                <option value="">Loading models...</option>
            </select>
        </div>

        <!-- Generate Text Section -->
        <div id="generate-section" class="api-section">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Generate Text</h2>
            <div class="mb-6">
                <label for="prompt-input" class="block text-gray-700 text-sm font-medium mb-2">Prompt:</label>
                <textarea id="prompt-input" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="Enter your prompt here..."></textarea>
            </div>
            <button id="generate-button" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                Generate Response
            </button>
        </div>

        <!-- Chat Section -->
        <div id="chat-section" class="api-section hidden">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Chat with Model</h2>
            <div id="chat-history-output" class="bg-gray-50 p-4 rounded-lg border border-gray-200 mb-4 h-64 overflow-y-auto flex flex-col space-y-2">
                <!-- Chat messages will be appended here -->
            </div>
            <div class="mb-4">
                <input type="checkbox" id="show-thinking-checkbox" class="mr-2">
                <label for="show-thinking-checkbox" class="text-gray-700 text-sm font-medium">Display Thinking Process</label>
            </div>
            <div id="thinking-output" class="hidden text-sm mb-4">
                <!-- Thinking process will be streamed here -->
            </div>
            <div class="mb-6">
                <label for="chat-input" class="block text-gray-700 text-sm font-medium mb-2">Your Message:</label>
                <textarea id="chat-input" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="Type your message..."></textarea>
            </div>
            <button id="send-chat-button" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                Send Message
            </button>
        </div>

        <!-- Model Management Section -->
        <div id="model-management-section" class="api-section hidden">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Model Management</h2>
            <div class="mb-4">
                <label for="model-action-select" class="block text-gray-700 text-sm font-medium mb-2">Select Installed Model for Action:</label>
                <select id="model-action-select" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent">
                    <option value="">No models loaded</option>
                </select>
                <button id="refresh-models-button" class="mt-2 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                    Refresh Installed Models List
                </button>
            </div>

            <div class="mb-4">
                <label for="available-model-select" class="block text-gray-700 text-sm font-medium mb-2">Select Model to Install (from Ollama Registry):</label>
                <select id="available-model-select" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent">
                    <option value="">Loading available models...</option>
                </select>
                <div id="available-model-description" class="mt-2 p-3 bg-gray-100 border border-gray-200 rounded-lg text-sm text-gray-600 hidden"></div>
                <button id="pull-available-model-button" class="mt-2 w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2">
                    Pull Selected Model
                </button>
            </div>

            <div class="mb-4">
                <label for="model-action-input" class="block text-gray-700 text-sm font-medium mb-2">Or, Enter Model Name Manually (for Pull/Delete):</label>
                <input type="text" id="model-action-input" class="shadow-sm appearance-none border rounded-lg w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="e.g., new-model:latest or llama2:7b-chat">
            </div>
            <div class="flex space-x-4">
                <button id="pull-manual-model-button" class="flex-1 bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2">
                    Pull Manual Model
                </button>
                <button id="delete-model-button" class="flex-1 bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2">
                    Delete Model
                </button>
            </div>
            <div id="model-action-output" class="mt-4 bg-gray-50 p-4 rounded-lg border border-gray-200 whitespace-pre-wrap text-gray-700 text-base"></div>
        </div>

        <div id="loading-indicator" class="text-center mt-4 text-indigo-600 font-semibold">
            Generating... Please wait.
        </div>

        <!-- Unified Response Output for Generate/Chat -->
        <div id="unified-response-output" class="mt-8 bg-gray-50 p-6 rounded-lg border border-gray-200">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Response:</h2>
            <div id="response-output" class="whitespace-pre-wrap text-gray-700 text-base"></div>
        </div>
    </div>

    <div id="custom-alert-modal" class="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center hidden">
        <div class="bg-white p-6 rounded-lg shadow-xl max-w-sm w-full">
            <h3 id="custom-alert-title" class="text-lg font-semibold text-gray-800 mb-4">Alert</h3>
            <p id="custom-alert-message" class="text-gray-700 mb-6"></p>
            <div class="flex justify-end space-x-4">
                <button id="custom-alert-cancel" class="hidden bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2">
                    Cancel
                </button>
                <button id="custom-alert-ok" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
                    OK
                </button>
            </div>
        </div>
    </div>

    <script>
        const apiTypeSelect = document.getElementById('api-type-select');
        const modelSelect = document.getElementById('model-select');
        const promptInput = document.getElementById('prompt-input');
        const generateButton = document.getElementById('generate-button');
        const responseOutput = document.getElementById('response-output');
        const loadingIndicator = document.getElementById('loading-indicator');

        const generateSection = document.getElementById('generate-section');
        const chatSection = document.getElementById('chat-section');
        const modelManagementSection = document.getElementById('model-management-section');

        const chatInput = document.getElementById('chat-input');
        const sendChatButton = document.getElementById('send-chat-button');
        const chatHistoryOutput = document.getElementById('chat-history-output');
        const showThinkingCheckbox = document.getElementById('show-thinking-checkbox'); // New element
        const thinkingOutput = document.getElementById('thinking-output'); // New element

        const modelActionSelect = document.getElementById('model-action-select');
        const refreshModelsButton = document.getElementById('refresh-models-button');
        const availableModelSelect = document.getElementById('available-model-select');
        const availableModelDescription = document.getElementById('available-model-description');
        const pullAvailableModelButton = document.getElementById('pull-available-model-button');
        const modelActionInput = document.getElementById('model-action-input');
        const pullManualModelButton = document.getElementById('pull-manual-model-button');
        const deleteModelButton = document.getElementById('delete-model-button');
        const modelActionOutput = document.getElementById('model-action-output');
        const unifiedResponseOutput = document.getElementById('unified-response-output');
        const commonModelSelectContainer = document.getElementById('common-model-select-container');

        const customAlertModal = document.getElementById('custom-alert-modal');
        const customAlertTitle = document.getElementById('custom-alert-title');
        const customAlertMessage = document.getElementById('custom-alert-message');
        const customAlertOkButton = document.getElementById('custom-alert-ok');
        const customAlertCancelButton = document.getElementById('custom-alert-cancel');

        let resolveAlertPromise;

        function showAlert(message, title = "Alert") {
            customAlertTitle.textContent = title;
            customAlertMessage.textContent = message;
            customAlertCancelButton.classList.add('hidden');
            customAlertOkButton.textContent = 'OK';
            customAlertModal.classList.remove('hidden');
            return new Promise(resolve => {
                resolveAlertPromise = resolve;
            });
        }

        function showConfirm(message, title = "Confirm") {
            customAlertTitle.textContent = title;
            customAlertMessage.textContent = message;
            customAlertCancelButton.classList.remove('hidden');
            customAlertOkButton.textContent = 'Confirm';
            customAlertModal.classList.remove('hidden');
            return new Promise(resolve => {
                resolveAlertPromise = resolve;
            });
        }

        customAlertOkButton.addEventListener('click', () => {
            customAlertModal.classList.add('hidden');
            if (resolveAlertPromise) {
                resolveAlertPromise(true);
            }
        });

        customAlertCancelButton.addEventListener('click', () => {
            customAlertModal.classList.add('hidden');
            if (resolveAlertPromise) {
                resolveAlertPromise(false);
            }
        });

        let chatMessages = [];

        // Hardcoded list of common Ollama models with descriptions for "available to install"
        const availableModels = [
            { name: "llama2", description: "A powerful open-source large language model from Meta." },
            { name: "mistral", description: "A small, yet powerful, language model from Mistral AI, optimized for performance." },
            { name: "gemma", description: "Lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models." },
            { name: "phi", description: "A small language model from Microsoft, ideal for research and experimentation." },
            { name: "codellama", description: "A family of large language models from Meta designed for code generation and understanding." },
            { name: "neural-chat", description: "Fine-tuned for engaging conversational AI experiences." },
            { name: "dolphin-phi", description: "A fine-tuned version of Phi-2, designed for helpful and harmless chat." },
            { name: "openhermes", description: "A powerful model trained on a diverse range of datasets for general conversational tasks." },
            { name: "tinyllama", description: "A compact language model, great for resource-constrained environments or quick experiments." },
            { name: "vicuna", description: "A chatbot trained by fine-tuning LLaMA on user-shared conversations." },
            { name: "wizardlm", description: "An instruction-following LLM, based on LLaMA, fine-tuned with a large amount of instruction data." },
            { name: "zephyr", description: "A series of language models that are fine-tuned versions of Mistral, optimized for helpfulness." },
            { name: "stable-beluga", description: "A powerful instruction-tuned model, based on Llama 2, known for strong performance." },
            { name: "orca-mini", description: "A smaller, fine-tuned version of Orca, designed for efficient performance on various tasks." },
            { name: "medllama2", description: "A medical domain-specific version of Llama 2, useful for healthcare-related text generation." },
            { name: "nous-hermes2", description: "A strong conversational model, part of the Nous Research efforts." }
        ];

        async function fetchAndPopulateModels() {
            try {
                const response = await fetch('/api/models');
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error("HTTP error! status: " + response.status + ", message: " + errorText);
                }
                const data = await response.json();
                
                modelSelect.innerHTML = ''; 
                modelActionSelect.innerHTML = '';

                if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = model.name;
                        modelSelect.appendChild(option);

                        const actionOption = document.createElement('option');
                        actionOption.value = model.name;
                        actionOption.textContent = model.name;
                        modelActionSelect.appendChild(actionOption);
                    });
                    if (Array.from(modelSelect.options).some(option => option.value === 'llama2')) {
                        modelSelect.value = 'llama2';
                    } else {
                        modelSelect.selectedIndex = 0;
                    }
                    if (Array.from(modelActionSelect.options).some(option => option.value === 'llama2')) {
                        modelActionSelect.value = 'llama2';
                    } else {
                        modelActionSelect.selectedIndex = 0;
                    }
                    modelSelect.disabled = false;
                    modelActionSelect.disabled = false;
                    generateButton.disabled = false;
                    sendChatButton.disabled = false;
                    pullManualModelButton.disabled = false;
                    deleteModelButton.disabled = false;
                } else {
                    const option = document.createElement('option');
                    option.value = "";
                    option.textContent = "No models found. Run 'ollama pull <model_name>'";
                    modelSelect.appendChild(option);
                    modelActionSelect.appendChild(option.cloneNode(true));
                    
                    modelSelect.disabled = true;
                    modelActionSelect.disabled = true;
                    generateButton.disabled = true;
                    sendChatButton.disabled = true;
                    pullManualModelButton.disabled = true;
                    deleteModelButton.disabled = true;
                    showAlert("No Ollama models found. Please ensure Ollama is running and you have downloaded models (e.g., 'ollama pull llama2').");
                }

            } catch (error) {
                console.error('Error fetching models:', error);
                modelSelect.innerHTML = '<option value="">Error loading models</option>';
                modelActionSelect.innerHTML = '<option value="">Error loading models</option>';
                modelSelect.disabled = true;
                modelActionSelect.disabled = true;
                generateButton.disabled = true;
                sendChatButton.disabled = true;
                pullManualModelButton.disabled = true;
                deleteModelButton.disabled = true;
                let userMessage = 'Failed to load Ollama models. Please ensure Ollama is running on http://localhost:11434. Error: ' + error.message;
                showAlert(userMessage);
            }
        }

        // Function to populate the "Available Models to Install" dropdown
        function populateAvailableModels() {
            availableModelSelect.innerHTML = ''; // Clear existing options
            if (availableModels.length > 0) {
                availableModels.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name; // Display only name in dropdown
                    availableModelSelect.appendChild(option);
                });
                availableModelSelect.disabled = false;
                pullAvailableModelButton.disabled = false;
                // Trigger change to display initial description
                availableModelSelect.dispatchEvent(new Event('change')); 
            } else {
                const option = document.createElement('option');
                option.value = "";
                option.textContent = "No available models listed.";
                availableModelSelect.appendChild(option);
                availableModelSelect.disabled = true;
                pullAvailableModelButton.disabled = true;
                availableModelDescription.classList.add('hidden'); // Hide description if no models
            }
        }

        // Event listener for selecting an available model to display its description
        availableModelSelect.addEventListener('change', () => {
            const selectedModelName = availableModelSelect.value;
            const selectedModel = availableModels.find(model => model.name === selectedModelName);
            if (selectedModel && selectedModel.description) {
                availableModelDescription.textContent = selectedModel.description;
                availableModelDescription.classList.remove('hidden');
            } else {
                availableModelDescription.textContent = '';
                availableModelDescription.classList.add('hidden');
            }
        });


        function showSection(sectionId) {
            const sections = [generateSection, chatSection, modelManagementSection];
            sections.forEach(section => {
                if (section.id === sectionId) {
                    section.classList.remove('hidden');
                } else {
                    section.classList.add('hidden');
                }
            });

            if (sectionId === 'model-management-section') {
                commonModelSelectContainer.classList.add('hidden');
                unifiedResponseOutput.classList.add('hidden');
                populateAvailableModels(); // Populate available models when showing this section
            } else {
                commonModelSelectContainer.classList.remove('hidden');
                unifiedResponseOutput.classList.remove('hidden');
            }
        }

        apiTypeSelect.addEventListener('change', (event) => {
            const selectedType = event.target.value;
            showSection(selectedType + '-section');
            responseOutput.textContent = '';
            modelActionOutput.textContent = '';
            // Clear thinking output and hide it when switching sections
            thinkingOutput.textContent = '';
            thinkingOutput.classList.add('hidden');
            showThinkingCheckbox.checked = false; // Uncheck checkbox
            if (selectedType === 'chat') {
                chatMessages = [];
                chatHistoryOutput.innerHTML = '';
            }
        });

        document.addEventListener('DOMContentLoaded', () => {
            fetchAndPopulateModels();
            showSection(apiTypeSelect.value + '-section');
        });

        refreshModelsButton.addEventListener('click', fetchAndPopulateModels);

        generateButton.addEventListener('click', async () => {
            const prompt = promptInput.value.trim();
            const model = modelSelect.value;
            if (!prompt) { showAlert('Please enter a prompt.'); return; }
            if (!model) { showAlert('Please select an Ollama model.'); return; }

            responseOutput.textContent = '';
            loadingIndicator.style.display = 'block';
            generateButton.disabled = true;
            modelSelect.disabled = true;
            apiTypeSelect.disabled = true;

            try {
                const response = await fetch('/api/ollama-action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ actionType: 'generate', prompt, model }),
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error("HTTP error! status: " + response.status + ", message: " + errorText);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) { break; }
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.substring(5);
                            if (data === '[DONE]') { reader.cancel(); return; }
                            try {
                                const jsonChunk = JSON.parse(data);
                                if (jsonChunk.response) {
                                    responseOutput.textContent += jsonChunk.response;
                                }
                            } catch (e) { console.warn('Could not parse JSON chunk:', data, e); }
                        }
                    }
                }
                if (buffer.startsWith('data: ')) {
                    const data = buffer.substring(5);
                    if (data !== '[DONE]') {
                        try {
                            const jsonChunk = JSON.parse(data);
                            if (jsonChunk.response) { responseOutput.textContent += jsonChunk.response; }
                        } catch (e) { console.warn('Could not parse final JSON chunk:', data, e); }
                    }
                }

            } catch (error) {
                console.error('Error:', error);
                let userMessage = 'An unexpected error occurred: ' + error.message;
                if (error.message.includes("Could not connect to Ollama")) {
                    userMessage = "Could could not connect to Ollama. Please ensure Ollama is running on http://localhost:11434 and the model '" + model + "' is available (e.g., 'ollama run " + model + "').";
                } else if (error.message.includes("404")) {
                    userMessage = "Ollama API error: Model '" + model + "' not found. Please ensure the model is installed (e.g., 'ollama run " + model + "').";
                } else if (error.message.includes("400")) {
                    userMessage = "Ollama API error: Bad request. Check your prompt or model name.";
                } else if (error.message.includes("500")) {
                    userMessage = "Internal server error. Please check the Go application logs for details.";
                }
                showAlert(userMessage);
                responseOutput.textContent = userMessage;
            } finally {
                loadingIndicator.style.display = 'none';
                generateButton.disabled = false;
                modelSelect.disabled = false;
                apiTypeSelect.disabled = false;
            }
        });

        sendChatButton.addEventListener('click', async () => {
            const userMessageContent = chatInput.value.trim();
            const model = modelSelect.value;
            if (!userMessageContent) { showAlert('Please enter a message.'); return; }
            if (!model) { showAlert('Please select an Ollama model.'); return; }

            chatMessages.push({ role: "user", content: userMessageContent });
            appendChatMessage("user", userMessageContent);
            chatInput.value = '';

            // Clear thinking output and show it if checkbox is checked
            thinkingOutput.textContent = '';
            if (showThinkingCheckbox.checked) {
                thinkingOutput.classList.remove('hidden');
            } else {
                thinkingOutput.classList.add('hidden');
            }

            loadingIndicator.style.display = 'block';
            sendChatButton.disabled = true;
            modelSelect.disabled = true;
            apiTypeSelect.disabled = true;

            try {
                const response = await fetch('/api/ollama-action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ actionType: 'chat', messages: chatMessages, model }),
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error("HTTP error! status: " + response.status + ", message: " + errorText);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';
                let assistantResponseContent = '';

                // Create a temporary div for the assistant's final message (will be populated later)
                const assistantMessageDiv = document.createElement('div');
                assistantMessageDiv.classList.add('chat-message', 'assistant');
                chatHistoryOutput.appendChild(assistantMessageDiv);


                while (true) {
                    const { done, value } = await reader.read();
                    if (done) { break; }
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.substring(5);
                            if (data === '[DONE]') { reader.cancel(); break; }
                            try {
                                const jsonChunk = JSON.parse(data);
                                if (jsonChunk.message && jsonChunk.message.content) {
                                    assistantResponseContent += jsonChunk.message.content;
                                    // Update thinking output with streamed content
                                    if (showThinkingCheckbox.checked) {
                                        thinkingOutput.textContent += jsonChunk.message.content;
                                        thinkingOutput.scrollTop = thinkingOutput.scrollHeight; // Scroll thinking output
                                    }
                                }
                            } catch (e) { console.warn('Could not parse JSON chunk:', data, e); }
                        }
                    }
                }
                // After streaming, set the final content for the assistant's message
                assistantMessageDiv.textContent = assistantResponseContent;
                chatHistoryOutput.scrollTop = chatHistoryOutput.scrollHeight; // Scroll main chat history

                // Add the complete assistant response to chatMessages
                if (assistantResponseContent) {
                    chatMessages.push({ role: "assistant", content: assistantResponseContent });
                }

            } catch (error) {
                console.error('Error:', error);
                let userMessage = 'An unexpected error occurred during chat: ' + error.message;
                if (error.message.includes("Could not connect to Ollama")) {
                    userMessage = "Could not connect to Ollama. Please ensure Ollama is running on http://localhost:11434 and the model '" + model + "' is available (e.g., 'ollama run " + model + "').";
                } else if (error.message.includes("404")) {
                    userMessage = "Ollama API error: Model '" + model + "' not found. Please ensure the model is installed (e.g., 'ollama run " + model + "').";
                } else if (error.message.includes("400")) {
                    userMessage = "Ollama API error: Bad request. Check your message or model name.";
                } else if (error.message.includes("500")) {
                    userMessage = "Internal server error. Please check the Go application logs for details.";
                }
                showAlert(userMessage);
                appendChatMessage("error", userMessage);
            } finally {
                loadingIndicator.style.display = 'none';
                sendChatButton.disabled = false;
                modelSelect.disabled = false;
                apiTypeSelect.disabled = false;
                // Always clear and hide thinking output after response (or error)
                thinkingOutput.textContent = '';
                thinkingOutput.classList.add('hidden');
            }
        });

        // Event listener for the "Display Thinking Process" checkbox
        showThinkingCheckbox.addEventListener('change', () => {
            if (showThinkingCheckbox.checked) {
                // If already hidden and checked, show it (e.g., if streaming is ongoing)
                if (thinkingOutput.textContent !== '') { // Only show if there's content
                    thinkingOutput.classList.remove('hidden');
                }
            } else {
                thinkingOutput.classList.add('hidden');
            }
        });


        function appendChatMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('chat-message', role);
            messageDiv.textContent = content;
            chatHistoryOutput.appendChild(messageDiv);
            chatHistoryOutput.scrollTop = chatHistoryOutput.scrollHeight;
        }

        // Unified pull function for both manual input and dropdown selection
        async function performPullModel(modelName) {
            modelActionOutput.textContent = 'Pulling model ' + modelName + '... This may take a while.';
            loadingIndicator.style.display = 'block';
            pullManualModelButton.disabled = true;
            pullAvailableModelButton.disabled = true;
            deleteModelButton.disabled = true;
            modelSelect.disabled = true;
            apiTypeSelect.disabled = true;
            refreshModelsButton.disabled = true;
            availableModelSelect.disabled = true;

            try {
                const response = await fetch('/api/ollama-action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ actionType: 'pull', model: modelName }),
                });

                const result = await response.text();
                if (!response.ok) {
                    throw new Error("HTTP error! status: " + response.status + ", message: " + result);
                }
                modelActionOutput.textContent = 'Pull successful for ' + modelName + ':\n' + result;
                await fetchAndPopulateModels(); // Refresh installed models list
            } catch (error) {
                console.error('Error pulling model:', error);
                let userMessage = 'Failed to pull model ' + modelName + '. Error: ' + error.message;
                showAlert(userMessage);
                modelActionOutput.textContent = userMessage;
            } finally {
                loadingIndicator.style.display = 'none';
                pullManualModelButton.disabled = false;
                pullAvailableModelButton.disabled = false;
                deleteModelButton.disabled = false;
                modelSelect.disabled = false;
                apiTypeSelect.disabled = false;
                refreshModelsButton.disabled = false;
                availableModelSelect.disabled = false;
            }
        }

        // Event listener for pulling from the "Available Models" dropdown
        pullAvailableModelButton.addEventListener('click', async () => {
            const model = availableModelSelect.value;
            if (!model) {
                showAlert('Please select a model from the list to pull.');
                return;
            }
            performPullModel(model);
        });

        // Event listener for pulling from the manual input field
        pullManualModelButton.addEventListener('click', async () => {
            const model = modelActionInput.value.trim();
            if (!model) {
                showAlert('Please enter a model name in the manual input field.');
                return;
            }
            performPullModel(model);
        });


        deleteModelButton.addEventListener('click', async () => {
            let model = modelActionInput.value.trim();
            if (!model) {
                model = modelActionSelect.value;
            }
            if (!model) { showAlert('Please enter or select a model name to delete.'); return; }

            const confirmed = await showConfirm('Are you sure you want to delete model: ' + model + '? This action cannot be undone.');
            if (!confirmed) {
                return;
            }

            modelActionOutput.textContent = 'Deleting model ' + model + '...';
            loadingIndicator.style.display = 'block';
            pullManualModelButton.disabled = true;
            pullAvailableModelButton.disabled = true; // Disable this too during delete
            deleteModelButton.disabled = true;
            modelSelect.disabled = true;
            apiTypeSelect.disabled = true;
            refreshModelsButton.disabled = true;
            availableModelSelect.disabled = true; // Disable this too during delete

            try {
                const response = await fetch('/api/ollama-action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ actionType: 'delete', model }),
                });

                const result = await response.text();
                if (!response.ok) {
                    throw new Error("HTTP error! status: " + response.status + ", message: " + result);
                }
                modelActionOutput.textContent = 'Delete successful for ' + model + ':\n' + result;
                await fetchAndPopulateModels();
            } catch (error) {
                console.error('Error deleting model:', error);
                let userMessage = 'Failed to delete model ' + model + '. Error: ' + error.message;
                showAlert(userMessage);
                modelActionOutput.textContent = userMessage;
            } finally {
                loadingIndicator.style.display = 'none';
                pullManualModelButton.disabled = false;
                pullAvailableModelButton.disabled = false; // Re-enable
                deleteModelButton.disabled = false;
                modelSelect.disabled = false;
                apiTypeSelect.disabled = false;
                refreshModelsButton.disabled = false;
                availableModelSelect.disabled = false; // Re-enable
            }
        });

    </script>
</body>
</html>
`)
}

// handleOllamaAction is a unified handler for all Ollama API interactions.
func handleOllamaAction(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var clientReq ClientRequest
	if err := json.NewDecoder(r.Body).Decode(&clientReq); err != nil {
		http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
		return
	}

	client := &http.Client{Timeout: 300 * time.Second} // Long timeout for LLM operations

	switch clientReq.ActionType {
	case "generate":
		callGenerateAPI(w, r, clientReq, client)
	case "chat":
		callChatAPI(w, r, clientReq, client)
	case "pull":
		callModelPullAPI(w, r, clientReq, client)
	case "delete":
		callModelDeleteAPI(w, r, clientReq, client)
	default:
		http.Error(w, "Unknown action type: "+clientReq.ActionType, http.StatusBadRequest)
	}
}

// callGenerateAPI handles the /api/generate endpoint
func callGenerateAPI(w http.ResponseWriter, r *http.Request, clientReq ClientRequest, client *http.Client) {
	ollamaReq := OllamaGenerateRequestPayload{
		Model:  clientReq.Model,
		Prompt: clientReq.Prompt,
		Stream: true,
	}
	payloadBytes, err := json.Marshal(ollamaReq)
	if err != nil {
		http.Error(w, "Error marshalling Ollama generate request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest(http.MethodPost, ollamaGenerateAPI, bytes.NewBuffer(payloadBytes))
	if err != nil {
		http.Error(w, "Error creating generate request to Ollama: "+err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error connecting to Ollama generate API: %v", err)
		http.Error(w, "Could not connect to Ollama. Please ensure Ollama is running on "+ollamaBaseURL+". "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("Ollama generate API returned non-200 status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		http.Error(w, fmt.Sprintf("Ollama API error: Status %d, Message: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes))), resp.StatusCode)
		return
	}

	// Set headers for Server-Sent Events (SSE)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Println("Streaming not supported by this connection for generate API.")
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk OllamaResponseChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			log.Printf("Error unmarshalling Ollama generate response chunk: %v, line: %s", err, line)
			continue
		}

		if chunk.Response != "" {
			fmt.Fprintf(w, "data: %s\n\n", line) // Send the full JSON chunk as data
			flusher.Flush()
		}

		if chunk.Done {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			break
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading Ollama generate response stream: %v", err)
	}
}

// callChatAPI handles the /api/chat endpoint
func callChatAPI(w http.ResponseWriter, r *http.Request, clientReq ClientRequest, client *http.Client) {
	ollamaReq := OllamaChatRequestPayload{
		Model:    clientReq.Model,
		Messages: clientReq.Messages,
		Stream:   true,
	}
	payloadBytes, err := json.Marshal(ollamaReq)
	if err != nil {
		http.Error(w, "Error marshalling Ollama chat request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest(http.MethodPost, ollamaChatAPI, bytes.NewBuffer(payloadBytes))
	if err != nil {
		http.Error(w, "Error creating chat request to Ollama: "+err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error connecting to Ollama chat API: %v", err)
		http.Error(w, "Could not connect to Ollama. Please ensure Ollama is running on "+ollamaBaseURL+". "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("Ollama chat API returned non-200 status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		http.Error(w, fmt.Sprintf("Ollama API error: Status %d, Message: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes))), resp.StatusCode)
		return
	}

	// Set headers for Server-Sent Events (SSE)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Println("Streaming not supported by this connection for chat API.")
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk OllamaResponseChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			log.Printf("Error unmarshalling Ollama chat response chunk: %v, line: %s", err, line)
			continue
		}

		// For chat, we stream the 'message' content
		if chunk.Message != nil && chunk.Message.Content != "" {
			fmt.Fprintf(w, "data: %s\n\n", line) // Send the full JSON chunk as data
			flusher.Flush()
		}

		if chunk.Done {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			break
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading Ollama chat response stream: %v", err)
	}
}

// callModelPullAPI handles the /api/pull endpoint
func callModelPullAPI(w http.ResponseWriter, r *http.Request, clientReq ClientRequest, client *http.Client) {
	ollamaReq := OllamaModelActionPayload{
		Model: clientReq.Model,
	}
	payloadBytes, err := json.Marshal(ollamaReq)
	if err != nil {
		http.Error(w, "Error marshalling Ollama pull request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest(http.MethodPost, ollamaPullAPI, bytes.NewBuffer(payloadBytes))
	if err != nil {
		http.Error(w, "Error creating pull request to Ollama: "+err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error connecting to Ollama pull API: %v", err)
		http.Error(w, "Could not connect to Ollama. Please ensure Ollama is running on "+ollamaBaseURL+". "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Error reading Ollama pull response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("Ollama pull API returned non-200 status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		http.Error(w, fmt.Sprintf("Ollama API error pulling model: Status %d, Message: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes))), resp.StatusCode)
		return
	}

	w.Header().Set("Content-Type", "text/plain") // Pull often returns plain text status
	w.Write(bodyBytes)
}

// callModelDeleteAPI handles the /api/delete endpoint
func callModelDeleteAPI(w http.ResponseWriter, r *http.Request, clientReq ClientRequest, client *http.Client) {
	ollamaReq := OllamaModelActionPayload{
		Model: clientReq.Model,
	}
	payloadBytes, err := json.Marshal(ollamaReq)
	if err != nil {
		http.Error(w, "Error marshalling Ollama delete request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// DELETE request for Ollama's /api/delete
	req, err := http.NewRequest(http.MethodDelete, ollamaDeleteAPI, bytes.NewBuffer(payloadBytes))
	if err != nil {
		http.Error(w, "Error creating delete request to Ollama: "+err.Error(), http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error connecting to Ollama delete API: %v", err)
		http.Error(w, "Could not connect to Ollama. Please ensure Ollama is running on "+ollamaBaseURL+". "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Error reading Ollama delete response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("Ollama delete API returned non-200 status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		http.Error(w, fmt.Sprintf("Ollama API error deleting model: Status %d, Message: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes))), resp.StatusCode)
		return
	}

	w.Header().Set("Content-Type", "text/plain") // Delete often returns plain text success
	w.Write(bodyBytes)
}

// handleListModels fetches the list of available Ollama models from the /api/tags endpoint.
func handleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	client := &http.Client{Timeout: 10 * time.Second} // Shorter timeout for listing models
	resp, err := client.Get(ollamaTagsAPI)
	if err != nil {
		log.Printf("Error connecting to Ollama tags API: %v", err)
		http.Error(w, "Could not connect to Ollama to list models. Please ensure Ollama is running on "+ollamaTagsAPI+".", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		log.Printf("Ollama tags API returned non-200 status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		http.Error(w, fmt.Sprintf("Ollama API error fetching models: Status %d, Message: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes))), resp.StatusCode)
		return
	}

	var tagsResponse OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResponse); err != nil {
		log.Printf("Error unmarshalling Ollama tags response: %v", err)
		http.Error(w, "Error parsing Ollama models response.", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tagsResponse)
}
