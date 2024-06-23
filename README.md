[Watch this video on YouTube](https://youtu.be/AWLNC8hsyzs)

<img src="images/thumbnail.jpg" alt="Thumbnail" width="50%">

# Visa and Travel Information Assistant

This repository contains a Gradio-based web application that serves as a Visa and Travel Information Assistant. It uses a combination of advanced technologies including NVIDIA API, LangChain agent, LangGraph, and Tavily search to provide accurate and up-to-date information about visa requirements and travel information.

## Technologies Used

- **LangChain Agent**: Utilizes `create_react_agent` for handling user interactions and queries.
- **LangGraph**: Manages the flow and logic of conversations.
- **Tavily Search**: Provides search capabilities for visa and travel information.
- **NVIDIA API**: Uses the LLaMA3-70B endpoint for processing and the nv-embed-qa endpoint for embeddings.

## Novelty of ReACT Agent

The ReACT agent (`create_react_agent`) is a key component in this application that enhances the interactivity and responsiveness of the assistant. 
ReACT (Reflective, Adaptive, and Contextual Technology) agents are designed to dynamically adapt their responses based on the user's input and the context of the conversation. 

This is novel in this application for several reasons:

- **Overcomes Hallucination**: ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a web search API (Tavily).
- **Human-like task-solving**: The ReACT agent simulates human-like problem-solving abilities by integrating various sources of information and providing thoughtful, contextually appropriate solutions.
- **Reflective**: The ReACT agent reflects on previous interactions to provide more accurate and contextually relevant responses.
- **Adaptive**: It adapts its behavior based on the user's queries, offering a personalized experience.
- **Contextual**: The agent maintains context throughout the conversation, ensuring continuity and coherence in its responses.
- **Explainable AI**: The agent explains why its final answer differed from its initial response. This is usually due to it retrieving more accurate information from a web search.

These features make the Visa and Travel Information Assistant more robust and user-friendly, providing users with precise and timely information tailored to their specific needs.

## Setup Instructions

### Get an NVIDIA API key:

1. Go to https://build.nvidia.com/explore/discover#llama3-70b and **Create a free account** with NVIDIA, which hosts NVIDIA AI Foundation models.
2. In the right pane you will see Python code with example usage of this endpoint. In the top right corner, click **Get API Key**. Then click **Generate Key**.
3. **Copy** the generated key in the .env file for `NVIDIA_API_KEY`.  (without any quotes)

### Get a TAVILY API key (1000 free credits, no payment method required):
Steps to Get a Free Tavily API Key

#### 1. Visit the Tavily Website
- Go to [Tavily](https://tavily.com)

#### 2. Sign Up for an Account
- Look for a "Sign Up" or "Get Started" button on the Tavily homepage
- Click on it to begin the registration process

#### 3. Choose the Free Plan
- Tavily offers a free plan for new creators and researchers
- This plan includes 1,000 API calls per month
- Select this option during the sign-up process

#### 4. Complete the Registration
- Fill out the required information to create your account or sign up with Google SSO.
- This typically includes your name, email address, and a password

#### 5. Verify Your Email
- Check your email inbox for a verification message from Tavily
- Click on the verification link to confirm your account

#### 6. Access Your API Key
- Once your account is verified and set up, log in to your dashboard
- Look for a section that provides your unique API key

#### 7. Copy and Secure Your API Key
- Copy the API key provided
- Paste it into the .env file line for 'TAVILY_API_KEY' (without any quotes)

Follow these steps to set up and run the Visa and Travel Information Agent Assistant:

1. **Clone the repository**
    ```bash
    git clone git@github.com:sunsetcoder/aitravelagent.git
    cd <repository-name>
    ```
2. **Set up a virtual environment (optional but recommended)**
    ```bash
    python -m venv .venv
    ```
3. **Activate the virtual environment**
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source .venv/bin/activate
      ```
4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
5. **Modify `.env` file**
    Replace `your_openai_api_key_here`, `your_tavily_api_key_here`, and `your_nvidia_api_key_here` with your actual API keys (without quotes).

    **Note:** The OpenAI API key is optional. If you use it, set `useOpenAI` to `True` in `app.py`.
6. **Run the application**
    ```bash
    python travelagent.py
    ```
    This will start the Gradio interface. Open the provided URL (typically http://127.0.0.1:7860) in your web browser to interact with the Visa and Travel Information Assistant.

## Usage

Once the Gradio interface is running:

1. Enter your visa or travel-related question in the text box, or click on one of the example questions.
2. Click the "Submit" button or press Enter.
3. The assistant will provide an answer based on the available information. It will provide it's initial thoughts, and its final thoughts. 

You can also try the example questions provided in the interface to see how the assistant works.

## Note

This application requires an active internet connection to function properly, as it relies on external APIs for information retrieval and processing.

## Python Version

This application was tested on Python 3.10.11.

---