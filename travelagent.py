import os
from typing import Dict, List, Tuple, Any, Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

useOpenAI = False  # Set to False to use NVIDIA

# Initialize model and embeddings based on the chosen provider
if useOpenAI:
    # Set up environment variables
    try:
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
    except Exception as e:
        print(f"Error retrieving API keys. {e}")
    llm = ChatOpenAI(model="gpt-4o", streaming=True)
    embeddings = OpenAIEmbeddings()
else:
    # Set up environment variables
    try:
        NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
        TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
    except Exception as e:
        print(f"Error retrieving API keys. {e}")
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", api_key=NVIDIA_API_KEY)
    embeddings = NVIDIAEmbeddings(api_key=NVIDIA_API_KEY)

# Initialize vector store
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Define Tavily search tool
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)

# Define tools
tools = [
    Tool(
        name="Visa Information Search",
        func=lambda q: vectorstore.similarity_search(q),
        description="Search for visa information in the local database"
    ),
    Tool(
        name="Tavily Search",
        func=lambda q: tavily_tool.run(q),
        description="Search for up-to-date visa requirements and travel information online including wikipedia"
    ),
]

# Create agent
agent_prompt = PromptTemplate.from_template(
    """You are a helpful visa and travel information assistant. Use the following tools to answer the user's question:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. If the question involves visas, search the internet for visa requirements for their passport country for traveling to the destination country.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the initial answer, then my final answer to the original input question and why it is different from the initial answer given my thought process. explain thought process. list URLs referenced. 

Begin!

Question: {input}
{agent_scratchpad}"""
)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Define state
class AgentState(TypedDict):
    messages: List[HumanMessage]
    user_message: str
    response: str
    thoughts: str
    initial_response: str

# Define LangGraph nodes
def user_input(state: AgentState) -> AgentState:
    return state

def generate_response(state: AgentState) -> AgentState:
    thoughts = []
    
    class ThoughtCapturingHandler(StreamingStdOutCallbackHandler):
        def on_llm_start(self, *args, **kwargs):
            pass
        
        def on_llm_end(self, *args, **kwargs):
            pass
        
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            thoughts.append(token)
    
    callback_manager = CallbackManager([ThoughtCapturingHandler()])
    
    with get_openai_callback() as cb:
        response = agent_executor.invoke(
            {"input": state["user_message"]},
            callbacks=[callback_manager]
        )
    
    state["response"] = response["output"]
    state["thoughts"] = "".join(thoughts)
    
    # Generate initial response
    initial_response = llm.predict(f"Given the question '{state['user_message']}', provide a brief initial response before conducting a thorough search.")
    state["initial_response"] = initial_response
    
    return state

# Create StateGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("user_input", user_input)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("user_input", "generate_response")
workflow.add_edge("generate_response", END)

# Set entry point
workflow.set_entry_point("user_input")

# Compile the graph
app = workflow.compile()

# Function to handle user queries
def handle_query(query: str) -> Tuple[str, str, str]:
    result = app.invoke({"user_message": query, "messages": [], "thoughts": "", "initial_response": ""})
    return result["initial_response"], result["response"], result["thoughts"]

# Gradio interface
def gradio_interface(query: str) -> Tuple[str, str, str]:
    initial_response, response, thoughts = handle_query(query)
    return initial_response, response, thoughts

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your visa or travel question here..."),
    outputs=[
        gr.Textbox(label="Initial Response"),
        gr.Textbox(label="Final Response"),
    ],
    title="Visa and Travel Information Assistant",
    description="Ask questions about visa requirements and travel information.",
examples = [
    ["What visa do I need to visit the USA with an Indian passport?"],
    ["I have an Indian passport and USA tourist visa. What visa do I need to travel to Philippines?"],
    ["Do I need a visa for Japan if traveling with an Australian passport"],
    ["What are the visa options for digital nomads in Portugal?"],
    ["What are the entry requirements for Brazil for Indian passport holders?"],
    ["Are there any safety alerts for France?"],
    ["How long can I stay in the USA with a B1/B2 visa?"],
    ["What's the weather like in Tokyo?"],
    ["What is the visa policy for New Zealand for Indian tourists?"],
    ["Do I need a visa for Japan if traveling with an Australian passport?"],
    ["Tell me about Schengen visas."],
    ["Can I travel to Canada with a Schengen visa?"],
    ["What is the process to get a UK student visa?"],
    ["Do I need a transit visa for Germany if I have a connecting flight?"],
    ["What are the visa requirements for an Indian citizen visiting Dubai?"],
    ["What documents do I need to apply for a Schengen visa?"],
    ["Is a visa required for Singapore for a US citizen?"],
    ["Can I enter multiple countries with a Schengen visa?"],
    ["What are the COVID-19 travel restrictions for Italy?"],
    ["How do I extend my stay in the USA with a tourist visa?"],
    ["Can I apply for a Canadian visa online?"],
    ["What is the processing time for an Australian student visa?"],
    ["Do I need a visa to visit Thailand with a UK passport?"],
    ["What is the visa on arrival policy for Indians in Indonesia?"],
    ["How can I check the status of my US visa application?"],
    ["Do I need a visa to visit South Korea with a UK passport?"]
]
)

if __name__ == "__main__":
    # Launch Gradio interface
    iface.launch()