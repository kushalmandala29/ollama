import os
import json
import torch
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import  ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Set the environment variable for user-agent
os.environ['USER_AGENT'] = 'myagent'

# Custom JSONLoader to convert dict to string
class CustomJSONLoader(JSONLoader):
    def _get_text(self, sample):
        if isinstance(sample, dict):
            return json.dumps(sample)
        return super()._get_text(sample)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize directory loader
loader = DirectoryLoader(
    './data',
    glob="*.json",
    loader_cls=CustomJSONLoader,
    loader_kwargs={'jq_schema': '.'}
)

# Load the data
data = loader.load()

# Initialize text splitter and split the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
splits = text_splitter.split_documents(data)

# Initialize embedding function with GPU
embedding_function = OllamaEmbeddings(
    model="llama3",
    #device=device  # Ensure the model uses the GPU
)

# Create Chroma vectorstore
persist_directory = 'db'
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=persist_directory)
retriever = vectorstore.as_retriever()

# Initialize LLM with GPU
llm = ChatOllama(model="llama3", device=device)

# Create history-aware retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Set up the question-answer chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_retrieval_chain(llm, qa_prompt)

# Create RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Manage chat history statefully
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Example invocation of the chain
# Define the input dictionary as required
input_data = {
    "input": "What are common ways of doing it?"
}

# Configuration dictionary
config_dict = {
    "configurable": {
        "session_id": "abc123"
    }
}

# Invoke the chain with the specified input and configuration
result = conversational_rag_chain.invoke(
    input_data,      # Provide the input data as a dictionary
    config=config_dict  # Configuration dictionary
)["answer"]

# Print the result
# print(result
# Print the result
print(result)
