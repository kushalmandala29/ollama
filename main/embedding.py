# from langchain_community.llms import Ollama
from langchain_ollama import  ChatOllama, OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
import torch
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
import json
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import logging
import os
import jq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from elasticsearch import Elasticsearch
# from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.vectorstores.elastic_vector_search import ElasticKNNSearch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logging.basicConfig(level=logging.DEBUG)  # Enable verbose logging
logger = logging.getLogger(__name__)
os.environ['USER_AGENT'] = 'myagent'
llm = ChatOllama(
    model="tinyllama",
    temperature=0,
    device=device
    # other params...
)
class CustomJSONLoader(JSONLoader):
    def __init__(self, file_path, jq_schema=".", **kwargs):
        super().__init__(file_path,jq_schema, **kwargs)
        self.jq_schema = jq_schema

    def load_and_split(self):
        """Load and split the JSON file."""
        with open(self.file_path, 'r') as file:
            json_text = file.read()

        # Apply jq_schema if it is provided
        if self.jq_schema:
            try:
                # Using the `jq` library to apply the jq schema
                jq_filter = jq.compile(self.jq_schema)
                json_text = jq_filter.input(json.loads(json_text)).text()
            except Exception as e:
                raise ValueError(f"Error applying jq filter: {e}")

        # Continue with the regular JSONLoader processing
        json_data = json.loads(json_text)
        docs = []
        for item in json_data:
            if isinstance(item, dict):
                content = json.dumps(item, ensure_ascii=False)
            elif isinstance(item, str):
                content = item
            else:
                content = str(item)
        # doc = Document(page_content=content, metadata={})
        docs.append(content)
        return docs
        
def load_documents(directory: str) -> str:
    """Load documents from the specified directory."""
    loader = DirectoryLoader(
        directory,
        glob="*.json",
        loader_cls=CustomJSONLoader,
        loader_kwargs={'jq_schema': '.','text_content': False}
    )
    logger.debug("Loading data from directory...")
    data=loader.load()
    # print(data)
    # new_data=[Document(page_content=doc.page_content,metadata={}) for doc in data]
    logger.debug(f"Loaded {len(data)} documents.")
    return data

def split_documents(data:list)->list[Document]:
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    split_documents=text_spliter.split_documents(data)
    logger.debug(f'split into {len(split_documents)} chunks')
    return split_documents



def create_vectorstore(splits:list, model_name:str, persist_directory:str)->Chroma:
    '''create a vector for the document splits'''
    embedding_function=OllamaEmbeddings(model="tinyllama")
    try:
        create_vectorstore=Chroma.from_documents(documents=splits,embedding=embedding_function,persist_directory=persist_directory)
        logger.debug('stored in chromadb')
        return create_vectorstore
    except Exception as e:
        logger.error(f"Failed to store vectors in ChromaDB: {e}")



data=load_documents('main/dataset')
# pp=[doc.page_content for doc in data]
# print(pp)
split_documents=split_documents(data)
# print(split_documents.page_content)
vectorstore=create_vectorstore(split_documents,'tinyllama','vector_db')

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates English language to hindi language.",
#         ),
#         ("human", "{input}"),
#     ]
# )

# chain = prompt | llm
# chain.invoke(
#     {
#         # "input_language": "English",
#         # "output_language": "German",
#         "input": "I love programming, now translate this into hindi.",
#     }
# )