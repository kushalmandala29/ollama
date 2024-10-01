from langchain_chroma import Chroma
from langchain_ollama import  ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional
import bs4
from typing import List, Tuple, Dict, Any
from langchain.schema import BaseRetriever, Document
from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import create_history_aware_retriever
from typing import List, Tuple
from langchain.schema import BaseRetriever, Document
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import torch
from pydantic import BaseModel, Field
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# from model import llm
embedding_function=OllamaEmbeddings(model="tinyllama")
db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_function,
)
query = "What is the capital of India?"
retriever = db.as_retriever(search_kwargs={"k": 4}) 
# chroma_retriever.get_relevant_documents(query)


llm=ChatOllama(model="tinyllama")





def create_custom_document_chain(llm, prompt_template=None):
    # Define a default prompt template if none is provided
    if prompt_template is None:
        system_template = """Use the following pieces of context and the chat history to answer the human's question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        
        Chat History:
        {chat_history}
        
        Human: {question}
        AI: """

        human_template = "{question}"

        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    else:
        # If a custom prompt_template is provided, ensure it's a ChatPromptTemplate
        if not isinstance(prompt_template, ChatPromptTemplate):
            raise ValueError("prompt_template must be a ChatPromptTemplate")
        chat_prompt = prompt_template

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=chat_prompt)

    # Create the StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    return stuff_chain

# Usage
def create_custom_retrival_chain(retrival, qa_chain):
    retrival_prompt=PromptTemplate(
        template="Given the following question, retrieve relevant information: {question}",
        input_variables=['question']   
    )
    retrival_chain=LLMChain(llm=qa_chain.llm_chain.llm, prompt=retrival_prompt)

    class CustomRetrivalChain:
        def __init__(self,retrival,qa_chain):
            self.retrival=retrival
            self.qa_chain=qa_chain  
        def run(self,question:str,chat_history:str=""):
            # Use the retrieval chain to potentially reformulate the question
            retrieval_result = retrival_chain.run(question=question)
            
            #Retrieve the documents
            docs=self.retrival.get_relevant_documents(retrieval_result)


            # use the QA chain to answer the question
            result=self.qa_chain.run(
                input_documents=docs,
                question=question,
                chat_history=chat_history
            
            )

            return result
    return CustomRetrivalChain(retrival,qa_chain)


class HistoryAwareRetriever:
    def __init__(self, base_retriever: BaseRetriever, llm, context_prompt: str):
        self.base_retriever = base_retriever
        self.llm = llm
        self.context_prompt = context_prompt

    def get_relevant_documents(self, query: str, chat_history: List[Tuple[str, str]] = None) -> List[Document]:
        if chat_history:
            # Format chat history
            formatted_history = "\n".join([f"Human: {h}\nAI: {a}" for h, a in chat_history])
            
            # Create a prompt that includes the chat history and the current query
            full_prompt = f"{self.context_prompt}\n\nChat History:\n{formatted_history}\n\nCurrent Question: {query}\n\nReformulated Question:"
            
            # Use the LLM to reformulate the query based on the chat history
            reformulated_query = self.llm(full_prompt).strip()
        else:
            reformulated_query = query

        # Use the base retriever to get relevant documents
        return self.base_retriever.get_relevant_documents(reformulated_query)

def create_custom_history_aware_retriever(
    llm, 
    retriever: BaseRetriever, 
    context_prompt: str
) -> HistoryAwareRetriever:
    return HistoryAwareRetriever(retriever, llm, context_prompt)




# llm.invoke('ji')
contextualize_q_system_prompt = (
"Given a chat history and the latest user question "
"which might reference context in the chat history or imply knowledge about college-related topics, "
"formulate a standalone question which can be understood "
"without the chat history. Expand any college-specific abbreviations or colloquialisms. "
"If the question is vague, add specificity based on common college inquiries "
"(e.g., fees, admissions, courses, campus life). "
"If a specific college is implied but not named, include it based on the chat history. "
"Include relevant numerical context if mentioned (e.g., years, semesters, amounts). "
"For comparative or ranking questions about colleges, clearly state this aspect. "
"For questions about deadlines or dates, include the current year if not specified. "
"Ensure any mentioned specific programs or majors are clearly stated. "
"Do NOT answer the question, just reformulate it if needed and otherwise return it as is. "
"The goal is to create a clear, standalone question that a college information system can accurately respond to."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_custom_history_aware_retriever(llm, retriever, contextualize_q_prompt)


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


question_answer_chain = create_custom_document_chain(llm, qa_prompt)    

rag_chain = create_custom_retrival_chain(history_aware_retriever, question_answer_chain)


response = rag_chain.run({"input": "What is Task Decomposition?"})
print(response["answer"])