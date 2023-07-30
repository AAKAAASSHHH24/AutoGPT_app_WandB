"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging

import wandb
#from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from prompts import load_chat_prompt

from typing import List,Tuple


logger = logging.getLogger(__name__)


def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    # load vector store artifact
    vector_store_artifact_dir = wandb_run.use_artifact("vector_store_artifact:latest").download()
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )

    return vector_store


def load_chain(question:str,db:Chroma, wandb_run: wandb.run, vector_store: Chroma, openai_api_key: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """
    docs = db.similarity_search(question, k=5)
    docs_page_content = " ".join([d.page_content for d in docs])

    """memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    )
    memory.chat_memory.add_user_message("My name is Akash, what is your name?")
    memory.chat_memory.add_ai_message("Hello, Akash! My name is AI_YIUTUBE_CHAT. How can I help you today?")"""

    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=wandb_run.config.model_name,
        temperature=wandb_run.config.chat_temperature,
        max_retries=wandb_run.config.max_fallback_retries,
        max_tokens=2048
    )
    chat_prompt_dir = wandb_run.use_artifact("chat_prompt_artifact:latest", type="prompt").download()
    qa_prompt = load_chat_prompt(docs,f_name=f"{chat_prompt_dir}/prompt.json")
    
    """"qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        qa_prompt=qa_prompt,
        memory=memory
    )"""
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    return qa_chain,docs_page_content

