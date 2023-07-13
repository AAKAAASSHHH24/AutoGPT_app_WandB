"""Ingest a directory of documentation files into a vector store and store the relevant artifacts in Weights & Biases"""
import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple

import langchain
import wandb
from langchain.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import find_dotenv, load_dotenv

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

def load_documents(video_url: str) -> str:
    """Load documents from given url of youtube video

    Args:
        video_url (str): youtube video url 

    Returns:
        str: Trancsript of the video
        
    """
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
      
    return transcript


def chunk_documents(
    documents: str, chunk_size: int = 500, chunk_overlap=100
) -> List[Document]:
    """Split documents into chunks

    Args:
        documents: A transcript to split into chunks
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 0.

    Returns:
        List[Document]: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents= text_splitter.split_documents(documents)
    
    return split_documents


def create_vector_store(
    documents,
    vector_store_path: str = "./vector_store",
) -> Chroma:
    """Create a Chroma vector store from a list of documents

    Args:
        documents (_type_): A list of documents to add to the vector store
        vector_store_path (str, optional): The path to the vector store. Defaults to "./vector_store".

    Returns:
        Chroma: A Chroma vector store containing the documents.
    """
    api_key = os.environ.get("OPENAI_API_KEY", None)
    embedding_function = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path,
    )
    vector_store.persist()
    return vector_store


def log_dataset(documents: List[Document], run: "wandb.run"):
    """Log a dataset to wandb

    Args:
        documents (List[Document]): A list of documents to log to a wandb artifact
        run (wandb.run): The wandb run to log the artifact to.
    """
    document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
    with document_artifact.new_file("documents.json") as f:
        for document in documents:
            f.write(document.json() + "\n")

    run.log_artifact(document_artifact)


def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb

    Args:
        vector_store_dir (str): The directory containing the vector store to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)


def log_prompt(prompt: dict, run: "wandb.run"):
    """Log a prompt to wandb

    Args:
        prompt (str): The prompt to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    prompt_artifact = wandb.Artifact(name="chat_prompt", type="prompt")
    with prompt_artifact.new_file("prompt.json") as f:
        f.write(json.dumps(prompt))
    run.log_artifact(prompt_artifact)


def ingest_data(
    video_url: str,
    chunk_size: int,
    chunk_overlap: int,
    vector_store_path: str,
) -> Tuple[List[Document], Chroma]:
    """Ingest a directory of markdown files into a vector store

    Args:
        video_url (str):
        chunk_size (int):
        chunk_overlap (int):
        vector_store_path (str):


    """
    # load the documents
    documents = load_documents(video_url)
    # split the documents into chunks
    split_documents = chunk_documents(documents, chunk_size, chunk_overlap)
    # create document embeddings and store them in a vector store
    vector_store = create_vector_store(split_documents, vector_store_path)
    return split_documents, vector_store


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_url",
        type=str,
        required=True,
        help="The link of the youtube url",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=0,
        help="The number of tokens to overlap between document chunks",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="The directory to save or load the FAISS db to/from",
    )
    parser.add_argument(
        "--prompt_file",
        type=pathlib.Path,
        default="./chat_prompt.json",
        help="The path to the chat prompt to use",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmapps",
        type=str,
        help="The wandb project to use for storing artifacts",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project=args.wandb_project, config=args)
    documents, vector_store = ingest_data(
        video_url=args.video_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store,
    )
    log_dataset(documents, run)
    log_index(args.vector_store, run)
    log_prompt(json.load(args.prompt_file.open("r")), run)
    run.finish()


if __name__ == "__main__":
    main()