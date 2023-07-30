"""Prompts for the chatbot and evaluation."""
import json
import logging
import pathlib
from typing import Union

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


def load_chat_prompt(context, f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        template = json.load(f_name.open("r"))
    else:
        logger.warning(
            f"No chat prompt provided. Using default chat prompt from {__name__}"
        )
        # Template to use for the system message prompt
        template = {"system_template": "You are a helpful assistant that that can answer questions about youtube videos based on the video's transcript: {context} You may use markdown or bullet points format as per convenience and user question to answer the questions. Only use the factual formation given the from the transcript to answer the question.If you feel like you don't have enough information to answer the question, say I don't know. Your answers should be verbose and detailed.", "human_template": "Answer the following question:{question}"}

    system_message_prompt = SystemMessagePromptTemplate.from_template(template["system_template"])

    # Human question prompt
    human_template = "Answer the following question:{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]  )      
    
    return prompt

