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


def load_chat_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        template = json.load(f_name.open("r"))
    else:
        logger.warning(
            f"No chat prompt provided. Using default chat prompt from {__name__}"
        )
        # Template to use for the system message prompt
        template = str("""
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript:
            You may use markdown or bullet points format as per convenience and user question to answer the questions.
            Only use the factual information from the transcript to answer the question.
            
            If you feel like you don't have enough information to answer the question, say "I don't know".
            
            Your answers should be verbose and detailed.
            """)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template["system_template"])

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]  )      
    
    return prompt


def load_eval_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        human_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No human prompt provided. Using default human prompt from {__name__}"
        )

        human_template = """\nQUESTION: {query}\nCHATBOT ANSWER: {result}\n
        ORIGINAL ANSWER: {answer} GRADE:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """You are an evaluator for the AI_YOUTUBE_CHAT.You are given a question, the chatbot's answer, and the original answer, 
        and are asked to score the chatbot's answer as either CORRECT or INCORRECT. Note 
        that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the 
        best answer. You are evaluating the chatbot's answer only. Example Format:\nQUESTION: question here\nCHATBOT 
        ANSWER: student's answer here\nORIGINAL ANSWER: original answer here\nGRADE: CORRECT or INCORRECT here\nPlease 
        remember to grade them based on being factually accurate. Begin!"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt