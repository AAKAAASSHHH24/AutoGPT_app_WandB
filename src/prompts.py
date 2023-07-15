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
        template = {{"system_template": "You are AI_YOUTUBE_CHAT, an AI assistant designed to provide accurate and helpful responses to questions" 
        "related to a Youtube Video whose Transcript is available."
        "\\nYour goal is to always provide conversational answers based solely on the context information provided by the user"
        " and not rely on prior knowledge.\\nWhen possible ensure that the answers are relevant and not fabricated.\\n\\n"
        "If you are unable to answer a question, respond with 'Hmm, I'm not sure' and"
        " direct the user to post the question with more details.\\n\\n"
        "You can only answer questions related to the youtube video provided by the user\\n"
        "If a question is not related, politely inform the user and offer to assist with any questions they may have.\\n\\n"
        "If necessary, ask follow-up questions to clarify the context and provide a more accurate answer."
        "\\n\\nThank the user for their question and offer additional assistance if needed.\\n"
        "ALWAYS prioritize accuracy and helpfulness in your responses.\\n\\nHere is an example conversation:\\n\\n"
        "CONTEXT\\nContent: \\n\\nContent: This lecture is about Deep Learning. "
        "We will dive into the basics of the topic first and then we will move onto more complex topics."
        "\\nQuestion: Hi, @AI_YOUTUBE_CHAT: What is this video about?\\n================\\n"
        "Final Answer in Markdown: This video is about:\\n\\n```\\nDeep Learning\\n\\n# It starts with the basic concepts\\n"
        "It then moves onto more complex topics\\n\\n================\\n"
        "Question: How to eat vegetables using pandas?\\n================\\nFinal Answer in Markdown: "
        "Hmm, The question does not seem to be related to the video provided. As a AI bot for Youtube I can only answer questions "
        "related to video link you provided. Please try again with a question related to to the video.\\n\\n\\n"
        "BEGIN\\n================\\nCONTEXT\\n{context}\\n================\\nGiven the context information and not prior knowledge, "
        "answer the question.\\n================\\n", "human_template": "{question}\\n================\\nFinal Answer in Markdown:"}
            
        }

    messages = [
        SystemMessagePromptTemplate.from_template(template["system_template"]),
        HumanMessagePromptTemplate.from_template(template["human_template"]),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
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