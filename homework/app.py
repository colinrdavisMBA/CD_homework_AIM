#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary packages
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub



#from langchain.utils import itemgetter, RunnablePassthrough
#from langchain.chains import build_chain
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import PyMuPDFLoader


# In[2]:


#load environment var
from dotenv import load_dotenv
load_dotenv()


# In[3]:


#load in embeddings model
out_fp = './data'
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#vector_store = FAISS.from_documents(documents, embeddings)
faiss_fn = 'nvidia_10k_faiss_index.bin'
vector_store=FAISS.load_local(out_fp+faiss_fn, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()
openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# In[4]:


# ChatOpenAI Templates
template = """Answer the question based only on the following context. If you cannot answer the question with the context, respond with 'I don't know'. You'll get a big bonus and a potential promotion if you provide a high quality answer:

Context:
{context}

Question:
{question}
"""
prompt_template = ChatPromptTemplate.from_template(template)


# In[5]:


#create chain
retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=primary_qa_llm)
document_chain = create_stuff_documents_chain(primary_qa_llm, retrieval_qa_prompt)
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)


# In[6]:


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 250,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


# In[8]:


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    # Use the retrieval_augmented_qa_chain_openai pipeline with the user's question
    question = message.content  # Extracting the question from the message content
    response = retrieval_chain.invoke({"input": question}) # Invoke the pipeline
    #print(response['answer'])
    # Extract the response content and context documents
    response_content = response['answer']
    #context_documents = '\n'.join([document.page_content for document in response["context"]])
    #page_numbers = set([document.metadata['page'] for document in response["context"]])

    # Stream the response content back to the user
    msg = cl.Message(content="")
    await msg.stream_token(response_content)


# In[ ]:




