import os
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
from helper_functions import *
from ast import literal_eval

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embeddings_path="D:\\Projects\\MQA\\Embeddings\\embedding.csv"

if os.path.exists(embeddings_path):
    df=pd.read_csv(embeddings_path)
    df["Embedding"]=df["Embedding"].apply(literal_eval)
else:
    docs=create_chunks('D:\\Projects\\MQA\\data')
    df=create_dataframe(docs)
    df['Embedding']=df.Content.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    try:
        df.to_csv("D:\\Projects\\MQA\\Embeddings\\embedding.csv",index=False)
    except:
        print("Continuing without exporting embeddings to csv")

## Finalising the code
question="What is the validity of Doubt Support?"    
qe=get_ques_embedding(question)     
context=get_context(qe,df)

content=f'''Find the answer from given context only. Otherwise answer: "Could not find answer in the context"
Context: {context}
Q: {question}
A: 
'''

print(content[106:130])

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role":"system","content":"You are a question answer system which picks answer from the user context"},
        {"role":"user","content":content}
    ]
)
print("**********Response***********\n")
print(response.choices[0].message.content)

