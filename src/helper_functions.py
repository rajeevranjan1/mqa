import os
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np

client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def create_chunks(directory):
    print("Creating Chunks...")
    documents=DirectoryLoader(directory).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs=text_splitter.split_documents(documents)
    print(f"{len(docs)} Chunks created...")
    return docs

def create_dataframe(chunks):
    print("Creating Dataframe from Chunks...")
    df=pd.DataFrame(chunks)
    df.rename(columns={0:"Content",1:"Metadata",2:"Type"},inplace=True)
    df['Content']=df['Content'].apply(lambda x: x[1])
    df['Content']=df['Content'].str.replace("\n\n"," ")
    df.drop(['Type','Metadata'],axis=1,inplace=True)
    print("DataFrame Created...")
    return df

def get_embedding(text,model='text-embedding-3-small'):
    embedding=client.embeddings.create(input=[text],model=model)
    return embedding.data[0].embedding

def get_ques_embedding(ques):
    print("Creating Question Embedding...")
    return client.embeddings.create(input=ques,model='text-embedding-3-small').data[0].embedding

def get_context(query_embedding,dataframe):
    cosine_similarity=[]
    for i in range(dataframe.shape[0]):
        cosine_similarity.append(np.dot(query_embedding,dataframe['Embedding'][i]))
    index = np.argmax(np.array(cosine_similarity))
    print("MATCHING INDEX: ",index)
    print("Context Created...")
    return dataframe['Content'][index]

# completion=client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role":"user",
#         "content":"Mention some of the areas you are not trained on"
#         }
#         ]
#     )
# print(completion.choices[0].message.content)
