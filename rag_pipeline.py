import os
import openai
from openai import OpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#GLOBAL VARIABLES
need_to_load = False #defines whether we need to create and load the embeddings - once done once, shouldn't need to do again
taking_query = True #defines whether we should take a query from user input
gpt_prompt = True #defines whether we should use this query to construct a prompt to send to gpt
k = 5 #defines how many results we should return when querying the embeddings from our txt files
open_ai_key = os.getenv('KEY') #retrieves the key from environment variables
if not open_ai_key:
     raise ValueError("Key not initialized properly")
embedding_type = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(api_key=open_ai_key, model=embedding_type) #used for embedding
client = OpenAI(api_key=open_ai_key)
MODEL="gpt-4o-mini"


def get_embedding(text_to_embed):
    response = openai.embeddings.create(
        model= "text-embedding-3-small",
        input=[text_to_embed]
    )
    
    return response.data[0].embedding # Change this


if(need_to_load):
    #load in the three documents to the documents variable
    loader = DirectoryLoader("text_files", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    #split the docs into smaller chunks of text for embedding purposes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    docs = text_splitter.split_documents(documents)
    collection = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="local_embeddings")
    print('Done')
else:
    collection = Chroma(persist_directory="local_embeddings", embedding_function=embedding_type)
if(taking_query):
    query = input("Enter your query" + '\n')
    query_embedding = embeddings.embed_query(query)
    results = collection.similarity_search_by_vector(embedding=query_embedding, k=k)
   #for result in results:
        #print(result)
if(taking_query and gpt_prompt):
    prompt_context = "Documents: \n"
    for result in results:
        prompt_context =  prompt_context + "START DOC: " + result.page_content + " END DOC" + '\n' + '\n'
    prompt_question = query
    final_instructions = "\n Answer the question according to the documents provided. If the documents do not provide an adequate answer, please return NONE."
    prompt = prompt_context + '\n' + "Question: " + prompt_question + final_instructions
    print(prompt)
    completion = client.chat.completions.create(
	model=MODEL,
	messages=[{"role":"user", "content": prompt}]
    )
    print(completion.choices[0].message.content)
    
	    
