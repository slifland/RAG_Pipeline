# RAG_Pipeline
 Creates a basic RAG pipeline using Chromadb and the OpenAI api

 Time Taken:
 Initial research: 1 hour
 Setup: 3 hours (includes lots of computer specific issues, because my computer did not want to cooperate)
 Development/testing: 1 and a half hours

 The project is a basic RAG pipeline, which first creates embeddings for three provided documents and stores them in a database. This is done using ChromaDB. The three documents I used were various chronicles and histories of Florence, including Machiavelli's famous Florentine Histories, first published in 1532. The first step of the process is the splitting of the documents into smaller text chunks to allow for easier processing. This is done using langchain's recursive text splitter. In this specific case, the documents are split into chunks of 500, with an overlap of 25 characters on either side. After the documents are split, they are loaded into a Chromadb database. Note that this database is not created, but instead loaded, if it is already created; which process is preferred is specified by the "need_to_load" variable, which indicates whether the documents need to be embedded and loaded into a new database. After all this is done, the user is asked for a query, which is turned into a vector. A vector similiarity search is completed to find relevant documents, and the top k documents are returned. These documents are then passed as context to ChatGPT, along with the question provided. The response given by gpt is printed, and gpt will respond NONE if no answers are provided in the documents.

 Setup instructions:
1. Use pip to install all of the required libraries (openai, langchain-community, chromadb, langchain-text-splitters, langchain-openai; os should already be installed)
2. Load the api key into system variables with the name KEY
3. If desired to use the three documents loaded in, keep the file as is and run - otherwise, delete the local embeddings folder, place new files into the text_files folder, and change the need_to_load variable to True; make sure to change it to False after one run, or it will throw an error
4. Run the file, and provide a query when prompted

Process Description:
The basic process is that the documents are stored as embeddings, and when the query is given, it is also stored as a vector embedding. Cosine similarity can then be used to find which of the documents are the most relevant to the query. The most similar k ones as defined by the k parameter are returned. When these documents are compiled, they are then used by GPT as grounding and context when generating a response to the user given prompt. If GPT does not detect relevant information in the documents, it insteads responds with "NONE".
 
