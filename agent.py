import tempfile
import time
import os
import redis
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain_openai.llms import OpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from io import BytesIO
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import datetime


from dotenv import load_dotenv
load_dotenv()


class Agent():
  

    def __init__(self,filename,contents,chunking_method,embedding_model_name,embedding_provider):   

        self.filename = filename    
        self.contents = contents
        self.chunking_method = chunking_method
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key =  os.getenv("PINECONE_API_KEY")
        self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
        self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")


        tools = [
            Tool(
                name="document_search",
                description="Search through uploaded documents for relevant information",
                func=self.search_document() 
            )          
        ]

        self.agent = initialize_agent( 
            tools=tools,        
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )


    def text_splitters(self, chunking_method, documents):
           
           match chunking_method.lower():

            case "recursive":
                recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
                chunked_documents = recursive_text_splitter.split_documents(documents)  
                return chunked_documents
               
            case "character":
                character_text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
                chunked_documents = character_text_splitter.split_documents(documents)  
                return chunked_documents
            
            case "token":
                token_text_splitter = TokenTextSplitter(encoding_name="o200k_base", chunk_size=1000,chunk_overlap=100)
                chunked_documents = token_text_splitter.split_documents(documents)  
                return chunked_documents
               
            case "semantic":
                semantic_text_splitter = SemanticChunker(OpenAIEmbeddings(model='text-embedding-3-large'), buffer_size=1,
                                    breakpoint_threshold_type='percentile', breakpoint_threshold_amount=70)
                chunked_documents = semantic_text_splitter.split_documents(documents)

                return chunked_documents 
                   
            case _:
                   return "no chunking method is selected"


    def check_embeding_latency(self,documents):    

        match self.embedding_provider.lower():
            case "openai":
                embedding = OpenAIEmbeddings(api_key=self.openai_api_key,model=self.embedding_model_name)
                starting_time = time.time()
                embedding.embed_documents(documents)
                ending_time = time.time()
                embedding_latency = ending_time - starting_time    

                return embedding, embedding_latency
            
            case "google":
                
                embedding = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key,model="models/"+self.embedding_model_name)
                starting_time = time.time()
                embedding.embed_documents(documents)
                ending_time = time.time()
                embedding_latency = ending_time - starting_time    

                return embedding, embedding_latency
            
            case "huggingface":

                embedding = HuggingFaceEmbeddings(model=self.embedding_model_name)
                # print(len(embedding))
                starting_time = time.time()
                embedding.embed_documents(documents)                
                ending_time = time.time()
                embedding_latency = ending_time - starting_time    

                return embedding, embedding_latency
            
            case _:
                return "no embedding model is selected"
            


    def retriever_latency(self,pc_vectordb,query=None):        
        query = "Can you provide topis in Microdegree AI?"
        starting_time = time.time()
        results = pc_vectordb.similarity_search(query, k=3)
        # results = pc_vectordb.similarity_search_with_score(query, k=3)        
        ending_time = time.time()
        embedding_latency = ending_time - starting_time
        return results, embedding_latency
    

    def process_file(self):
        
        if self.filename.endswith("pdf"):          

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(self.contents)
                temp_path = temp_file.name           

            try:
               
                read_docs = PyPDFLoader(temp_path)
                pdf_loader = read_docs.load()                  

                splitted_documents = self.text_splitters(self.chunking_method,pdf_loader)    
                chunk_count = len(splitted_documents)                           
                
                docs = [content.page_content for content in splitted_documents]
                embedding, embedding_lantency = self.check_embeding_latency(docs)
               

                pc = Pinecone(api_key=self.pinecone_api_key)
                index_openai = pc.Index("pinecodeindex")      
                index_google = pc.Index("pineconegoogleindex")  
                index_hugging = pc.Index("pineconehuggingfaceindex")  
                
                if self.embedding_provider == "openai":
                    pc_vector_openai = PineconeVectorStore.from_texts(
                    docs,
                    embedding,
                    metric = 'consine',
                    # metric = 'dotproduct',
                    # metric = 'euclidean',
                    index_name = 'pinecodeindex'
                    )     

                    # print("openai")
                    results, retreival_latency = self.retriever_latency(pc_vector_openai)

                elif self.embedding_provider == "google":
                    pc_vector_google = PineconeVectorStore.from_texts(
                    docs,
                    embedding,
                    metric = 'consine',
                    # metric = 'dotproduct',
                    index_name = 'pineconegoogleindex'
                    ) 
                    # print("google")
                    results, retreival_latency = self.retriever_latency(pc_vector_google)
                
                elif self.embedding_provider == "huggingface":
                    pc_vector_google = PineconeVectorStore.from_texts(
                    docs,
                    embedding,
                    metric = 'consine',
                    # metric = 'dotproduct',
                    index_name = 'pineconehuggingfaceindex'
                    ) 
                    # print("huggingface")
                    results, retreival_latency = self.retriever_latency(pc_vector_google)
                

                return chunk_count,embedding_lantency, retreival_latency, results 
            
            except Exception as e:
                print(e)              

            finally:
                os.unlink(temp_path)
            


    def process_query(self, query: str, session_id: str):

        try:
            memory = self.get_previous_memory(session_id)
            
        
            response = self.agent.run(
                input=query,
                memory=memory 
            )

            embedding = OpenAIEmbeddings(api_key=self.openai_api_key,model=self.embedding_model_name)
            pc_vector_openai = PineconeVectorStore.from_existing_index(
                embedding,
                index_name = 'pinecodeindex'
                )     

        
            results, retreival_latency = self.retriever_latency(pc_vector_openai, query)

            self.save_redis_memory(session_id, query, results)

        
            return {
                "answer": response,
                "sources": results,
                "session_id": session_id,            
            }
        
        except Exception as e:
            print(e)
          

    
      
    def get_previous_memory(self, session_id: str) -> ConversationBufferMemory:
        
        try:
            memory_data = self.redis_client.get(f"memory:{session_id}")
            if memory_data:
                messages = json.loads(memory_data)
                memory = ConversationBufferMemory()
                memory.chat_memory.messages = messages
                return memory
            
        except Exception as e:
            print(e)
        
        
    

    def save_redis_memory(self, session_id: str, query: str, response: str):
        
        try:
            memory_data = {
                "human": query,
                "ai": response,
                "timestamp": datetime.today()
            }            
            
            self.redis_client.setex(
                f"memory:{session_id}", 
                3600, 
                json.dumps(memory_data)
            )

        except Exception as e:
            print(e)

    if __name__== "__main__":
        process_file()

            


