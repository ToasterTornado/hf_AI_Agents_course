from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, FunctionTool
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from dotenv import load_dotenv

load_dotenv()


def build_retriever_agent(llm=None, similarity_top_k=3) -> AgentWorkflow:
    if llm is None:
        llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load the existing ChromaDB database
    db = chromadb.PersistentClient(path="./invitees_chroma_db")
    collection = db.get_or_create_collection(name="alfred")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Create index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        embed_model=embed_model
    )

    # Create query engine
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        response_mode="tree_summarize",
    )
    
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="invitees_info_retriever",
        description="Information retrieval tool to answer questions about the invitees based on the provided data.",
        return_direct=False,
    )

    # Create query engine agent
    query_engine_agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_engine_tool],
        llm=llm,
        system_prompt="You are the Butler Alfred and you serve as the host of the party at the Wayne Mansion. "
        "You are responsible for answering any questions about the invitees based on the provided data. "
        "You can use the invitees_info_retriever tool to retrieve information about the invitees. "
        "Always use the tool when you need to answer questions about the invitees. "
        "Be sure to always provide an input when using the tool. "
        "Only use the invitees_info_retriever tool to answer questions about the invitees or if it seems relevant to the task."
    )
    return query_engine_agent


def get_retriever_agent_as_tool(llm=None):
    """Wraps the retriever agent as a FunctionTool so it can be used by another agent"""
    retriever_agent = build_retriever_agent(llm=llm)
    
    async def query_invitees(query: str) -> str:
        """
        Query information about party invitees.
        
        Args:
            query: A question about the invitees (e.g., "What are Bruce Wayne's dietary preferences?")
        
        Returns:
            Information about the invitees based on the query
        """
        response = await retriever_agent.run(query)  # Add await here
        return str(response)
    
    # Wrap the async function as a tool
    retriever_tool = FunctionTool.from_defaults(
        fn=query_invitees,  # FunctionTool handles async functions automatically
        name="invitees_specialist",
        description=(
            "A specialist agent that can answer detailed questions about party invitees. "
            "Use this tool when you need information about guests, their preferences, "
            "dietary restrictions, backgrounds, or any other invitee-specific details."
        )
    )
    
    return retriever_tool