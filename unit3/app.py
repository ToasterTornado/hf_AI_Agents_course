from retriever import get_retriever_agent_as_tool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import tools as toolbox
from llama_index.core.workflow import Context

def create_alfred_agent():

    retriever_agent = get_retriever_agent_as_tool()

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", max_new_tokens=4096, timeout=120)


    tool_list =[retriever_agent, 
                toolbox.get_most_downloaded_model_by_creator_tool, 
                toolbox.websearch_tool, 
                toolbox.get_latest_news_tool,
                toolbox.get_coordinates_tool,
                toolbox.get_weather_forecast_tool,
                toolbox.get_current_weather_tool]

    tool_descriptions = "\n".join([f"{tool.metadata.name}: {tool.metadata.description}" for tool in tool_list])

    alfred_agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=tool_list,
        llm=llm,
        system_prompt=(
            "You are Butler Alfred at Wayne Mansion, coordinating a high-profile party. "
            "You have access to multiple tools:\n\n"
            f"{tool_descriptions}\n\n"
            "Delegate to the appropriate tool based on the question. "
            "For guest-related queries, always use the invitees_specialist."
            "Provide complete, detailed responses without truncation."
        )
    )
    ctx = Context(alfred_agent)
    return alfred_agent, ctx

