from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
import yaml
import os
from tools.final_answer import FinalAnswerTool
from tools.get_timezone import FindTimezone
from tools.timezone_time import GetTimeInTimezone
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DDGWebSearch

from Gradio_UI import GradioUI



# gathering our tools
final_answer = FinalAnswerTool()
find_timezone_of_location = FindTimezone()
find_time_of_timezone = GetTimeInTimezone()
visit_webpage = VisitWebpageTool()
web_search = DDGWebSearch()

# Import tool from Hub
image_generator  = load_tool("agents-course/text-to-image", trust_remote_code=True)

# defining model
model = InferenceClientModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)

# Resolve prompts.yaml relative to this file to avoid CWD issues
_base_dir = os.path.dirname(__file__)
_prompts_path = os.path.join(_base_dir, "prompts.yaml")
with open(_prompts_path, 'r', encoding='utf-8') as stream:
    prompt_templates = yaml.safe_load(stream)


    
agent = CodeAgent(
    model=model,
    tools=[final_answer, image_generator, find_timezone_of_location, find_time_of_timezone, visit_webpage, web_search], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="Krümmel",
    description=None,
    prompt_templates=prompt_templates,
    additional_authorized_imports=["numpy", "pandas", "matplotlib", "requests", "json", "PIL", "smolagents.agent_types"]  # PIL and agent_types for image handling
)


GradioUI(agent).launch()