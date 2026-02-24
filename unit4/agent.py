from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.anthropic import Anthropic
import tools as toolbox
from llama_index.core.workflow import Context

def create_agent():

    llm = Anthropic(model="claude-haiku-4-5-20251001", temperature=0.1, max_tokens=1024)

    tool_list = [toolbox.websearch_tool,
                 toolbox.analyze_image_tool,
                 toolbox.wolfram_alpha_tool,
                 toolbox.read_pdf_tool,
                 toolbox.read_spreadsheet_tool,
                 toolbox.transcribe_audio_tool,
                 toolbox.execute_python_tool,
                 toolbox.youtube_transcript_tool]

    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=tool_list,
        llm=llm,
        system_prompt=(
            "You are a general AI assistant. I will ask you a question. "
            "Report your thoughts, and finish your answer with the following template: "
            "FINAL ANSWER: [YOUR FINAL ANSWER]. "
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "
            "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "
            "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "
            "If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n\n"

            "## TOOLS\n"
            "1. ALWAYS use a tool to find the answer. Never guess or rely on memory alone.\n"
            "2. For attached files, pick the right tool based on the file extension:\n"
            "   - .pdf -> read_pdf\n"
            "   - .png / .jpg / .jpeg / .gif / .webp -> analyze_image\n"
            "   - .mp3 / .wav / .m4a / .flac -> transcribe_audio\n"
            "   - .csv / .xlsx / .xls -> read_spreadsheet\n"
            "3. If a web search returns insufficient results, try a more specific or differently worded query.\n"
            "4. For math, prefer wolfram_alpha. For complex logic or counting, use execute_python.\n"
        )
    )
    ctx = Context(agent)
    return agent, ctx