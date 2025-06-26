# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# response = client.responses.create(
#     model="gpt-4.1",
#     input=[
#         {
#             "role": "developer",
#             "content": "Talk like a pirate."
#         },
#         {
#             "role": "user",
#             "content": "Are semicolons optional in JavaScript?"
#         }
#     ]
# )

# print(response.output_text)


from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils import search_tool, wiki_tool, save_tool
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Set your Gemini API key (from makersuite)
client = genai.configure(api_key=api_key)
# model = genai.GenerativeModel("gemini-1.5-flash")
llm = genai.GenerativeModel("gemini-1.5-flash")
# # llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Step 4: Define the prompt template using LangChain style
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Use available tools: "search", "wiki", "save".

            - Use "search" if external data is needed.
            - Use "wiki" for known topics.
            - Use "save" if the user wants to store the result to a file.
            Wrap the output in this format and provide no other text:
            
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# tools = [search_tool, wiki_tool, save_tool]
# agent = create_tool_calling_agent(
#     llm=llm,
#     prompt=prompt,
#     tools=tools
# )

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("🔍 What can I help you research? ")

# Step 5: Generate response from Gemini
formatted_prompt = prompt.format(query=query)
response = llm.generate_content(formatted_prompt)
raw_text = response.text

# Step 6: Parse into structured output
try:
    structured = parser.parse(raw_text)
    print("\n📄 Structured Output:\n", structured)
except Exception as e:
    print("❌ Failed to parse Gemini response:", e)
    print("Raw response:\n", raw_text)
    exit()

# Normalize tools
tool_aliases = {
    "google search": "search",
    "search": "search",
    "wiki": "wiki",
    "wikipedia": "wiki",
    "un population division website": "search",
    "world bank data api": "search",
    "save": "save",
    "save to file": "save",
}

used_tools = set()
for t in structured.tools_used:
    t_lower = t.lower()
    for key in tool_aliases:
        if key in t_lower:
            used_tools.add(tool_aliases[key])

# ✅ Force save if query mentions it
if "save" in query.lower() or "file" in query.lower() or "store" in query.lower():
    used_tools.add("save")

# Run the tools
print("\n🛠️ Tool Usage:")
if "search" in used_tools:
    print("🔍 Search Tool Output:")
    print(search_tool.func(query))

if "wiki" in used_tools:
    print("📚 Wikipedia Tool Output:")
    print(wiki_tool.run(query))

if "save" in used_tools:
    print("💾 Save Tool Output:")
    filename = structured.topic.lower().replace(" ", "_") + "_output.txt"
    print(save_tool.func(structured.model_dump_json(indent=2), filename=filename))

# raw_response = agent_executor.invoke({"query": query})
# try:
#     structured_response = parser.parse(raw_response.get("output")[0]["text"])
#     print("\n📄 Structured Research Output:\n", structured_response)
# except Exception as e:
#     print("❌ Error parsing response:", e)
#     print("🔃 Raw Response:\n", raw_response)