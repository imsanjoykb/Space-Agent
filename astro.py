# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import streamlit as st
import os

# Read the OpenAI API key from the secrets.toml file
openai_api_key = st.secrets["openai"]["api_key"]

# Set up API keys and environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'  # Using GPT-4 as in your original code

# Initialize tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define agents
mission_planner_agent = Agent(
    role="Mission Planner",
    goal="Provide high-level, strategic advice for space mission planning based on the user's query.",
    backstory="An experienced mission planner specializing in strategic insights and identifying key steps to address space mission problems.",
    verbose=True,
    allow_delegation=True,
    tools=[]  # No tools for Mission Planner
)

space_operations_expert_agent = Agent(
    role="Space Operations Expert",
    goal="Elaborate on the Mission Planner's answer and provide detailed, actionable points for space operations.",
    backstory="A space operations expert with deep knowledge of mission functionalities and best practices for implementation.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool]
)

space_data_analyst_agent = Agent(
    role="Space Data Analyst",
    goal="Write SQL queries based on the provided database schema to address the user's question about space missions.",
    backstory="A skilled space data analyst with expertise in SQL and database management.",
    verbose=True,
    allow_delegation=True,
    tools=[]  # No tools for Space Data Analyst
)

qa_expert_agent = Agent(
    role="Quality Assurance Expert",
    goal="Provide insights into testing strategies and quality assurance practices for space missions.",
    backstory="An experienced QA professional specializing in space mission implementations and best practices for ensuring software quality.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool]
)

# Define tasks
mission_planner_task = Task(
    description="Analyze the user's query and provide high-level, strategic advice for space mission planning. Identify key steps needed to address the problem.",
    agent=mission_planner_agent,
    expected_output="A high-level strategic advice and key steps to address the space mission problem."
)

space_operations_expert_task = Task(
    description="Elaborate on the Mission Planner's answer by breaking down the key steps into detailed, actionable points for space operations. Integrate relevant information from space mission resources.",
    agent=space_operations_expert_agent,
    expected_output="Detailed, actionable points and relevant information from space mission resources."
)

space_data_analyst_task = Task(
    description="Write an optimized and correct SQL query that addresses the user's question about space missions based on the provided database schema.",
    agent=space_data_analyst_agent,
    expected_output="An optimized and correct SQL query."
)

qa_expert_task = Task(
    description="Provide insights into testing strategies and quality assurance practices for the given space mission query. Offer analysis on space mission functionalities and final recommendations.",
    agent=qa_expert_agent,
    expected_output="Insights into testing strategies, quality assurance practices, and final recommendations."
)

# Create the crew
space_mission_crew = Crew(
    agents=[mission_planner_agent, space_operations_expert_agent, space_data_analyst_agent, qa_expert_agent],
    tasks=[mission_planner_task, space_operations_expert_task, space_data_analyst_task, qa_expert_task],
    manager_llm=ChatOpenAI(model="gpt-4", temperature=0.7),
    process=Process.sequential,
    verbose=True
)

# Streamlit app
st.title("Space Agents")
st.write("Space missions require managing complex operations, often relying on repetitive and time-intensive tasks.")

user_query = st.text_input("Enter your query related to space missions:")

if st.button("Submit"):
    if user_query:
        with st.spinner("Processing your query..."):
            result = space_mission_crew.kickoff({"query": user_query})
            st.write("Result:", result)
    else:
        st.write("Please enter a query.")
