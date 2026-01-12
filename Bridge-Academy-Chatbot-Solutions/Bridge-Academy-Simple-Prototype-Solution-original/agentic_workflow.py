from typing import TypedDict, List
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Time
from sqlalchemy.orm import declarative_base, sessionmaker
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import tools
from data_model import create_database_session

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load environment variables from the .env file (if present)
load_dotenv()

class QueryExtraction(BaseModel):
    grade_name: str = Field(description="The specific grade name extracted from the query, e.g., '10-Alpha'")

# Initialize GPT-4o mini
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind the schema to the LLM
structured_llm = llm.with_structured_output(QueryExtraction)

class AgentState(TypedDict):
    input_query: str
    extracted_grade: str
    plan: str
    raw_data: List[str]
    final_response: str
    critique: str
    satisfied: bool

def planner(state: AgentState):
    # Prompt to extract the grade name from natural language
    prompt = ChatPromptTemplate.from_template(
        "Extract the high school grade name from this user query: {query}. "
        "Common formats include '10-Alpha', '11-Beta', etc."
    )
    chain = prompt | structured_llm
    extraction = chain.invoke({"query": state['input_query']})
    
    # Store the extracted grade for the executor
    return {
        "plan": f"Searching for grade: {extraction.grade_name}",
        "extracted_grade": extraction.grade_name 
    }

def executor(state: AgentState):
    # Act: Use the extracted grade in the SQL query
    class_name = state['extracted_grade']
    results = tools.get_schedule_v2(class_name)
    print(f"Executor found {len(results)} lessons for class {class_name}.") 
    
    data = [f"{l.day_of_week} at {l.start_time}" for l in results]
    return {"raw_data": data}

def reflector(state: AgentState):
    # Reflect: If no data found, use LLM to explain why or ask for clarification
    if not state["raw_data"]:
        return {"satisfied": False, "critique": f"I couldn't find a schedule for '{state['extracted_grade']}'. Please verify the class name."}
    
    # Use LLM to format the raw data back into a natural language response
    response_prompt = f"The user asked about {state['extracted_grade']}. The database found: {state['raw_data']}. Write a friendly response."
    formatted_res = llm.invoke(response_prompt).content
    
    return {"satisfied": True, "final_response": formatted_res}

def create_agentic_workflow() -> StateGraph:
    # --- 3. CONSTRUCT LANGGRAPH ---
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("reflector", reflector)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")

    # Condition: If not satisfied, end with critique; else end with final response
    workflow.add_conditional_edges(
        "reflector",
        lambda x: "end" if x["satisfied"] else "end", 
        {"end": END}
    )
    agent_app = workflow.compile()
    return agent_app