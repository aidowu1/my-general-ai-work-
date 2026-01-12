from typing import TypedDict, List, Optional
import datetime
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Time, or_
from sqlalchemy.orm import declarative_base, sessionmaker
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import tools
from data_model import create_new_database_session

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load environment variables from the .env file (if present)
load_dotenv()

class QueryExtraction(BaseModel):
    subject_type: Optional[str] = Field(None, description="Extracted subject type, e.g., 'Maths' or 'English' or 'Physics'")
    teacher_name: Optional[str] = Field(None, description="Extracted teacher name, e.g., 'Ms. Elara Vance'")
    is_next_query: bool = Field(False, description="True if user is asking for the 'next' or 'upcoming' class specifically.")

# Initialize GPT-4o mini
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind the schema to the LLM
structured_llm = llm.with_structured_output(QueryExtraction)

# --- 1. AGENT STATE ---
class AgentState(TypedDict):
    input_query: str
    extracted_data: QueryExtraction
    raw_results: List[str]
    final_response: str
    satisfied: bool

# --- 2. PLANNER: NL ENTITY EXTRACTION ---
def planner(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Extract entities from this high school query: '{query}'. "
        "Return the teacher name (e.g., 'Ms. Vance') and/or subject type (e.g., 'Maths' or 'English' or 'Physics'). "
        "Also detect if they are asking for the 'next' or 'upcoming' lesson."
    )
    chain = prompt | structured_llm # Uses GPT-4o mini as defined previously
    extraction = chain.invoke({"query": state['input_query']})
    print(f"Planner extracted: {extraction}\n\n")
    return {"extracted_data": extraction}

# --- 3. EXECUTOR: DYNAMIC DB SEARCH ---
def executor(state: AgentState):    
    ext = state["extracted_data"]
    
    # 2026 Contextual Time (Simulated or Real)
    now = datetime.datetime.now() # e.g., Monday, 09:00:00
    current_day = now.strftime("%A")
    current_time = now.strftime("%H:%M:%S")    

    # Filtering logic
    if ext.teacher_name:
        results = tools.get_time_schedule_by_teacher(ext.teacher_name)
    if ext.subject_type:
        results = tools.get_time_schedule_by_subject(ext.subject_type)

    # "Next Class" logic: Find lessons later today or on following days
    if ext.is_next_query:
        results = tools.get_next_time_schedule_by_subject(              
            current_day,
            current_time    
        )        
    
    # Format result list
    # data = [f"{l.day_of_week} at {l.start_time} for {l.end_time}" for l in results]
    return {"raw_results": results}

# --- 4. REFLECTOR: NL RESPONSE GENERATION ---
def reflector(state: AgentState):
    if not state["raw_results"]:
        return {"satisfied": False, "critique": "I couldn't find any upcoming classes for that teacher or subject."}
    
    # Format a conversational answer using the LLM
    answer_prompt = (
        f"User asked: {state['input_query']}. "
        f"Database results: {state['raw_results'][:1]}. " # Just the next one
        "Provide a polite, direct answer for a student or parent."
    )
    formatted_res = llm.invoke(answer_prompt).content
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
