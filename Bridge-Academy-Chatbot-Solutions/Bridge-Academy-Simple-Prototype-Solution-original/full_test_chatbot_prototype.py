import gradio as gr
from typing import TypedDict, List
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Time
from sqlalchemy.orm import declarative_base, sessionmaker
from langgraph.graph import StateGraph, END

import tools

DB_PATH = 'sqlite:///school_timetable.db'

# --- 1. DATABASE SETUP ---
Base = declarative_base()
class SchoolClass(Base):
    __tablename__ = 'classes'
    id = Column(Integer, primary_key=True)
    grade_name = Column(String)

class Teacher(Base):
    __tablename__ = 'teachers'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class Lesson(Base):
    __tablename__ = 'lessons'
    id = Column(Integer, primary_key=True)
    class_id = Column(Integer, ForeignKey('classes.id'))
    teacher_id = Column(Integer, ForeignKey('teachers.id'))
    day_of_week = Column(String)
    start_time = Column(String) # For simplicity in this prototype
    end_time = Column(String)

# engine = create_engine('sqlite:///school_timetable.db')
# Base.metadata.create_all(engine)
# Session = sessionmaker(bind=engine)

engine = create_engine(DB_PATH)
session_instance = sessionmaker(bind=engine)

# --- 2. AGENT STATE & LOGIC ---
class AgentState(TypedDict):
    input_query: str
    plan: str
    raw_data: List[str]
    final_response: str
    critique: str
    satisfied: bool

def planner(state: AgentState):
    # Reason: Identify that we need to search for '01-Alpha' in the DB
    return {"plan": f"Look up all lessons for grade '{state['input_query']}'."}

def executor(state: AgentState):
    # Act: Perform the database query
    session = session_instance()
    lessons = session.query(Lesson).join(SchoolClass).all()
    for lesson in lessons:
        print(f"Lesson ID: {lesson.id}, Class ID: {lesson.class_id}, Teacher ID: {lesson.teacher_id}, Day: {lesson.day_of_week}, Start: {lesson.start_time}, End: {lesson.end_time}")

    # results = session.query(Lesson).join(SchoolClass).filter(
    #     SchoolClass.grade_name == state['input_query']
    # ).all()
    # results = session.query(Lesson).join(SchoolClass).filter(
    #     SchoolClass.grade_name == '01-Alpha'
    # ).all()
    class_name = str(state['input_query'])
    results = tools.get_schedule(class_name) 
    print(f"Executor found {len(results)} lessons for class {state['input_query']}.")
    session.close()
    data = [f"{l.day_of_week} at {l.start_time}" for l in results]
    return {"raw_data": data}

def reflector(state: AgentState):
    # Reflect: Check if data was found. If empty, plan a 'not found' message.
    if not state["raw_data"]:
        return {"satisfied": False, "critique": "No schedule found for this class."}
    return {"satisfied": True, "final_response": f"Schedule for {state['input_query']}: {', '.join(state['raw_data'])}"}

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

# --- 4. GRADIO INTERFACE ---
def chat_function(message, history):
    # Invoke the agent workflow
    result = agent_app.invoke({"input_query": message})
    
    if result["satisfied"]:
        return result["final_response"]
    else:
        return f"Agent Reflection: {result['critique']}"

demo = gr.ChatInterface(
    fn=chat_function,
    title="High School Timetable AI (2026)",
    description="Ask about class schedules (e.g., '01-Alpha').",
    examples=["01-Alpha", "10-Beta"]
)

if __name__ == "__main__":
    demo.launch()
