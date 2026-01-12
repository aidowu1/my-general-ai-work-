import gradio as gr

from agentic_workflow_v2 import AgentState, planner, executor, reflector, create_agentic_workflow

def chat_function(message, history):
    # Create workflow
    agent_app = create_agentic_workflow()

    # The workflow now processes natural language questions
    result = agent_app.invoke({"input_query": message})
    
    if result.get("satisfied"):
        return result["final_response"]
    else:
        return f"Agent Reflection: {result['critique']}"

demo = gr.ChatInterface(
    fn=chat_function,
    title="High School Timetable AI (2026)",
    description="Ask about lesson schedules by teacher (e.g. 'Dr. Aris Thorne') or subject (e.g., 'Maths' or 'English').",
    examples=["What is the time schedule for Maths", 
              "What is the time schedule for English", 
              "What lessons are schudled to be taught by Dr. Aris Thorne", 
              "Next lesson for Maths"]
)

if __name__ == "__main__":
    demo.launch()