import gradio as gr

from agentic_workflow import AgentState, planner, executor, reflector, create_agentic_workflow

# def chat_function(message, history):
#     # Create workflow
#     agent_app = create_agentic_workflow()

#     # Invoke the agent workflow
#     result = agent_app.invoke({"input_query": message})
    
#     if result["satisfied"]:
#         return result["final_response"]
#     else:
#         return f"Agent Reflection: {result['critique']}"

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
    description="Ask about class schedules (e.g., '01-Alpha').",
    examples=["01-Alpha", "10-Beta"]
)

if __name__ == "__main__":
    demo.launch()