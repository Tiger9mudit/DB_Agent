# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import json
import asyncio
import uuid # For generating thread IDs if needed

# Use the specific name exported from DBA_Agent.py
from DBA_agent_sql import graph_app
# Import message types for input preparation
from langchain_core.messages import HumanMessage, AIMessage

# Import SSE response type
from sse_starlette.sse import EventSourceResponse

# Initialize FastAPI app
app = FastAPI(title="SQL AI Agent API (Streaming)",
              description="API for natural language to SQL conversion with streaming responses.")

# CORS configuration (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None # Allow client to provide thread_id
    
# New Response Model for the synchronous endpoint
class SyncQueryResponse(BaseModel):
    result: List[str]
    thread_id: str

# --- Serve HTML Frontend ---
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serves the chat frontend HTML"""
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error reading index.html: {e}")

# --- SSE Streaming Endpoint ---
@app.post("/stream_query")
async def stream_query(request: ChatRequest):
    """
    Processes a message within a conversation thread and streams back events.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4()) # Use provided ID or generate a new one
    print(f"\nReceived request for thread_id: {thread_id}")

    # Configuration for LangGraph checkpointer
    config = {"configurable": {"thread_id": thread_id}}

    # Input for the graph: only the new human message
    inputs = {"messages": [HumanMessage(content=user_message)]}

    # Define the async generator for SSE events
    async def event_generator():
        try:
            # Yield the thread_id back to the client immediately
            yield json.dumps({"event": "metadata", "data": {"thread_id": thread_id}})

            # Use astream_events to get graph events (V2 provides more structure)
            async for event in graph_app.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                # print(f"Event Kind: {kind}") # Debugging: See all event kinds
                # print(f"Event Data: {event['data']}") # Debugging: See event data

                # Stream LLM tokens for the final answer generation
                # Check the name of the node that generates the final answer
                final_answer_nodes = ["generate_human_readable_answer", "generate_professional_response", "end_max_iterations"]
                if kind == "on_llm_stream" and event["name"] == "ChatGoogleGenerativeAI" and event["tags"] and any(node_name in event["tags"] for node_name in final_answer_nodes):
                     chunk = event["data"].get("chunk")
                     # Get the name of the node that produced this stream event
                     node_name = event["name"]
                     print(f"\nDEBUG: Received 'on_llm_stream' from node: '{node_name}'") # Print node name
                     if chunk and hasattr(chunk, 'content'):
                         print(f"Streaming chunk: {chunk.content}") # Debugging
                         yield json.dumps({"event": "ui_stream", "data": chunk.content})

                # Optionally, send messages about which node is running
                # elif kind == "on_chain_start":
                #      node_name = event.get("name", "Unknown Node")
                #      if node_name != "LangGraph": # Ignore the overall graph start
                #         yield json.dumps({"event": "ui_status", "data": f"Running: {node_name}..."})

                # Signal completion (or handle errors)
                elif kind == "on_chain_end":
                    # Check if this is the end of a final answer node
                    node_name = event.get("name", "Unknown Node")
                    print(f"\nDEBUG: Received 'on_chain_end' for node: '{node_name}'") # DEBUG
                    if node_name in final_answer_nodes:
                        # Extract final answer if not fully streamed (optional backup)
                        output = event.get("data", {}).get("output")
                        if output and isinstance(output, dict) and output.get('final_answer'):
                        #      # This might be needed if streaming fails or for non-streaming nodes
                            yield json.dumps({"event": "ui_final", "data": output['final_answer']})
                        # yield json.dumps({"event": "ui_status", "data": f"Finished: {node_name}."})

                # You could add more event handling here (on_tool_start, on_tool_end, etc.)
                # to provide more granular feedback to the UI.

            # Signal the absolute end of the stream
            yield json.dumps({"event": "stream_end", "data": "Stream finished."})

        except Exception as e:
            print(f"Error during stream for thread {thread_id}: {e}")
            # Send an error event to the client
            try:
                yield json.dumps({"event": "error", "data": f"An server error occurred: {str(e)}"})
            except Exception:
                # Ignore if client disconnected
                pass
        finally:
            print(f"Stream closed for thread_id: {thread_id}")


    # Return the SSE response using the generator
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# --- NEW Synchronous Query Endpoint ---
@app.post("/query_sync", response_model=SyncQueryResponse)
async def query_sync(request: ChatRequest):
    """
    Processes a message and returns a single JSON response after completion.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    print(f"\nReceived synchronous query for thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=user_message)]}

    try:
        # Invoke the graph and wait for the final state
        # Use .ainvoke for async compatibility with FastAPI
        final_state = await graph_app.ainvoke(inputs, config=config)
        
        # print(f"DEBUG: Final agent state for thread {thread_id}: {final_state}") # For debugging

        result_strings = []
        
        # Attempt to extract information from the final_state.
        # This part is highly dependent on your AgentState structure in DBA_agent_sql.py

        # 1. Extract Generated SQL Query
        # Try to find a generated SQL query. Common places:
        #   - A specific key like 'sql_query' or 'generated_sql'
        #   - Within 'intermediate_steps' if your agent collects tool calls
        #   - Sometimes embedded in the content of an AIMessage before the final answer
        
        # Placeholder - you'll need to adjust this based on your AgentState
        generated_sql = final_state.get("sql_query") # Example key
        if not generated_sql:
            # Look in messages if intermediate steps are added as messages
            # This is a common pattern if you use `ToolMessage` or store steps in `messages`
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.tool_calls: # Check for tool calls
                    for tc in msg.tool_calls:
                        if tc.get("name", "").lower() == "sql_db_query": # Or your SQL tool name
                           # Assuming the query is in `args` if it's a ToolCall object
                           if isinstance(tc.get("args"), dict) and "query" in tc["args"]:
                               generated_sql = tc["args"]["query"]
                               break
                           elif isinstance(tc.get("args"), str): # If args is just the query string
                               generated_sql = tc["args"]
                               break
                if generated_sql:
                    break
        
        if generated_sql:
            result_strings.append(f"generated query: {generated_sql}")

        # 2. Extract Output Result (Final Answer or DB Result)
        # This is typically the main answer from the agent.
        # It could be in a key like 'answer', 'final_answer', or the content of the last AIMessage.
        output_result = final_state.get("answer") # Example common key for final answer
        
        if not output_result:
            # Get content from the last AIMessage in the 'messages' list
            if final_state.get("messages") and isinstance(final_state["messages"][-1], AIMessage):
                output_result = final_state["messages"][-1].content
        
        if output_result:
            result_strings.append(f"output-result: {output_result}")
        else:
            result_strings.append("output-result: No specific output found in final state.")


        if not result_strings: # Fallback if no specific items were found
            result_strings.append("output-result: Agent finished, but no structured output was extracted.")

        return SyncQueryResponse(result=result_strings, thread_id=thread_id)

    except Exception as e:
        print(f"Error during synchronous query for thread {thread_id}: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An server error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # You could add a quick DB check here if needed
    # try:
    #     from DBA_Agent import db
    #     db._execute("SELECT 1")
    #     db_status = "connected"
    # except Exception as e:
    #     db_status = f"error: {e}"
    # return {"status": "healthy", "database": db_status}
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Use reload=True for development, turn off in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)