# DBA_Agent.py
import os
import operator # <-- Import
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
import requests
import urllib
import sqlalchemy
from sqlalchemy import create_engine
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any # <-- Import Annotated, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # <-- Import message types
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # <-- Import MemorySaver
from sqlalchemy.orm import sessionmaker
# from config import settings, Settings
from sqlalchemy.orm import declarative_base
from sqlalchemy.engine import URL
import asyncio # <-- Import asyncio if using async nodes

load_dotenv()

# --- LLM Initialization ---
LLM_MODEL = "gemini-2.0-flash"
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0) # Consistent temperature

# --- Database Setup ---
def get_db() -> SQLDatabase:
    """Connect to SQL Server database with robust error handling"""
    # Configuration
    driver = "ODBC Driver 18 for SQL Server"  # Updated to ODBC 18
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")

    # Validate environment variables
    if not all([server, database, username, password]):
        missing = [var for var in ["SQL_SERVER", "SQL_DATABASE", "SQL_USERNAME", "SQL_PASSWORD"]
                   if not os.getenv(var)]
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        # Build connection string
        connection_string = (
            f"DRIVER={{{driver}}};" # Ensure driver name is enclosed in braces
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "Encrypt=yes;"  # Recommended for security
            "TrustServerCertificate=yes;"  # Often needed for dev/local SQL Server instances without proper certs. Change to 'no' and configure trust if needed.
            "Connection Timeout=30;"
        )
        params = urllib.parse.quote_plus(connection_string)
        connection_uri = f"mssql+pyodbc:///?odbc_connect={params}"

        # Engine configuration (optional but good practice)
        engine_args = {
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_recycle": 3600,
            # Removed connect_args with timeout/autocommit here, handled in connection string/driver
        }

        # Create and test connection
        db = SQLDatabase.from_uri(
            connection_uri,
            # include_tables=['your_table1', 'your_table2'], # Optional
            sample_rows_in_table_info=2, # Reduce sample rows
            engine_args=engine_args
        )

        # Test connection using a method less likely to conflict with table names
        db._execute("SELECT 1")
        print(f"✅ Successfully connected to {database} on {server}")
        return db

    except Exception as e:
        print(f"❌ Connection failed to {server}/{database}")
        print(f"Error details: {str(e)}")
        if "ODBC Driver" in str(e):
            print("\n⚠️ Possible solutions:")
            print("1. Verify ODBC driver is installed and the name is EXACTLY correct (including braces if needed by pyodbc).")
            print("2. Check SQL Server allows remote connections and firewall rules.")
            print("3. Ensure username/password are correct.")
            print("4. Check 'TrustServerCertificate' setting.")
        raise

try:
    db = get_db()
    # Cache schema to avoid repeated calls within a single request stream
    CACHED_SCHEMA = db.get_table_info()
    print(f"Retrieved and cached database schema (Length: {len(CACHED_SCHEMA)}).")
except Exception as e:
    print(f"FATAL: Failed to connect to database on startup: {str(e)}")
    # Application should not start if DB connection fails initially
    raise SystemExit(f"Database connection failed: {e}")

def get_database_schema(force_refresh: bool = False):
    """Get schema information, using cache by default."""
    global CACHED_SCHEMA
    if force_refresh or not CACHED_SCHEMA:
        print("Refreshing database schema...")
        CACHED_SCHEMA = db.get_table_info()
        print(f"Retrieved and cached database schema (Length: {len(CACHED_SCHEMA)}).")
    return CACHED_SCHEMA


# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question: str          # Current question being processed
    sql_query: str
    query_result: str      # Can be data string, error string, or success message
    query_rows: list       # Store actual rows from SELECT here if needed downstream
    attempts: int
    relevance: str
    sql_error: bool
    # Add a field to carry the final response for streaming if needed
    final_answer: str | None


# --- Pydantic Models ---
class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question relates to SQL operations (SELECT, CREATE TABLE, INSERT) based on the schema. Must be exactly 'relevant' or 'not_relevant'."
    )

class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The T-SQL query for SQL Server corresponding to the user's natural language question based on schema and history."
    )

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question, clarified for better SQL generation.")


# --- Graph Nodes ---

def check_relevance(state: AgentState):
    """Checks relevance of the latest user question."""
    if not state["messages"]:
         state["relevance"] = "not_relevant"
         state["final_answer"] = "I need a question first!"
         state["messages"] = [AIMessage(content=state["final_answer"])]
         return state

    question = state["messages"][-1].content
    state["question"] = question # Store current question
    schema = get_database_schema()
    print(f"\n--- Checking Relevance for: '{question}' ---")

    # Slightly updated prompt for clarity
    system = f"""You are an assistant determining if a question likely requires SQL operations (SELECT, CREATE TABLE, INSERT) based on the provided SQL Server schema. Ignore requests for DELETE or Drop commands.

    Schema:
    {schema}

    Respond ONLY with the JSON structure containing 'relevant' or 'not_relevant'.
    """
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"User Question: {question}"),
        ]
    )
    structured_llm = llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm

    # Ensure attempts are initialized
    state["attempts"] = state.get("attempts", 0)

    try:
        relevance_result = relevance_checker.invoke({})
        state["relevance"] = relevance_result.relevance
        print(f"Relevance determined: {state['relevance']}")
    except Exception as e:
        print(f"Error during relevance check: {e}")
        state["relevance"] = "not_relevant" # Default to not relevant on error
        state["final_answer"] = f"Sorry, I had trouble understanding if your question is relevant to the database schema. Error: {e}"
        state["messages"] = [AIMessage(content=state["final_answer"])] # Add error message

    state["final_answer"] = None # Clear any potential carry-over
    return state


def convert_nl_to_sql(state: AgentState):
    """Converts the user's question (with history context) to a T-SQL query."""
    messages = state["messages"]
    current_question = state["question"] # Use question set in relevance check
    schema = get_database_schema()
    print(f"\n--- Converting to SQL for: '{current_question}' ---")

    system = f"""You are an expert T-SQL assistant for SQL Server. Based on the provided database schema and the ongoing conversation history, convert the user's LATEST natural language question into a syntactically correct T-SQL query.

    Schema:
    {schema}

    Rules:
    - Only provide the T-SQL query.
    - Do not add any explanations or markdown formatting (like ```sql).
    - Use T-SQL syntax appropriate for SQL Server.
    - Consider the conversation history to resolve ambiguities or references (e.g., "list their details", "insert another one like the last").
    - Ensure table and column names match the schema EXACTLY.
    """
    # Filter history slightly if too long? (Optional optimization)
    # history = messages[-10:] # Example: Keep last 10 messages
    history = messages # Use full history for now

    convert_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            *history, # Include history
            ("human", f"Generate the T-SQL query for my last request: '{current_question}'"),
        ]
    )
    structured_llm = llm.with_structured_output(ConvertToSQL)
    sql_generator = convert_prompt | structured_llm

    try:
        result = sql_generator.invoke({}) # History passed in prompt messages
        sql_query = result.sql_query.strip().replace("```sql", "").replace("```", "").strip()
        # Basic check for disallowed operations (redundant with relevance prompt, but safer)
        if any(op in sql_query.upper() for op in ["UPDATE ", "DELETE "]):
             raise ValueError("Attempted to generate disallowed SQL operation (UPDATE/DELETE).")
        state["sql_query"] = sql_query
        print(f"Generated SQL query: {state['sql_query']}")
    except Exception as e:
        print(f"Error during SQL generation: {e}")
        state["sql_query"] = ""
        state["sql_error"] = True
        state["final_answer"] = f"Sorry, I encountered an error trying to generate the SQL query: {e}"
        state["messages"] = [AIMessage(content=state["final_answer"])] # Add error message

    return state

def execute_sql(state: AgentState):
    """Executes the generated SQL query against the SQL Server database."""
    sql_query = state.get("sql_query", "").strip()
    if not sql_query:
        print("--- Skipping SQL Execution (No Query) ---")
        # Ensure error flag is set if we got here without a query
        state["sql_error"] = True
        if not state.get("final_answer"): # Check if an error message already exists
             state["final_answer"] = "Cannot execute SQL as query generation failed."
             state["messages"] = [AIMessage(content=state["final_answer"])]
        return state

    print(f"\n--- Executing SQL: {sql_query} ---")
    state["sql_error"] = False # Assume success initially
    state["query_rows"] = [] # Reset rows

    try:
        result = db.run(sql_query) # Execute the SQL query

        state["query_result"] = str(result) # Store raw string result first

        if sql_query.upper().startswith("SELECT"):
            # Try to parse the string result if it looks like a list
            # This part is heuristic and might need adjustment based on db.run's exact output format
            parsed_rows = []
            if isinstance(result, str) and result.startswith('[') and result.endswith(']'):
                try:
                    # Use ast.literal_eval for safer evaluation than eval()
                    import ast
                    parsed_rows = ast.literal_eval(result)
                    if not isinstance(parsed_rows, list):
                         parsed_rows = [] # Treat as non-list if eval result isn't a list
                except (ValueError, SyntaxError, TypeError) as eval_e:
                    print(f"Warning: Could not safely evaluate db.run result string: {eval_e}. Treating as raw string.")
                    parsed_rows = [] # Fallback if parsing fails

            elif isinstance(result, list): # If db.run returns a list directly
                 parsed_rows = result

            state["query_rows"] = parsed_rows

            if not parsed_rows:
                 state["query_result"] = "Query executed successfully, but returned no results." # More informative message
                 print("SQL SELECT query executed successfully - No results.")
            else:
                 print(f"SQL SELECT query executed successfully. Rows fetched: {len(parsed_rows)}")
                 # Keep query_result as the string representation for the LLM to summarize later
        else:
            # For non-SELECT (INSERT, CREATE TABLE potentially)
            state["query_result"] = result if result else "The command executed successfully."
            print(f"SQL command executed successfully. Result: {state['query_result']}")

    except Exception as e:
        error_message = f"Error executing SQL query: {str(e)}"
        state["query_result"] = error_message
        state["sql_error"] = True
        state["query_rows"] = []
        print(error_message)
        # Increment attempts on execution failure to allow retry
        state["attempts"] = state.get("attempts", 0) + 1

    state["final_answer"] = None # Execution successful (or error handled), clear final answer field
    return state


def generate_human_readable_answer(state: AgentState):
    """Generates a natural language response based on the SQL execution outcome."""
    print("\n--- Generating Human-Readable Answer ---")
    sql = state.get("sql_query", "N/A")
    result = state.get("query_result", "N/A") # String result/error/message
    query_rows = state.get("query_rows", []) # Parsed rows, if any
    sql_error = state.get("sql_error", False) # SQL execution error flag
    last_question = state["messages"][-1].content

    # Use the prompt logic from the original code, adapted slightly
    system = """You are an assistant that converts SQL query results or errors into clear, natural language responses for the user. Use the original question and the query results/errors for context. Be concise and helpful. If the result is tabular data with many rows/columns, format it using Markdown tables."""

    if sql_error:
        human_message = f"The user asked: '{last_question}'\nI tried to run the SQL query: `{sql}`\nBut it failed with this error: {result}\n\nPlease explain this error to the user in simple terms and apologize."
    elif sql.lower().strip().startswith("select"):
        if not query_rows: # Check the parsed rows list
             human_message = f"The user asked: '{last_question}'\nI ran the SQL query: `{sql}`\nResult: {result}\n\nPlease inform the user that their query ran successfully but returned no results."
        else:
            # Pass the potentially large string result for summarization/formatting
            human_message = f"The user asked: '{last_question}'\nI ran the SQL query: `{sql}`\nResults:\n{result}\n\nPlease summarize these results clearly in natural language. If appropriate (multiple rows/columns), format the key information as a Markdown table."
            max_len = 3000 # Limit input to LLM
            if len(result) > max_len:
                 human_message = f"The user asked: '{last_question}'\nI ran the SQL query: `{sql}`\nResults (truncated):\n{result[:max_len]}...\n\nPlease summarize these (potentially truncated) results clearly in natural language. If appropriate (multiple rows/columns), format the key information as a Markdown table. Mention if results were truncated."
    else: # Non-SELECT command success
        human_message = f"The user asked: '{last_question}'\nI ran the command: `{sql}`\nResult: {result}\n\nPlease confirm to the user in natural language that their requested action was successfully completed, referencing the result message."

    generate_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human_message)])
    response_generator = generate_prompt | llm | StrOutputParser()

    try:
        answer = response_generator.invoke({})
        print(f"Generated Answer: {answer}")
        state["final_answer"] = answer
        state["messages"] = [AIMessage(content=answer)] # Append final answer to history
    except Exception as e:
        print(f"Error generating human-readable answer: {e}")
        state["final_answer"] = f"I executed the query, but had trouble summarizing the result: {e}. Raw result: {result}"
        state["messages"] = [AIMessage(content=state["final_answer"])]

    return state

def regenerate_query(state: AgentState):
    """Attempts to rewrite the question after an SQL execution error."""
    print("\n--- Regenerating Query (Attempting Rewrite) ---")
    original_question = state["question"] # Question that led to error
    sql_query = state.get("sql_query", "N/A")
    error_message = state.get("query_result", "Unknown error")
    schema = get_database_schema()
    current_attempts = state.get("attempts", 1) # Should be at least 1 if we got here
    MAX_ATTEMPTS = 3 # Define max attempts

    print(f"Attempt {current_attempts} of {MAX_ATTEMPTS}. Original question: '{original_question}'")
    print(f"Failed Query: {sql_query}")
    print(f"Error: {error_message}")

    system = f"""You are an expert SQL Server T-SQL assistant. A previous attempt to execute a generated SQL query failed.
    Database Schema:
    {schema}

    Original User Question: {original_question}
    Failed SQL Query: {sql_query}
    Error Message: {error_message}

    Your task is to analyze the error and rewrite the *original user question* to be clearer or more specific, aiming to fix the likely cause of the SQL error (e.g., ambiguous names, wrong syntax, missing joins, incorrect table refs for SQL Server). Output ONLY the rewritten question in the required JSON format. Do NOT generate SQL.
    """
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Provide the rewritten question based on the error."),
        ]
    )
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm

    try:
        rewritten_result = rewriter.invoke({})
        state["question"] = rewritten_result.question # Update question for the *next* attempt
        state["sql_error"] = False # Reset error flag for retry
        state["query_result"] = "" # Clear previous error
        state["sql_query"] = ""    # Clear previous SQL
        print(f"Rewritten question: {state['question']}")
        # DO NOT add AI message here - internal step
    except Exception as e:
        print(f"Error during question rewrite: {e}")
        state["final_answer"] = f"Sorry, I tried to fix the query but encountered an error during the rewrite process ({e}). Please try rephrasing your question."
        state["messages"] = [AIMessage(content=state["final_answer"])]
        state["attempts"] = MAX_ATTEMPTS # Force end if rewrite fails

    state["final_answer"] = None # Clear final answer field
    return state

def generate_professional_response(state: AgentState):
    """Generates a professional response for irrelevant questions."""
    print("\n--- Generating Professional Response (Irrelevant Question) ---")
    system = """You are a professional database assistant bot. The user asked something unrelated to querying or managing the database content (schema provided previously). Respond politely and professionally, stating that your function is limited to SQL-related tasks (like SELECT, CREATE TABLE, INSERT) on the connected database and you cannot help with the request."""
    user_question = state["messages"][-1].content

    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"The user asked: '{user_question}'. Explain your limitations professionally."),
        ]
    )
    response_chain = response_prompt | llm | StrOutputParser()

    try:
        message = response_chain.invoke({})
        print(f"Generated Professional Response: {message}")
        state["final_answer"] = message
        state["messages"] = [AIMessage(content=message)]
    except Exception as e:
         print(f"Error generating professional response: {e}")
         state["final_answer"] = "Sorry, I can only assist with database-related questions."
         state["messages"] = [AIMessage(content=state["final_answer"])]

    return state


def end_max_iterations(state: AgentState):
    """Handles the case where maximum regeneration attempts are reached."""
    print("\n--- Max Iterations Reached ---")
    final_message = "I'm sorry, I tried generating and fixing the SQL query multiple times but couldn't get it right after the execution errors. Could you please try rephrasing your question?"
    state["final_answer"] = final_message
    state["messages"] = [AIMessage(content=final_message)] # Append final message
    print("Maximum attempts reached. Ending the workflow.")
    return state


# --- Conditional Edge Routers ---

def relevance_router(state: AgentState):
    """Routes based on relevance."""
    print(f"Routing based on relevance: {state.get('relevance', 'not_relevant').lower()}")
    # If an error message was already set in check_relevance, end
    if state.get("final_answer"):
        return END
    if state.get('relevance', 'not_relevant').lower() == "relevant":
        return "convert_to_sql"
    else:
        return "generate_professional_response"


def execute_sql_router(state: AgentState):
    """Routes based on SQL execution success/failure and attempts."""
    MAX_ATTEMPTS = 3 # Define max attempts

    # If an error message was already set (e.g., SQL generation failed), end
    if state.get("final_answer"):
        return END

    if not state.get("sql_error", False):
        print("Routing to: generate_human_readable_answer (SQL OK)")
        return "generate_human_readable_answer"
    else:
        # Check attempts *after* error occurred
        current_attempts = state.get("attempts", 0)
        if current_attempts < MAX_ATTEMPTS:
             print(f"Routing to: regenerate_query (SQL Error, Attempt {current_attempts+1})")
             # No need to increment attempts here, it's done in execute_sql on error
             return "regenerate_query"
        else:
             print("Routing to: end_max_iterations (SQL Error, Max Attempts Reached)")
             return "end_max_iterations"

def regenerate_router(state: AgentState):
    """Routes after attempting to regenerate the query."""
    MAX_ATTEMPTS = 3
    # If regenerate_query itself failed and set a final_answer/max attempts, end
    if state.get("final_answer") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        # If attempts are maxed out here, it means rewrite failed critically
        if state.get("attempts", 0) >= MAX_ATTEMPTS and not state.get("final_answer"):
             state["final_answer"] = "Failed to rewrite the question after multiple attempts. Please rephrase."
             state["messages"] = [AIMessage(content=state["final_answer"])]
        return END
    else:
        # Go back to convert the rewritten question to SQL
        return "convert_to_sql"

# --- Build the Graph ---
print("Building the LangGraph workflow...")

# Define the checkpointer
# Use MemorySaver for demo purposes. For production, consider SqliteSaver or other persistent options.
memory = MemorySaver()

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("convert_to_sql", convert_nl_to_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("generate_professional_response", generate_professional_response)
workflow.add_node("end_max_iterations", end_max_iterations)

# Set entry point
workflow.set_entry_point("check_relevance")

# Add edges and conditional edges
workflow.add_conditional_edges(
    "check_relevance",
    relevance_router,
    {
        "convert_to_sql": "convert_to_sql",
        "generate_professional_response": "generate_professional_response",
        END: END # Route to end if relevance check itself failed and set final_answer
    },
)

workflow.add_edge("convert_to_sql", "execute_sql")

workflow.add_conditional_edges(
    "execute_sql",
    execute_sql_router,
    {
        "generate_human_readable_answer": "generate_human_readable_answer",
        "regenerate_query": "regenerate_query",
        "end_max_iterations": "end_max_iterations",
        END: END # Route to end if SQL generation failed earlier and set final_answer
    },
)

# Route after trying to regenerate
workflow.add_conditional_edges(
    "regenerate_query",
    regenerate_router, # New router after regeneration attempt
    {
        "convert_to_sql": "convert_to_sql", # Try converting the rewritten query
        END: END # End if rewrite failed or decided to give up
    }
)


# End states for final response generation nodes
workflow.add_edge("generate_human_readable_answer", END)
workflow.add_edge("generate_professional_response", END)
workflow.add_edge("end_max_iterations", END)


# Compile the graph with the checkpointer
# Use the checkpointer here
graph_app = workflow.compile(checkpointer=memory)
print("Workflow compiled successfully with MemorySaver checkpointing.")

# Expose the compiled app
__all__ = ["graph_app"] # Use a different name to avoid conflict with FastAPI app