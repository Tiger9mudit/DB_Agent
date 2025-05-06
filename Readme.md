# SQL AI Agent

This project is a SQL AI Agent that allows users to interact with a database using natural language queries. The agent converts these queries into SQL commands and provides results in a user-friendly chat interface.

## Features
- Natural language to SQL conversion.
- Real-time streaming responses.
- Interactive chat interface with Bootstrap styling.
- Support for SQL Server database connections.

---

## Project Setup

### Prerequisites
1. **Python**: Ensure Python 3.9 or higher is installed.
2. **Node.js**: Required for frontend dependencies (optional if using CDN).
3. **SQL Server**: A running SQL Server instance.
4. **ODBC Driver**: Install the [ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

### Installation
1. Clone the repository:

   git clone https://github.com/your-username/DB_Agent_v2.git
   cd DB_Agent_v2

2. Create a virtual environment and activate it:
    python -m venv venv
    venv\Scripts\activate  # On Windows
3. Install dependencies:
    pip install -r requirements.txt

4. Environment File Setup:

    Create a .env file in the root directory with the following variables:
    GOOGLE_API_KEY = "xxxxx"
    SQL_SERVER="xxxxx"
    SQL_DATABASE="xxxxxx"
    SQL_USERNAME="xxxxxxxx"
    SQL_PASSWORD="xxxxxxxx"

5. Running the Project

    Start the backend server:
        uvicorn main:app --reload
    Open the frontend:
        Navigate to http://127.0.0.1:8000 in your browser.