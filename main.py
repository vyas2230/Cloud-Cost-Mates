from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, LLM
import pandas as pd
from flask_cors import CORS

# Load environment variables
load_dotenv()
# Watsonx Credentials
project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_key = os.getenv("WATSONX_APIKEY")
watsonx_url = os.getenv("WATSONX_URL")
tavily_key = os.getenv("TAVILY_API_KEY")

# Initialize Watsonx LLM
llm = LLM(
    model="watsonx/ibm/granite-3-2b-instruct",
    max_tokens=10000,  # Allow larger responses
    temperature=0.5,
    api_key=watsonx_key,
    watsonx_url=watsonx_url,
    project_id=project_id,
    tavily_key=tavily_key,
)

# Load CSV (Modify the file path if needed)
DATASET_PATH = "data.csv"
df = pd.read_csv(DATASET_PATH, sep="\t")  # `sep='\t'` for TSV files
data_content = df.head(50).to_json(orient="records", lines=True)  # Convert first 50 rows to JSON

# Create Flask App
app = Flask(__name__)

# Create CrewAI Agent for dataset analysis
csv_reader_agent = Agent(
    role="Data Analyst",
    goal="Analyze and answer questions based on the full dataset.",
    backstory="An AI-powered data analyst that works with complete datasets.",
    llm=llm
)

# Task for dataset analysis
csv_analysis_task = Task(
    description=f"Analyze this dataset:\n{data_content}\n\nExtract insights and prepare for user queries.",
    expected_output="A structured JSON of the dataset.",
    agent=csv_reader_agent
)

# Run CrewAI Processing
crew = Crew(agents=[csv_reader_agent], tasks=[csv_analysis_task])
full_dataset_result = crew.kickoff()  # This processes the dataset

@app.route("/query", methods=["POST"])
def ask_watsonx():
    """API endpoint to ask questions about the dataset."""
    data = request.get_json()
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "Missing 'question' field"}), 400

    # Create a prompt with the full dataset analysis
    prompt = f"Based on this dataset:\n{full_dataset_result}\n\nAnswer the question: {user_question}"

    # Get response from Watsonx LLM
    response = llm.call(prompt)

    return jsonify({"question": user_question, "answer": response})


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
    CORS(app)

