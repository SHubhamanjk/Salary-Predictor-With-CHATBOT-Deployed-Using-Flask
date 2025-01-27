from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from langchain_groq import ChatGroq
import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
groq=ChatGroq(api_key=os.environ.get('GROQ_API_KEY'),model="gemma2-9b-it")
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun,WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
wiki_wrapper = WikipediaAPIWrapper(max_results=1)
search = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

system_message = """
You are a helpful and concise assistant. Your purpose is to answer user queries in a clear and actionable manner, always keeping responses relevant to the provided context. Every response must adhere strictly to the rules, without exception.
"""

rule_book = """
- Always provide responses within 3 lines, regardless of the complexity of the query. 
- Do not apply any formatting (e.g., bold, italics, bullet points) under any circumstances.
- Consider the previous context of the conversation for continuity and relevance.
- Focus on delivering answers that are actionable and precise.
- Do not use markdown or special characters (e.g., * or #), even if explicitly requested.
- If a response requires separation into paragraphs, insert blank spaces manually rather than relying on formatting tools. This ensures clarity without using formatting.
- Every response must strictly follow these rules, without exception. No deviations are allowed at any point in the conversation.
"""

example_output = """
Example:

Chat History:
User: What is Python?
Assistant: Python is a programming language known for its simplicity and versatility.

User's Current Input:
What are its main uses?

Your Response:
Python is used for web development, data analysis, machine learning, and scripting. It is also popular in automation and creating APIs.
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=(
        f"""{system_message}
{rule_book}

{example_output}

Chat History:
{{chat_history}}

User's Current Input:
{{input}}

Your Response:
"""
    )
)



agent = initialize_agent(
    tools=[search, wiki],
    llm=groq,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_executor_kwargs={"prompt": prompt_template},
    handle_parsing_errors=True
)



app = Flask(__name__)

model_filename = 'salary_predictor_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv("Dataset09-Employee-salary-prediction.csv")
job_titles = data['Job Title'].unique()

chat_history = []

@app.route('/')
def home():
    return render_template('index.html', job_titles=job_titles, prediction=None, message=None, chat_history=chat_history)

@app.route('/predict', methods=['POST'])
def predict_salary():
    age = int(request.form['age'])
    education_level = request.form['education_level']
    years_of_experience = int(request.form['experience'])
    job_title = request.form['job_title']

    education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
    education_level = education_map.get(education_level, -1)

    input_data = np.zeros(len(model.feature_names_in_))
    input_data[0] = age
    input_data[1] = education_level
    input_data[2] = years_of_experience

    job_title_col = f"jobtitle_{job_title.replace(' ', '').lower()}"
    if job_title_col in model.feature_names_in_:
        job_title_index = list(model.feature_names_in_).index(job_title_col)
        input_data[job_title_index] = 1

    predicted_salary = model.predict([input_data])[0]

    return render_template(
        'index.html',
        job_titles=job_titles,
        prediction=f"${predicted_salary:,.2f}",
        message=None,
        chat_history=chat_history
    )
@app.route('/chat', methods=["POST"])
def chat():
    global chat_history

    user_message = request.form["msg"]
    chat_history.append({"sender": "user", "message": user_message})

    try:
        bot_response = agent.run(user_message)

        if hasattr(bot_response, "content"):
            bot_response_text = bot_response.content
        else:
            bot_response_text = str(bot_response)

    except Exception as e:
        bot_response_text = f"Error: {str(e)}"

    chat_history.append({"sender": "bot", "message": bot_response_text})

    return bot_response_text


if __name__ == '__main__':
    app.run(debug=True)
