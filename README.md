# Salary Predictor and Chatbot Deployed on Flask

This project is a web application that allows users to predict their expected salary based on various input parameters such as age, education level, years of experience, and job title. It also includes an interactive chatbot that uses AI to answer user questions.

## Features
- **Salary Prediction**: Predicts expected salary based on user inputs.
- **Interactive Chatbot**: Engages users with intelligent responses powered by the Groq LLM.
- **User-Friendly Interface**: Allows users to interact seamlessly with both the salary predictor and the chatbot.

## Technologies Used
- **Flask**: For building the web application.
- **Pickle**: For loading the pre-trained salary prediction model.
- **Groq LLM**: For generating responses in the chatbot.
- **HTML/CSS/JavaScript**: For front-end development.
- **Select2**: For enhancing the dropdown input for job titles.

## Prerequisites
Before you begin, make sure you have the following installed:
- Python 3.7 or higher
- Flask
- Pickle
- pandas
- numpy
- requests
- Groq API key

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SHubhamanjk/Salary-Predictor-With-CHATBOT-Deployed-Using-Flask
    cd Salary-Predictor-With-CHATBOT-Deployed-Using-Flask
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add your Groq API key to the `.env` file:
      ```
      GROQ_API_KEY=your_groq_api_key_here
      ```

5. **Run the application**:
    ```bash
    python app.py
    ```

   This will start the Flask server. You can access the web application by navigating to `http://127.0.0.1:5000/` in your browser.

## Usage

1. **Predict Salary**:
   - Enter the age, education level, years of experience, and job title in the form.
   - Click "Predict Salary" to see the expected salary for the given parameters.

2. **Chat with the AI Chatbot**:
   - Enter a question in the chatbot input field and press "Send" to get a response from the AI-powered chatbot.

## Files in the Project
- **app.py**: Main Python file where the application is built using Flask.
- **templates/index.html**: The HTML template for the web application interface.
- **static/style.css**: CSS file for styling the web pages.
- **Dataset09-Employee-salary-prediction.csv**: Dataset used for salary prediction.
- **salary_predictor_model.pkl**: Pre-trained salary prediction model (saved using Pickle).
- **.gitignore**: Excludes sensitive files such as `.env` from being pushed to GitHub.

## Acknowledgements
- Groq API for providing powerful AI responses.
- Flask for making it easy to build web applications.
- Select2 for enhancing the user experience with dropdown inputs.

