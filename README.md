# Stock-Analysis-Application
This application provides stock price analysis and prediction using historical data and LSTM models.

# Prerequisites
Python 3.7 or higher
Node.js and npm (for React components)

# Step-by-step Instructions
Clone the repository:

git clone <repository-url>
cd <repository-directory>
Set up a Python virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Python dependencies:

pip install -r requirements.txt
Install Node.js dependencies:

npm install
Train the initial model (optional, as the app can create a model if none exists):

python train_model.py
Run the Streamlit app:

streamlit run app.py
Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

In the app interface:

Enter a stock symbol (e.g., AAPL for Apple Inc.)
Select a date range for analysis
Click "Analyze and Predict" to view the results
Troubleshooting
If you encounter any issues:

Ensure all dependencies are correctly installed
Check that you're using compatible versions of Python and the required libraries
Verify that you have an active internet connection for fetching stock data
For any persistent problems, please open an issue in the repository.
