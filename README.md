# AI Data Analyst

A locally hosted AI Data Analyst tool that allows you to query CSV datasets using natural language. It uses a local LLM via LM Studio to generate and execute Python code for data analysis.

## Features

- Upload CSV files for analysis
- Query your data using natural language
- Generate Python code for data analysis automatically
- Visualize results with plots
- All computation happens locally (no data sent to external services)

## Requirements

- Python 3.10+
- LM Studio installed and running locally (serving a model at http://localhost:1234/v1)

## Installation

1. Clone this repository
```bash
git clone <repository-url>
cd AI-data-analyst
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have LM Studio running with a model served at http://localhost:1234/v1
   - Download LM Studio from https://lmstudio.ai/
   - Start the local server with your chosen model (recommended: llama-3-8b-instruct)

2. Run the application
```bash
python main.py
```

3. Open your browser at the URL shown in the terminal (typically http://127.0.0.1:7860)

4. Upload a CSV file using the interface

5. Ask questions about your data in natural language, for example:
   - "What's the average salary by department?"
   - "Show me a bar chart of sales by region"
   - "What's the correlation between age and income?"

## Configuration

You can configure the following settings in the UI:
- API URL: The endpoint for your local LLM server
- Model Name: The name of the model you're using in LM Studio

## Troubleshooting

- If you get connection errors, make sure LM Studio is running and serving a model
- For memory issues with large datasets, consider using a subset of your data
- Make sure your CSV file is properly formatted 