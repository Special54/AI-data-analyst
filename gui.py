import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from llm_interface import initialize_llm, process_query
from utils import load_csv, create_text_plot, save_plot_to_file
from config import DEFAULT_API_URL, DEFAULT_MODEL_NAME, ALLOWED_FILE_TYPES, MAX_DISPLAY_ROWS

# Global variables to store data
DATAFRAME = None
DATAFRAME_NAME = None

def upload_file(file):
    """Handle file upload and reset chat."""
    global DATAFRAME, DATAFRAME_NAME
    
    if file is None:
        return "No file uploaded", gr.update(interactive=False), "", gr.update(interactive=False)
    
    df, file_name = load_csv(file)
    
    if isinstance(df, pd.DataFrame):
        DATAFRAME = df
        DATAFRAME_NAME = file_name.replace('.csv', '') if file_name.endswith('.csv') else file_name
        
        # Generate a preview of the data
        preview = f"Successfully loaded: {file_name}\n\n"
        preview += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        preview += f"First {MAX_DISPLAY_ROWS} rows:\n"
        preview += df.head(MAX_DISPLAY_ROWS).to_string()
        
        # Enable the question input and submit button
        return preview, gr.update(interactive=True), "", gr.update(interactive=True)
    else:
        return f"Error loading file: {file_name}", gr.update(interactive=False), "", gr.update(interactive=False)

def clear_outputs():
    """Clear the output components when submitting a new query."""
    return "Analyzing your data...", None

def process_query_simple(message, api_url, model_name):
    """Process a user query and return text response and image separately"""
    global DATAFRAME
    
    # Check if dataframe is loaded
    if DATAFRAME is None:
        return "Please upload a CSV file first before asking questions.", None
    
    # Initialize LLM
    llm = initialize_llm(api_base_url=api_url, model_name=model_name)
    result = process_query(message, DATAFRAME, api_url, model_name, llm)
    
    response_text = result["response"]
    plot_image = result["plot"]
    
    # Extract key results (numeric values, summaries) from the response
    result_lines = []
    for line in response_text.split("\n"):
        # Only include lines with actual results, not code or imports
        if ":" in line and not any(x in line.lower() for x in ["import ", "fig", "plt.", "ax.", "sns."]):
            result_lines.append(line.strip())
    
    # Create a response that focuses on the actual results
    display_response = ""
    
    # Add the extracted results
    if result_lines:
        display_response += "\n".join(result_lines)
    else:
        # If no specific results found, use the first line that's not an import or plot command
        found_result = False
        for line in response_text.split("\n"):
            if not any(x in line.lower() for x in ["import ", "fig", "plt.", "ax.", "#"]):
                if line.strip():
                    display_response += line.strip()
                    found_result = True
                    break
        
        if not found_result:
            display_response += "Analysis complete. Please check the visualization below."
    
    # If no plot was generated but we have text results, create a simple text plot
    if plot_image is None and display_response.strip():
        try:
            text_fig = create_text_plot(display_response)
            plot_image = save_plot_to_file(text_fig)
            plt.close(text_fig)
        except Exception as e:
            print(f"Error creating text plot: {str(e)}")
    
    # Return text response and image separately
    return display_response, plot_image

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="AI Data Analyst") as interface:
        gr.Markdown("# AI Data Analyst 📊")
        gr.Markdown("""
        Upload a CSV file and ask questions about your data in natural language.
        The system will generate and run analysis code to answer your questions.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                api_url = gr.Textbox(
                    label="API URL",
                    value=DEFAULT_API_URL,
                    placeholder="Enter the local API URL"
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    value=DEFAULT_MODEL_NAME,
                    placeholder="Enter the model name"
                )
                file_upload = gr.File(
                    label="Upload CSV File",
                    file_types=ALLOWED_FILE_TYPES
                )
                upload_button = gr.Button("Upload and Process", variant="primary")
            
            with gr.Column(scale=2):
                file_info = gr.Textbox(
                    label="File Information",
                    placeholder="CSV file details will appear here after upload...",
                    interactive=False,
                    lines=10
                )
                
        gr.Markdown("## Ask Questions About Your Data")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Type your question here",
                placeholder="What is the average age in the dataset?",
                interactive=False
            )
            
        submit_button = gr.Button("Submit Question", variant="primary", interactive=False)
            
        with gr.Row():
            response_output = gr.Markdown(
                label="Analysis Results",
                value="Results will appear here...",
            )
            
        with gr.Row():
            plot_output = gr.Image(
                label="Generated Plot (if applicable)",
                visible=True,
                show_label=True,
                container=True,
                show_download_button=True,
                height=400
            )
        
        # Set up event handlers
        upload_button.click(
            fn=upload_file,
            inputs=[file_upload],
            outputs=[file_info, query_input, query_input, submit_button]
        )
        
        # New processing flow for the simplified UI
        submit_button.click(
            fn=clear_outputs,
            outputs=[response_output, plot_output]
        ).then(
            fn=process_query_simple,
            inputs=[query_input, api_url, model_name],
            outputs=[response_output, plot_output]
        )
        
        # Also trigger on Enter key in the textbox
        query_input.submit(
            fn=clear_outputs,
            outputs=[response_output, plot_output]
        ).then(
            fn=process_query_simple,
            inputs=[query_input, api_url, model_name],
            outputs=[response_output, plot_output]
        )
    
    return interface 