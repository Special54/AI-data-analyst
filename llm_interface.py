import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive mode
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from pydantic import SecretStr
from typing import Dict, List, Optional, Any

from utils import get_dataframe_info, save_plot_to_file
from config import DEFAULT_API_URL, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, DEFAULT_FIGURE_SIZE

def initialize_llm(api_base_url: str = DEFAULT_API_URL, 
                  model_name: str = DEFAULT_MODEL_NAME, 
                  temperature: float = DEFAULT_TEMPERATURE):
    """Initialize the LLM using LangChain with the local API."""
    try:
        llm = ChatOpenAI(
            streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            api_key=SecretStr("dummy-key"),  # Not used but required
            base_url=api_base_url,
            model=model_name,
            temperature=temperature,
            verbose=True,
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return None

def generate_code_for_query(query: str, df_name: str, df_info: str, llm) -> str:
    """Generate Python code to analyze the dataset based on the query."""
    system_message = f"""
    You are a data analysis expert. Given a question about a dataset, generate Python code to answer the question.
    
    The dataframe is stored in a variable named '{df_name}'. Here's information about the dataframe:
    
    {df_info}
    
    Generate ONLY Python code to answer the question. The code should:
    1. Use pandas, numpy, matplotlib, and/or seaborn
    2. Include comments explaining key steps
    3. Store any plots in a variable named 'fig' 
    4. Set the title of any plots to reflect the analysis
    5. IMPORTANT: Never call plt.show() as it will cause errors in the non-interactive environment
    6. For bar plots, use plt.tight_layout() at the end to ensure proper display
    7. Make sure to set appropriate figure size with figsize={DEFAULT_FIGURE_SIZE} or similar
    8. Return both textual results and any generated plots
    9. Be efficient and readable
    
    DO NOT include any markdown, explanations, or non-code content. ONLY return valid Python code.
    """
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query)
    ]
    
    try:
        response = llm.predict_messages(messages)
        # Extract just the code part (remove any markdown formatting if present)
        code = response.content.strip()
        
        # Handle various code block formats
        if "```python" in code:
            code = code.split("```python", 1)[1]
        elif "```py" in code:
            code = code.split("```py", 1)[1]
        elif "```" in code:
            code = code.split("```", 1)[1]
        
        if "```" in code:
            code = code.split("```", 1)[0]
            
        return code.strip()
    except Exception as e:
        return f"# Error generating code: {str(e)}\nraise Exception('Failed to generate analysis code')"

def execute_code_safely(code: str, df) -> tuple:
    """Execute the generated code safely and capture the output and plots."""
    # Create a new temporary namespace for execution
    local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    
    result_text = ""
    fig = None
    
    # Create a string buffer to capture print output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Redirect stdout and stderr
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    
    try:
        # Make sure we close any existing plots first
        plt.close('all')
        
        # Execute the code
        exec(code, {"__builtins__": __builtins__}, local_vars)
        
        # Capture print outputs
        stdout_output = stdout_buffer.getvalue()
        if stdout_output.strip():
            result_text += stdout_output
        
        # Check if there's a figure in the local variables
        if 'fig' in local_vars and hasattr(local_vars['fig'], 'figure'):
            fig = local_vars['fig']
        elif plt.get_fignums():
            # If fig not explicitly defined but plots were created
            fig = plt.gcf()
            
        # Extract any variables that might contain the answer
        for var_name, var_value in local_vars.items():
            if var_name not in ["df", "pd", "np", "plt", "sns", "fig"] and not var_name.startswith("_"):
                if isinstance(var_value, (pd.DataFrame, pd.Series)):
                    if var_value.shape[0] <= 5:  # Only show if small
                        result_text += f"{var_name}:\n{var_value.to_string()}\n"
                elif isinstance(var_value, (str, int, float, np.number)) and not callable(var_value):
                    # Format numeric values nicely
                    if isinstance(var_value, (float, np.floating)):
                        result_text += f"{var_name}: {float(var_value):.2f}\n"
                    else:
                        result_text += f"{var_name}: {var_value}\n"
    
    except Exception as e:
        result_text += f"Error executing code: {str(e)}\n"
        traceback_str = traceback.format_exc()
        # Only add the first few lines of the traceback to keep the output clean
        result_text += '\n'.join(traceback_str.split('\n')[:5])
    
    finally:
        # Restore stdout and stderr
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
        
        # Add any error outputs
        error_output = stderr_buffer.getvalue()
        if error_output.strip():
            # Filter out common warnings that don't affect the results
            filtered_error_lines = []
            for line in error_output.split('\n'):
                if line.strip() and not "FigureCanvasAgg" in line:
                    filtered_error_lines.append(line)
            
            if filtered_error_lines:
                result_text += f"\nError output:\n" + '\n'.join(filtered_error_lines[:3])
    
    return result_text.strip(), fig

def process_query(query: str, df, api_url: str = DEFAULT_API_URL, 
                 model_name: str = DEFAULT_MODEL_NAME, llm=None) -> Dict[str, Any]:
    """Process a user query about the dataset."""
    if df is None:
        return {
            "response": "Please upload a CSV file first.",
            "code": "",
            "plot": None
        }
    
    # Initialize LLM if not already done
    if llm is None:
        llm = initialize_llm(api_base_url=api_url, model_name=model_name)
        if llm is None:
            return {
                "response": "Failed to initialize the language model. Check your LM Studio setup.",
                "code": "",
                "plot": None
            }
    
    # Get dataframe info
    df_info = get_dataframe_info(df)
    
    # Generate code based on the query
    code = generate_code_for_query(query, "df", df_info, llm)
    
    # Execute the code
    result, fig = execute_code_safely(code, df)
    
    # Save the plot to a file if it exists
    plot_path = None
    if fig is not None:
        try:
            plot_path = save_plot_to_file(fig)
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing plot: {str(e)}")
    
    return {
        "response": result,
        "code": code,
        "plot": plot_path
    } 