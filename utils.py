import os
import io
import pandas as pd
import numpy as np
import tempfile
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    MATPLOTLIB_STYLE, 
    SEABORN_STYLE, 
    DEFAULT_FIGURE_SIZE,
    DEFAULT_DPI,
    MAX_DISPLAY_ROWS,
    MAX_UNIQUE_VALUES_DISPLAY
)

# Configure the default plotting style
plt.style.use(MATPLOTLIB_STYLE)
sns.set(style=SEABORN_STYLE)

def load_csv(file) -> tuple:
    """Load a CSV file and return the dataframe."""
    try:
        df = pd.read_csv(file.name)
        return df, os.path.basename(file.name)
    except Exception as e:
        return None, str(e)

def get_dataframe_info(df) -> str:
    """Generate a description of the dataframe."""
    buffer = io.StringIO()
    
    # Basic dataframe information
    buffer.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
    
    # Column information
    buffer.write("Column Information:\n")
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        
        buffer.write(f"- {col} (Type: {dtype})\n")
        buffer.write(f"  Unique values: {n_unique}, Missing values: {n_missing}\n")
        
        # For categorical/object columns with few unique values, show them
        if df[col].dtype == 'object' and n_unique <= MAX_UNIQUE_VALUES_DISPLAY:
            buffer.write(f"  Unique values: {', '.join(map(str, df[col].unique()))}\n")
        # For numeric columns, show some statistics
        elif np.issubdtype(df[col].dtype, np.number):
            buffer.write(f"  Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}\n")
    
    return buffer.getvalue()

def save_plot_to_file(fig):
    """Save a matplotlib figure to a file and return the path."""
    if fig is None:
        return None
    
    try:
        # Create a temporary file with a unique name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.close()
        
        # Save the figure with high quality
        fig.savefig(temp_file.name, format='png', bbox_inches='tight', dpi=DEFAULT_DPI)
        
        # Return the path to the saved image
        return temp_file.name
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
        return None

def create_text_plot(text):
    """Create a simple plot with text results when no visualization is available."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # Hide axes
    ax.axis('off')
    
    # Add the text
    ax.text(0.5, 0.5, text, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14,
            transform=ax.transAxes,
            wrap=True)
    
    # Tight layout
    plt.tight_layout()
    
    return fig 