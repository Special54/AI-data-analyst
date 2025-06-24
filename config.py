"""
Configuration settings for the AI Data Analyst application.
"""

# Default LLM settings
DEFAULT_API_URL = "http://localhost:1234/v1"
DEFAULT_MODEL_NAME = "llama-3-8b-instruct"
DEFAULT_TEMPERATURE = 0.5

# Matplotlib configuration
MATPLOTLIB_STYLE = "ggplot"
SEABORN_STYLE = "whitegrid"
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 100

# File handling
ALLOWED_FILE_TYPES = [".csv"]
MAX_DISPLAY_ROWS = 5
MAX_UNIQUE_VALUES_DISPLAY = 10 