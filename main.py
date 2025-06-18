from gui import create_gradio_interface

def main():
    """Main function to run the application."""
    interface = create_gradio_interface()
    interface.launch(debug=True)

if __name__ == "__main__":
    main() 