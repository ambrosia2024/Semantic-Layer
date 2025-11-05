import streamlit as st
import sys
import os

# --- Path Setup ---
project_directory = os.getcwd()
if project_directory not in sys.path:
    sys.path.insert(0, project_directory)

# --- Import Application Module ---
try:
    from SKOS_Generator.skos_generator import render_skos_generator_ui
except ImportError as import_err:
    st.error(f"Failed to import the SKOS Vocabulary Generator page function.")
    st.error(f"ImportError: {import_err}")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()

# --- Render Page ---
# Call the imported function to render the UI for this page
# embedded=True prevents it from calling st.set_page_config() again
# key_ns provides a unique namespace for all widgets and session_state keys
render_skos_generator_ui(embedded=True, key_ns="skos_vocab_generator")
