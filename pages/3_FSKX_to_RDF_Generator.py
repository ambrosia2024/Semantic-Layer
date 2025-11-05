import streamlit as st
import sys
import os

# --- Path Setup ---
# Use the Current Working Directory as the project root.
# This assumes `streamlit run Home.py` is executed from the project root.
project_directory = os.getcwd()
if project_directory not in sys.path:
    sys.path.insert(0, project_directory)

# --- Import Application Module ---
try:
    # Now we can import the app using an absolute path from the project root
    from FSKX_to_RDF.pipeline_app import render_fskx_to_rdf_ui
except ImportError as import_err:
    st.error(f"Failed to import the FSKX to RDF Generator page function.")
    st.error(f"ImportError: {import_err}")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()

# --- Render Page ---
# Call the imported function to render the UI for this page
# embedded=True prevents it from calling st.set_page_config() again
# key_ns provides a unique namespace for all widgets and session_state keys
render_fskx_to_rdf_ui(embedded=True, key_ns="fskx_to_rdf_generator")
