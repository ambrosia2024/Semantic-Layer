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
    from FSKX_to_RDF import pipeline_app
except ImportError as import_err:
    st.error(f"Failed to import the FSKX to RDF Generator page function.")
    st.error(f"ImportError: {import_err}")
    st.error(f"Current sys.path: {sys.path}")
