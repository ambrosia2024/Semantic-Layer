import streamlit as st

st.set_page_config(
    page_title="AMBROSIA Semantic Layer Tools",
    page_icon="-",
    layout="wide"
)

st.title("Welcome to the AMBROSIA Semantic Layer Tools")

st.markdown("""
This application provides a suite of tools to build and manage the AMBROSIA semantic layer. 
Each tool is designed to handle a specific part of the semantic workflow, from reconciling terms to generating linked data from models.

Please select a tool from the sidebar to get started.
""")

st.header("Available Tools")

st.subheader("1. Reconciliation Service")
st.markdown("""
The Reconciliation Service allows you to map terms from your datasets to established ontologies and vocabularies like AGROVOC, NCBI, and Wikidata. This process enriches your data by linking it to standardized concepts, improving interoperability and data discovery.
""")

st.subheader("2. SKOS Vocabulary Generator")
st.markdown("""
This tool helps you create and manage SKOS (Simple Knowledge Organization System) vocabularies. You can define concepts, establish relationships between them (such as broader, narrower, and related), and export the result as a Turtle (.ttl) file. This is essential for creating controlled vocabularies for your domain.
""")

st.subheader("3. FSKX to RDF Generator")
st.markdown("""
The FSKX to RDF Generator converts models described in the FSKX format into RDF graphs. It maps model parameters, hazards, and products to your SKOS vocabularies and generates linked data descriptions of the models, including wiring information for connecting model inputs to data sources.
""")
