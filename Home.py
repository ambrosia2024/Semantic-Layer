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
This tool helps you create and manage SKOS (Simple Knowledge Organization System) vocabularies. You can define concepts, establish relationships between them, and export the result as a Turtle (.ttl) file. 
""")

st.subheader("3. FSKX to RDF Generator")
st.markdown("""
This tool transforms FSKX models into rich, semantic RDF representations. It enriches the model's metadata using the FSKX Ontology, creating a solid foundation for linked data. The user-friendly interface allows for the manual mapping of hazards, products, and climate variables to input parameters. All mappings and linkages are saved semantically, making them readily available for downstream applications and analysis.
""")
