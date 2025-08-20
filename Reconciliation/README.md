# CSV Reconciliation Tool (PoC)

## Overview

This application is a Proof-of-Concept (PoC) tool built with Streamlit designed to help users reconcile terms from a CSV file against various external knowledge bases and controlled vocabularies. Users can upload a CSV, select data providers (like Wikidata, NCBI, BioPortal, etc., including a custom SPARQL endpoint), fetch potential matches (suggestions) for terms lacking URIs, review these suggestions using different ranking strategies, and select the appropriate URI to enrich their data. The final reconciled data can then be downloaded.

## Features

*   **CSV Upload:** Upload a CSV file containing terms to be reconciled.
*   **Required Columns:** Expects specific columns: `Term`, `URI`, `RDF Role`, `Match Type`.
*   **Multiple Data Providers:** Supports reconciliation against:
    *   Wikidata
    *   NCBI (Taxonomy, BioProject, Gene, etc.)
    *   BioPortal
    *   OLS (Ontology Lookup Service @ EBI)
    *   NCI Thesaurus
    *   GeoNames
    *   AGROVOC
    *   Custom SPARQL Endpoint
*   **Processing Queue:** Add selected providers to a queue for sequential processing.
*   **Suggestion Fetching:** Automatically fetches potential URI matches for terms with an empty 'URI' field.
*   **Matching/Ranking Strategies:** View and sort suggestions based on:
    *   **API Ranking:** The default order returned by the provider's API.
    *   **Levenshtein Similarity:** Sorts suggestions based on string edit distance between the input term and the suggestion label.
    *   **Cosine Similarity:** Sorts suggestions based on semantic similarity between the input term and the suggestion's label/description (requires a sentence-transformer model).
*   **Interactive Selection:** Review suggestions term-by-term for each provider and select the correct URI or "No Match".
*   **Dynamic Updates:** The main data table updates immediately upon selection.
*   **Progress Monitoring:** View the status and progress of the processing queue and term lookups in the sidebar.
*   **Custom SPARQL Configuration:** Define your own SPARQL endpoint and query template for custom data sources.
*   **Configuration File:** Manage API keys, usernames, and service endpoints via `config.yaml`.
*   **Download Results:** Download the updated CSV file with the reconciled URIs.

## Prerequisites

*   Python 3.10.16
*   pip (Python package installer)

## Setup Instructions

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Install Dependencies:** Place the file named `requirements.txt` in the same directory with the following content:

    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `sentence-transformers` will download model files on first use if the Cosine Similarity strategy is selected.*

3.  **Configure `config.yaml`:** Place the file named `config.yaml` in the same directory and **fill in your own API keys, usernames, and email address** where indicated (`YourAPIKey`, `YourUsername`, `your-email@example.com`).

    **Warning:** Do not commit your API keys or sensitive information to public repositories. Use environment variables or other secure methods for production deployments.

## Running the Application

1.  Navigate to the application directory in your terminal (where `reconciliation_app.py` is located).
2.  Make sure your (virtual) environment is activated.
3.  Run the Streamlit application:
    ```bash
    streamlit run reconciliation_app.py
    ```
4.  The application should open automatically in your web browser.

## Usage Guide

1.  **Upload CSV:**
    *   Click the "Browse files" button under "1. Upload CSV File".
    *   Select your CSV file. It **must** contain columns named `Term`, `URI`, `RDF Role`, and `Match Type`. UTF-8 encoding is recommended.
    *   The application will parse the CSV and display the data in the "Current Data" section. It identifies terms where the `URI` column is empty – these are the terms that need reconciliation.

2.  **Configure in Sidebar:** Use the sidebar on the left for configuration and control:
    *   **Reconciliation Sources:** Check the boxes next to the data providers (Wikidata, NCBI, etc.) you want to use for finding matches. Tooltips provide a brief description of each provider.
    *   **Custom SPARQL Provider:**
        *   If you want to use your own SPARQL endpoint, check "Enable Custom SPARQL Provider".
        *   Enter the full SPARQL Endpoint URL.
        *   Provide the SPARQL Query Template. Use `{term}` as a placeholder for the search term and `{limit}` for the maximum number of results. Ensure your query selects variables for URI, Label, and optionally Description.
        *   Specify the exact variable names used in your query (defaults are `uri`, `label`, `description`).
        *   If enabled, select "Custom SPARQL" from the Reconciliation Sources list.
    *   **Add to Queue:** Click the "Add / Re-queue Selected" button to add the checked providers to the processing queue. The queue status will update below. Re-adding a provider will clear its previous results and queue it again.
    *   **Matching Strategy:** Select how suggestions should be ranked/sorted when displayed:
        *   `API Ranking`: Default order from the source.
        *   `Levenshtein Similarity`: Best string match first.
        *   `Cosine Similarity`: Best semantic match first (requires `semantic_search.py` and downloads a model on first use).
    *   **Query Settings:** Adjust the "Max Suggestions per Term" slider to control how many potential matches are requested from each provider for each term.
    *   **Process Control:**
        *   Once providers are added to the queue and configuration is complete (including required API keys/usernames in `config.yaml`), click "Start Processing Queue".
        *   The button will be disabled if prerequisites are missing (e.g., empty queue, missing config for a selected provider, model unavailable for Cosine Similarity).
        *   Progress bars for provider processing and term lookup will update in the sidebar. Processing happens sequentially through the queue.

3.  **View Provider Results:**
    *   After processing starts/completes/errors, buttons representing each processed provider will appear under the "View Provider Results" section in the main area.
    *   Icons indicate status (✅ Completed, ❌ Error, ⚙️ Running, 🛑 Stopped).
    *   Click a provider button to load its suggestions into the "Select Suggestions" area below.

4.  **Select Suggestions:**
    *   This section appears after you click a provider button. It lists each term needing reconciliation.
    *   For each term, a dropdown menu shows the suggestions found by the selected provider, formatted and ranked according to the chosen "Matching Strategy".
    *   The score (Levenshtein or Cosine) is shown in brackets `[...]` if applicable for the strategy.
    *   Select the most appropriate URI from the dropdown.
    *   Choose `--- No Match ---` if none of the suggestions are correct.
    *   The description of the selected suggestion (if available) is shown below the dropdown.
    *   Your selection immediately updates the `URI` column in the main "Current Data" table.

5.  **Download Reconciled Data:**
    *   Once you have finished reviewing and selecting URIs, scroll down to the "Download Reconciled Data" section.
    *   Click the "Download 'filename_reconciled.csv'" button to save the updated CSV file. `No Match` selections will be saved as empty strings in the 'URI' column.

## Input CSV Format

Your input CSV file **must** contain the following columns:

*   `Term`: The term/label/string you want to reconcile.
*   `URI`: The corresponding URI for the term. Leave this **empty** for terms that need reconciliation. The tool will fill this column based on your selections.
*   `RDF Role`: (Contextual) An RDF property or role associated with the term (e.g., `predicate`, `object`). This column is currently informational for the user but not directly used in the matching logic.
*   `Match Type`: (Contextual) The type of match expected or desired (e.g., `Exact`, `Close`). This column is currently informational for the user but not directly used in the matching logic.

Troubleshooting / Notes

    API Keys/Usernames: Ensure you have correctly entered valid API keys/usernames in config.yaml for the providers you intend to use (NCBI, BioPortal, GeoNames). Also ensure the email for NCBI is provided.

    Missing Modules: If you get ImportError for processing_service or semantic_search, make sure the .py files are present in the same directory as app.py.

    Dependencies: Ensure all packages in requirements.txt are installed in your active Python environment.

    Cosine Similarity Model: The first time you select the "Cosine Similarity" strategy and start processing, the application will download the specified sentence transformer model (all-MiniLM-L6-v2 by default). This requires an internet connection and may take some time.

    Rate Limits: Be mindful of API rate limits imposed by external providers. Processing large files quickly might hit these limits, causing errors. The tool processes providers and terms sequentially.

    Custom SPARQL: Double-check your endpoint URL and SPARQL query syntax. Ensure the query template correctly uses {term} and {limit} and that the variable names match those specified in the sidebar settings.

    CSV Parsing: If the CSV fails to load, check its encoding (UTF-8 recommended) and delimiter (the app tries common ones like comma, semicolon, tab).