# FSKXtoRDF

A comprehensive toolkit for converting food safety knowledge (FSK) models from `.fskx` format into various Linked Data (RDF) serializations.

The project is based on ZBMED's fskx-linked-data project: https://github.com/zbmed/fskx-linked-data

## Overview

FSKXtoRDF provides a user-friendly Streamlit application to process `.fskx` files located in a local subfolder. The toolkit transforms model metadata into JSON-LD, validates it, maps it to the FSKX Ontology (FSKXO), and provides an interface for further semantic enrichment, such as linking model parameters to climate variables.

## Features

-   **Streamlit Interface**: An interactive web application (`pipeline_app.py`) to manage and run the entire conversion pipeline.
-   **Local File Processing**: Processes `.fskx` models from the local `./fskx_models` directory.
-   **Automated Pipeline**: A one-click pipeline that performs a sequence of conversion and validation steps.
-   **Interactive Mapping**: A user interface to manually map model parameters to concepts from controlled vocabularies, including climate data.
-   **Joined Model Support**: Intelligently handles complex `.fskx` archives containing multiple interconnected models.
-   **Multi-Format Export**: Converts the final mapped data into both JSON-LD and Turtle formats.

## Workflow

The primary way to use this toolkit is through the Streamlit application.

### 1. Launch the Application

Run the following command in your terminal:

```bash
streamlit run pipeline_app.py
```

### 2. Run the Pipeline

The Streamlit app provides a "Run Pipeline" button that executes the following steps automatically:

1.  **FSKX to JSON-LD (`FSKX_to_JSONLD.py`)**: Extracts metadata from each `metaData.json` file within the `.fskx` archives and converts it into an initial, unmapped JSON-LD file.
2.  **Validation (`ValidityChecker.py`)**: Checks the generated JSON-LD for semantic correctness and ensures it is valid Linked Data.
3.  **Automatic Mapping (`run_mapper.py`)**: Performs an automated mapping of terms within the JSON-LD files to entities from the FSKX Ontology (FSKXO).
4.  **Turtle Conversion (`jsonld_serialization_converter.py`)**: Converts the final mapped JSON-LD output into the Turtle (`.ttl`) RDF format.

### 3. Interactive Mapping

After the pipeline has run, the Streamlit application allows for further manual enrichment:

-   Select a model from the dropdown menu.
-   The interface will display the model's parameters (inputs and outputs), hazards, and products.
-   Users can map these terms to concepts from various controlled vocabularies, including climate-related variables like temperature and humidity.
-   When mappings are applied, the script saves these new semantic links directly into the final Turtle file for that model.

## Architecture

### IRI Generation Strategy

The system generates IRIs using the following hierarchy:

1. **Ontology-based IRIs**: For controlled vocabulary terms (hazards, products, units, etc.), the system searches the fskxo.owl ontology using:
   - Exact label matching
   - Synonym matching (via oboInOwl:hasSynonym)
   - Token-based matching
   - Prefix matching
   - Partial matching (Jaccard similarity)
   - Fuzzy matching (configurable threshold, default: 0.96)

2. **Semantic IRIs**: For named entities (creators, references, parameters), the system generates descriptive IRIs based on:
   - DOI for references
   - Parameter IDs for model parameters
   - Hash-based IDs for privacy-sensitive data (creators, authors)

3. **Fallback IRIs**: When no ontology match is found, generates IRIs based on sanitized names or content hashes

### Base URI

All generated IRIs use the base URI: `http://semanticlookup.zbmed.de/km/fskxo/`

### Matching Configuration

Fine-tune matching behavior in `FSKX_to_JSONLD.py` (lines 48-57):

```python
FUZZY_ENABLED = True              # Enable/disable fuzzy matching
FUZZY_MIN_SCORE = 0.96            # Fuzzy match threshold (0-1)
FUZZY_DISABLE_BELOW_LEN = 10      # Min chars for fuzzy matching
PARTIAL_MIN_TOKEN_OVERLAP = 0.8   # Jaccard overlap threshold
PREFIX_MIN_CHARS = 6              # Min chars for prefix matching
```

### Vocabulary Mapping

The script handles mapping of:

- **Units**: unit, unitCategory (e.g., "celsius", "log", "CFU/g")
- **Classifications**: classification, modelClass, modelSubClass
- **Parameters**: parameterType, parameterClassification
- **Organisms**: hazard, product, populationGroup (e.g., "Salmonella", "Listeria monocytogenes")
- **General**: language, software, format, publicationType

Mappings are cached in `vocab_mappings.pkl` for performance.

### Handling Joined (Composite) FSKX Models

The script intelligently handles complex `.fskx` archives containing multiple interconnected models.

-   **Detection**: A joined model is identified by the presence of multiple `metaData.json` files or specific `<comp:listOfSubmodels>` tags within the `manifest.xml`.
-   **Hierarchy Parsing**: The script parses the main SBML file to understand the relationships between sub-models and the links between their parameters as defined in `<comp:replacedBy>` tags.
-   **Robust Discovery**: It cross-references this information with the directory structure to ensure all models are discovered, even if they are not explicitly defined in the main SBML.
-   **Parameter Linking**: The script creates explicit `wasInformedBy` links between parameters by inferring the connections from the SBML structure. It intelligently matches parameter IDs (e.g., linking an output like `"dose12"` to an input named `"dose"`) to build a complete, machine-readable graph of the model system.

## Output Structure

### JSON-LD Files

Each generated JSON-LD file contains a top-level object representing the model. For composite models, this object contains a `hasPart` array listing the individual sub-models.

```json
{
  "@context": { ... },
  "@id": "http://semanticlookup.zbmed.de/km/fskxo/CompositeModel/threejoinedModels",
  "@type": "fskx:CompositeModel",
  "hasPart": [
    {
      "@id": "fskx:Model/ModelA",
      "@type": "fskx:PredictiveModel",
      ...
    },
    {
      "@id": "fskx:Model/ModelB",
      "@type": "fskx:PredictiveModel",
      ...
    }
  ]
}
```

### Controlled Vocabulary and Parameter Mappings

Vocabulary fields are mapped to ontology IRIs, and parameter connections are represented using the `wasInformedBy` property.

```json
{
  "unit": {
    "@id": "http://semanticlookup.zbmed.de/km/fskxo/FSKXO_0000123",
    "label": "celsius"
  },
  "parameter": [
    {
      "@id": "fskx:ModelB/parameter/dose",
      "id": "dose",
      "name": "Dose of Contaminant",
      "wasInformedBy": {
        "@id": "fskx:ModelA/parameter/dose_output"
      }
    }
  ]
}
```

## Installation

### Dependencies

- Python 3.10.16
- streamlit
- pandas
- rdflib
- openpyxl
- pyld

Install the required packages using pip:
```bash
pip install streamlit pandas rdflib openpyxl pyld
```

## Core Scripts

-   **`pipeline_app.py`**: The main Streamlit application for running the pipeline and interactive mapping.
-   **`FSKX_to_JSONLD.py`**: Handles the core extraction and conversion of `.fskx` files to JSON-LD, including the logic for joined models.
-   **`ValidityChecker.py`**: Validates the semantic structure of the generated JSON-LD files.
-   **`run_mapper.py`**: Orchestrates the automatic mapping of terms to the FSKX Ontology.
-   **`jsonld_serialization_converter.py`**: Converts the final JSON-LD files to Turtle format.
-   **`robust_metadata_matching.py`**: Contains the logic for intelligently matching models to their metadata within a complex archive.

## Credits

-   **Taras Günther (taras.guenther@bfr.bund.de)**: Provided the initial scripts for single model conversion, validation, serialization, and the FSKXO mapper.
-   **Julian Schneider (julian.schneider@bfr.bund.de)**: Provided the JSON-LD conversion framework.
