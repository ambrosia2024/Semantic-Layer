import streamlit as st
import pandas as pd
import re
import io
import rdflib
import json # Import the json module
from .prefix_manager import extract_prefixes, find_relevant_prefixes
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import SKOS, DCTERMS, RDF, RDFS

def slugify(value):
    """
    Converts a string into a URI-friendly format (slug).
    """
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    return value

def extend_vocabulary(graph, df, base_namespace_uri, language_pref_label=None, language_alt_label=None, language_definition=None):
    conflicts = []
    report = {"new": [], "updated": [], "conflicts": []}

    # Ensure all concept schemes exist before processing terms
    unique_concept_groups = df['ConceptGroup'].dropna().unique()
    for group in unique_concept_groups:
        group_slug = slugify(str(group))
        # Use /scheme/ path for ConceptScheme URIs
        scheme_uri = URIRef(f"{base_namespace_uri}scheme/{group_slug}")
        if (scheme_uri, RDF.type, SKOS.ConceptScheme) not in graph:
            graph.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
            graph.add((scheme_uri, DCTERMS.title, Literal(f"{group} Vocabulary", lang=language_pref_label)))
            graph.add((scheme_uri, DCTERMS.description, Literal(f"A controlled vocabulary of {group} concepts from the Ambrosia project.", lang=language_definition)))
    
    match_type_mapping = {
        "skos:exactmatch": SKOS.exactMatch,
        "skos:closematch": SKOS.closeMatch,
        "skos:broadmatch": SKOS.broadMatch,
        "skos:narrowmatch": SKOS.narrowMatch,
        "skos:relatedmatch": SKOS.relatedMatch
    }

    for index, row in df.iterrows():
        term = row.get('Term')
        if pd.isna(term) or not term.strip():
            continue

        # Check if a concept with this term already exists.
        # This check is now more specific to prevent conflicts with ConceptSchemes.
        # It explicitly looks for a subject that is of type skos:Concept.
        existing_concept_uri = None
        for s in graph.subjects(SKOS.prefLabel, Literal(term, lang=language_pref_label)):
            # Confirm that the found subject is actually a Concept.
            if (s, RDF.type, SKOS.Concept) in graph:
                existing_concept_uri = s
                break
        
        if not existing_concept_uri:
            # Also check altLabels, with the same explicit type check.
            alt_labels_raw = row.get('altLabel')
            if pd.notna(alt_labels_raw):
                for alt_label_text in str(alt_labels_raw).split(','):
                    stripped_alt_label = alt_label_text.strip()
                    if stripped_alt_label:
                        for s in graph.subjects(SKOS.altLabel, Literal(stripped_alt_label, lang=language_alt_label)):
                            if (s, RDF.type, SKOS.Concept) in graph:
                                existing_concept_uri = s
                                break
                    if existing_concept_uri:
                        break
        
        slugified_term = slugify(term)
        concept_group = row.get('ConceptGroup')
        group_slug = slugify(str(concept_group)) if pd.notna(concept_group) and str(concept_group).strip() else None
        
        if existing_concept_uri:
            # Concept exists, merge properties
            report["updated"].append(term)
            concept_uri = existing_concept_uri
        else:
            # Concept does not exist, create a new one
            report["new"].append(term)
            # Use /concept/ path for Concept URIs, nested under group if available
            if group_slug:
                concept_uri = URIRef(f"{base_namespace_uri}concept/{group_slug}/{slugified_term}")
            else:
                concept_uri = URIRef(f"{base_namespace_uri}concept/{slugified_term}")
            
            graph.add((concept_uri, RDF.type, SKOS.Concept))
            graph.add((concept_uri, SKOS.prefLabel, Literal(term, lang=language_pref_label)))

            if group_slug:
                # Link concept to its scheme using the new scheme URI structure
                scheme_uri = URIRef(f"{base_namespace_uri}scheme/{group_slug}")
                graph.add((concept_uri, SKOS.inScheme, scheme_uri))

        # --- Merge altLabels ---
        alt_labels_raw = row.get('altLabel')
        if pd.notna(alt_labels_raw):
            for alt_label_text in str(alt_labels_raw).split(','):
                stripped_alt_label = alt_label_text.strip()
                if stripped_alt_label:
                    graph.add((concept_uri, SKOS.altLabel, Literal(stripped_alt_label, lang=language_alt_label)))

        # --- Merge definitions and sources ---
        for i in range(1, 4):
            description = row.get(f'ProviderDescription{i}')
            if pd.notna(description):
                graph.add((concept_uri, SKOS.definition, Literal(description, lang=language_definition)))
            
            source_provider = row.get(f'SourceProvider{i}')
            if pd.notna(source_provider):
                graph.add((concept_uri, DCTERMS.source, Literal(str(source_provider).strip())))

        # --- Merge mappings and detect conflicts ---
        for i in range(1, 4):
            external_uri_str = row.get(f'URI{i}')
            match_type_str = row.get(f'Match Type{i}')
            if pd.notna(external_uri_str) and pd.notna(match_type_str):
                skos_property = match_type_mapping.get(str(match_type_str).lower().strip())
                if skos_property:
                    new_mapping_uri = URIRef(external_uri_str)
                    
                    # Check for conflicts by namespace
                    new_namespace = "/".join(new_mapping_uri.split("/")[:-1])
                    
                    conflict_found = False
                    for existing_mapping in graph.objects(concept_uri, skos_property):
                        existing_namespace = "/".join(existing_mapping.split("/")[:-1])
                        if new_namespace == existing_namespace and new_mapping_uri != existing_mapping:
                            conflict = {
                                "term": term,
                                "concept_uri": str(concept_uri),
                                "property": str(skos_property),
                                "existing_uri": str(existing_mapping),
                                "new_uri": str(new_mapping_uri),
                                "row_index": index
                            }
                            conflicts.append(conflict)
                            report["conflicts"].append(term)
                            conflict_found = True
                            break
                    
                if not conflict_found:
                    # Add the new mapping if it doesn't already exist
                    if (concept_uri, skos_property, new_mapping_uri) not in graph:
                        graph.add((concept_uri, skos_property, new_mapping_uri))

    # Remove duplicate terms from report
    report["updated"] = sorted(list(set(report["updated"])))
    report["new"] = sorted(list(set(report["new"])))
    report["conflicts"] = sorted(list(set(report["conflicts"])))

    return graph, conflicts, report


def display_conflict_resolution(conflicts):
    """
    Displays a UI for resolving conflicts and returns the user's choices.
    """
    resolutions = {}
    for i, conflict in enumerate(conflicts):
        st.subheader(f"Conflict for Term: '{conflict['term']}'")
        st.write(f"**Property:** `{conflict['property']}`")
        st.write(f"**Existing URI:** `{conflict['existing_uri']}`")
        st.write(f"**New URI:** `{conflict['new_uri']}`")
        
        resolution = st.radio(
            "Choose how to resolve this conflict:",
            ("Keep existing mapping", "Replace with new mapping", "Keep both (downgrade new to skos:closeMatch)"),
            key=f"conflict_{i}"
        )
        resolutions[i] = resolution
    return resolutions

def apply_resolutions(graph, resolutions, conflicts):
    """
    Applies the user's conflict resolutions to the graph.
    """
    for i, resolution in resolutions.items():
        conflict = conflicts[i]
        concept_uri = URIRef(conflict["concept_uri"])
        existing_uri = URIRef(conflict["existing_uri"])
        new_uri = URIRef(conflict["new_uri"])
        prop = URIRef(conflict["property"])

        if resolution == "Replace with new mapping":
            graph.remove((concept_uri, prop, existing_uri))
            graph.add((concept_uri, prop, new_uri))
        elif resolution == "Keep both (downgrade new to skos:closeMatch)":
            graph.add((concept_uri, SKOS.closeMatch, new_uri))
        # If "Keep existing mapping", do nothing
    return graph


def serialize_graph_to_custom_turtle(graph):
    """
    Serializes the graph to a well-ordered Turtle string by manually building the string.
    This ensures ConceptSchemes are first, properties are sorted correctly, and only
    necessary prefixes are included. This is the definitive, robust solution.
    """
    # 1. Collect only the URIs that require external prefix definitions.
    # This prevents including prefixes for common vocabularies like rdf, rdfs, etc.
    mapping_properties = [
        SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch,
        SKOS.narrowMatch, SKOS.relatedMatch
    ]
    external_uris = {str(o) for p in mapping_properties for s, o in graph.subject_objects(p) if isinstance(o, URIRef)}

    # 2. Find the minimal set of relevant prefixes for ONLY the external mapping URIs.
    #    And explicitly manage the namespaces for qname generation.
    
    # Create a new NamespaceManager to control what prefixes are bound for qname generation.
    # We pass a dummy graph to init manager, but we will manually control prefix output.
    ns_manager = rdflib.namespace.NamespaceManager(Graph()) 

    # Store prefixes we explicitly want to include in the output file.
    explicit_output_prefixes = {}

    # Always bind skos and dct as they are fundamental to the vocabulary, and add to explicit list.
    ns_manager.bind("skos", SKOS)
    explicit_output_prefixes["skos"] = str(SKOS)
    ns_manager.bind("dct", DCTERMS)
    explicit_output_prefixes["dct"] = str(DCTERMS)
    
    try:
        all_prefixes_from_file = extract_prefixes('all.file.sparql.txt')
        relevant_prefixes_from_file = find_relevant_prefixes(all_prefixes_from_file, list(external_uris))
        for prefix, namespace in relevant_prefixes_from_file.items():
            ns_manager.bind(prefix, URIRef(namespace))
            explicit_output_prefixes[prefix] = namespace # Add to our explicit list for output
    except FileNotFoundError:
        st.warning("`all.file.sparql.txt` not found. Only skos and dct prefixes will be used for external URIs.")


    # 3. Build the final prefix string from the explicitly collected prefixes.
    #    Sort by prefix name for consistent output.
    prefix_lines = [f"@prefix {p}: <{ns}> ." for p, ns in sorted(explicit_output_prefixes.items(), key=lambda x: x[0])]
    prefix_str = "\n".join(prefix_lines)

    # 4. Define the absolute property order
    prop_order = [
        DCTERMS.title, DCTERMS.description, SKOS.inScheme, SKOS.prefLabel,
        SKOS.altLabel, SKOS.definition, DCTERMS.source, SKOS.exactMatch,
        SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch, SKOS.relatedMatch
    ]
    prop_order_map = {prop: i for i, prop in enumerate(prop_order)}

    # 5. Get and sort subjects
    all_subjects = sorted([s for s in graph.subjects() if isinstance(s, URIRef)], key=str)
    scheme_subjects = {s for s in all_subjects if (s, RDF.type, SKOS.ConceptScheme) in graph}
    concept_subjects = {s for s in all_subjects if (s, RDF.type, SKOS.Concept) in graph}
    
    sorted_pure_schemes = sorted(list(scheme_subjects - concept_subjects), key=str)
    sorted_concepts = sorted(list(concept_subjects), key=str)

    # 6. Helper to format an RDF object into its string representation
    def format_object(obj, is_subject=False, is_in_scheme_object=False):
        if isinstance(obj, URIRef):
            # If the URI contains a '?', it's often problematic for parsers when prefixed.
            # Fallback to using the full URI in angle brackets to ensure compatibility.
            if '?' in str(obj):
                return f"<{obj}>"

            # Subjects and inScheme objects should always be full URIs.
            if is_subject or is_in_scheme_object:
                return f"<{obj}>"
            
            # For other URIRef objects, try to find a matching prefix from our explicit list.
            for pfx, ns_uri in explicit_output_prefixes.items():
                if str(obj).startswith(ns_uri):
                    # Manually construct the prefixed name if a match is found.
                    # This bypasses rdflib's internal qname generation which might add unwanted nsX prefixes.
                    local_name = str(obj)[len(ns_uri):]
                    return f"{pfx}:{local_name}"
            
            # If no explicit prefix was found, return the full URI.
            return f"<{obj}>"

        elif isinstance(obj, Literal):
            lang = f"@{obj.language}" if obj.language else ""
            # Use triple quotes for multiline strings or strings containing double quotes.
            if "\n" in obj or '"' in obj:
                escaped_obj = obj.replace('"""', '\\"\\"\\"')
                return f'"""{escaped_obj}"""{lang}'
            else:
                return f'"{obj}"{lang}'
        else:
            return obj.n3()

    # 7. Helper to build a complete Turtle block for a single subject
    def build_subject_block(subject):
        # Format the subject, ensuring it's always a full URI.
        subject_str = format_object(subject, is_subject=True)
        
        props_and_objects = list(graph.predicate_objects(subject))
        prop_dict = {}
        for p, o in props_and_objects:
            if p not in prop_dict:
                prop_dict[p] = []
            prop_dict[p].append(o)
            
        def sort_key(p):
            return (prop_order_map.get(p, len(prop_order)), str(p))
        
        sorted_prop_keys = sorted(prop_dict.keys(), key=sort_key)
        
        lines = []
        # Handle the 'a' (RDF.type) property first.
        if RDF.type in sorted_prop_keys:
            types = sorted(prop_dict[RDF.type], key=str)
            type_str = " , ".join(format_object(t) for t in types)
            lines.append(f"{subject_str} a {type_str}")
            sorted_prop_keys.remove(RDF.type)
        else:
            # This case should ideally not happen for SKOS data but is a fallback.
            lines.append(subject_str)

        # Process the remaining properties.
        for prop in sorted_prop_keys:
            # Use the custom ns_manager for qname generation
            # We need to ensure that prop_qname is generated using the same logic as format_object for objects.
            # If the predicate's namespace is not in explicit_output_prefixes, it should be a full URI.
            prop_qname = None
            for pfx, ns_uri in explicit_output_prefixes.items():
                if str(prop).startswith(ns_uri):
                    prop_qname = f"{pfx}:{str(prop)[len(ns_uri):]}"
                    break
            if prop_qname is None:
                prop_qname = f"<{prop}>" # Fallback to full URI for predicate if no explicit prefix

            # Sort objects to ensure consistent output order.
            sorted_objects = sorted(prop_dict[prop], key=str)
            # Format each object and join with commas for multiple values.
            # Pass is_in_scheme_object=True if the current property is SKOS.inScheme
            objects_str = " ,\n        ".join(format_object(o, is_in_scheme_object=(prop == SKOS.inScheme)) for o in sorted_objects)
            lines.append(f"    {prop_qname} {objects_str}")
            
        # Join the property lines with semicolons and end the block with a period.
        if len(lines) > 1:
            return " ;\n".join(lines) + " ."
        # Handle the case where there's only a subject and type declaration.
        elif lines:
            return lines[0] + " ."
        return ""

    # 8. Assemble the final Turtle string
    all_blocks = []
    for s in sorted_pure_schemes:
        all_blocks.append(build_subject_block(s))
    for s in sorted_concepts:
        all_blocks.append(build_subject_block(s))

    return "\n\n".join([prefix_str] + all_blocks), explicit_output_prefixes # Return explicit_output_prefixes

def render_skos_generator_ui(embedded=False, key_ns=""):
    if not embedded:
        st.set_page_config(page_title="SKOS Vocabulary Generator", layout="wide")

    st.title("SKOS Vocabulary Generator")

    st.markdown("""
This tool generates a SKOS vocabulary from an uploaded Excel file.
""")

    st.subheader("Excel File Structure Information")
    st.info("""
Your Excel file should be structured with the following columns:
- **Term**: The primary term for the SKOS Concept (required).
- **altLabel**: Alternative labels for the term (comma-separated, optional).
- **ProviderDescription1**: A description for the term (optional).
- **URI1, Match Type1**: First external mapping (URI and SKOS match type, e.g., `skos:exactMatch`).
- **URI2, Match Type2**: Second external mapping (optional).
- **URI3, Match Type3**: Third external mapping (optional).
- **ConceptGroup**: The group for the term, used to structure the URI.

**Maximum of 3 external mappings are supported.**
""")

    # Define the template columns
    template_columns = [
        "Term", "altLabel", "ConceptGroup",
        "URI1", "Match Type1", "SourceProvider1", "ProviderTerm1", "ProviderDescription1",
        "URI2", "Match Type2", "SourceProvider2", "ProviderTerm2", "ProviderDescription2",
        "URI3", "Match Type3", "SourceProvider3", "ProviderTerm3", "ProviderDescription3"
    ]

    # Create an empty DataFrame for the template
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(columns=template_columns).to_excel(writer, index=False, sheet_name='SKOS_Template')
    data = output.getvalue()

    st.download_button(
        label="Download Excel Template",
        data=data,
        file_name="skos_vocabulary_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download an Excel template with the required column headers."
    )

    st.markdown("---") # Add a separator for clarity

    uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

    with st.expander("Extend Existing Vocabulary (Optional)"):
        existing_vocab_file = st.file_uploader(
            "Upload existing SKOS vocabulary file",
            type=["ttl", "jsonld", "rdf", "xml"]
        )

    if "conflicts" not in st.session_state:
        st.session_state.conflicts = []
    if "resolutions" not in st.session_state:
        st.session_state.resolutions = {}
    if "graph" not in st.session_state:
        st.session_state.graph = None


    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(df.head())

            st.subheader("Configuration")

            base_namespace = st.text_input(
                "Base Namespace URI",
                value="https://www.ambrosia-project.eu/vocab/",
                help="This URI will be the base for all generated SKOS concepts and collections."
            )



            st.markdown("---")
            st.subheader("Optional Language Tags")

            pref_label_lang = None
            alt_label_lang = None
            definition_lang = None

            COMMON_LANGUAGES = {
                "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
                "pt": "Portuguese", "nl": "Dutch", "zh": "Chinese (Simplified)", "ja": "Japanese",
                "ar": "Arabic", "ru": "Russian", "ko": "Korean", "hi": "Hindi", "sv": "Swedish",
                "da": "Danish", "no": "Norwegian", "fi": "Finnish", "pl": "Polish", "tr": "Turkish",
                "el": "Greek", "cs": "Czech", "hu": "Hungarian", "id": "Indonesian",
                "th": "Thai", "vi": "Vietnamese", "uk": "Ukrainian", "ro": "Romanian",
                "bg": "Bulgarian", "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian",
                "lv": "Latvian", "et": "Estonian", "is": "Icelandic"
            }
            lang_options = [f"{code} ({name})" for code, name in COMMON_LANGUAGES.items()]
            default_lang_index = list(COMMON_LANGUAGES.keys()).index('en') if 'en' in COMMON_LANGUAGES else 0

            col1, col2, col3 = st.columns(3)

            with col1:
                enable_pref_label_lang = st.checkbox("Enable Language for prefLabel", value=False)
                if enable_pref_label_lang:
                    pref_label_selection = st.selectbox(
                        "Language for prefLabel",
                        options=lang_options,
                        index=default_lang_index,
                        key="pref_label_lang_select"
                    )
                    pref_label_lang = pref_label_selection.split(' ')[0]

            with col2:
                enable_alt_label_lang = st.checkbox("Enable Language for altLabel", value=False)
                if enable_alt_label_lang:
                    alt_label_selection = st.selectbox(
                        "Language for altLabel",
                        options=lang_options,
                        index=default_lang_index,
                        key="alt_label_lang_select"
                    )
                    alt_label_lang = alt_label_selection.split(' ')[0]

            with col3:
                enable_definition_lang = st.checkbox("Enable Language for Definition", value=False)
                if enable_definition_lang:
                    definition_selection = st.selectbox(
                        "Language for Definition",
                        options=lang_options,
                        index=default_lang_index,
                        key="definition_lang_select"
                    )
                    definition_lang = definition_selection.split(' ')[0]

            st.markdown("---")
            st.subheader("Output Format")
            serialization_format = st.radio(
                "Select output serialization format:",
                ('Turtle (.ttl)', 'RDF/XML (.rdf)', 'JSON-LD (.jsonld)'),
                index=0 # Default to Turtle
            )

            if existing_vocab_file and not st.session_state.get("conflicts_resolved"):
                if st.button("Check for Conflicts"):
                    g = Graph()
                    
                    # Explicitly bind common RDF namespaces that might be expected by the parser
                    g.bind("rdf", RDF)
                    g.bind("rdfs", RDFS)

                    # Load all known prefixes from all.file.sparql.txt and bind them to the graph
                    # This is crucial for parsing existing vocabulary files that might use these prefixes.
                    try:
                        all_prefixes_from_file = extract_prefixes('all.file.sparql.txt')
                        for namespace_uri, prefixes_list in all_prefixes_from_file.items():
                            # Use the first prefix found for a given namespace for binding
                            if prefixes_list:
                                g.bind(prefixes_list[0], URIRef(namespace_uri))
                    except FileNotFoundError:
                        st.warning("`all.file.sparql.txt` not found. Parsing of existing vocabulary might fail if it uses unknown prefixes.")
                    except Exception as e:
                        st.error(f"Error loading prefixes for parsing: {e}")

                    g.parse(existing_vocab_file, format=rdflib.util.guess_format(existing_vocab_file.name))
                    st.session_state.graph = g
                    
                    g, conflicts, report = extend_vocabulary(
                        g, df, base_namespace, pref_label_lang, alt_label_lang, definition_lang
                    )
                    st.session_state.conflicts = conflicts
                    st.session_state.report = report

                    if conflicts:
                        st.warning("Conflicts detected! Please resolve them below.")
                    else:
                        st.success("No conflicts detected. You can now generate the extended vocabulary.")
                        st.session_state.conflicts_resolved = True


                if st.session_state.conflicts:
                    resolutions = display_conflict_resolution(st.session_state.conflicts)
                    if st.button("Resolve Conflicts"):
                        st.session_state.resolutions = resolutions
                        st.session_state.conflicts_resolved = True
                        st.success("Conflicts resolved. You can now generate the extended vocabulary.")

            
            if st.button("Generate SKOS Vocabulary", disabled=(existing_vocab_file is not None and not st.session_state.get("conflicts_resolved"))):
                if not base_namespace.strip():
                    st.error("Base Namespace URI cannot be empty.")
                else:
                    with st.spinner("Generating SKOS vocabulary..."):
                        try:
                            g = st.session_state.graph if st.session_state.graph else Graph()

                            if existing_vocab_file:
                                if st.session_state.get('conflicts_resolved'):
                                    if st.session_state.resolutions:
                                        g = apply_resolutions(g, st.session_state.resolutions, st.session_state.conflicts)
                                    
                                    # Re-run extend_vocabulary to apply non-conflicting changes
                                    g, _, report = extend_vocabulary(
                                        g, df, base_namespace, pref_label_lang, alt_label_lang, definition_lang
                                    )
                                    st.session_state.report = report # Update report
                            else:
                                # This is for creating a new vocabulary from scratch
                                g, _, report = extend_vocabulary(
                                    g, df, base_namespace, pref_label_lang, alt_label_lang, definition_lang
                                )
                                st.session_state.report = report

                            # --- Finalize and Serialize Graph ---
                            # Determine format and file extension
                            format_map = {
                                'Turtle (.ttl)': ('turtle', 'ttl'),
                                'RDF/XML (.rdf)': ('xml', 'rdf'),
                                'JSON-LD (.jsonld)': ('json-ld', 'jsonld')
                            }
                            serialization_format_key, file_extension = format_map[serialization_format]

                            # Use the custom serializer for Turtle to ensure correct order
                            if serialization_format_key == 'turtle':
                                output_str, explicit_output_prefixes = serialize_graph_to_custom_turtle(g)
                            else:
                                # For other formats, use standard rdflib serialization
                                output_str = g.serialize(format=serialization_format_key)
                                # For non-Turtle formats, we don't have explicit_output_prefixes from the custom serializer
                                explicit_output_prefixes = {} 
                            
                            output_data = output_str.encode('utf-8')
                            mime_type = f"application/{serialization_format_key}" if serialization_format_key != 'turtle' else 'text/turtle'

                            st.success("SKOS vocabulary generated successfully!")
                            
                            # Prepare and provide downloadable log file (moved before vocabulary download)
                            import datetime
                            log_data = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "excel_file_name": uploaded_file.name if uploaded_file else None,
                                "existing_vocab_file_name": existing_vocab_file.name if existing_vocab_file else None,
                                "generation_report": st.session_state.report,
                                "explicitly_included_prefixes": explicit_output_prefixes
                            }
                            log_json = json.dumps(log_data, indent=2)
                            st.download_button(
                                label="Download Generation Log (.json)",
                                data=log_json.encode('utf-8'),
                                file_name="skos_generation_log.json",
                                mime="application/json"
                            )

                            st.download_button(
                                label=f"Download Generated SKOS Vocabulary as .{file_extension}",
                                data=output_data,
                                file_name=f"skos_vocabulary.{file_extension}",
                                mime=mime_type
                            )
                            
                            st.subheader("Preview of Generated SKOS Vocabulary")
                            st.code(output_str, language=serialization_format_key)

                        except Exception as e:
                            st.error(f"An error occurred during generation: {e}")
                            st.exception(e)

        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            st.exception(e)

    # Add logic to reset state when new files are uploaded
    if 'last_uploaded_file_id' not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    if 'last_existing_vocab_file_id' not in st.session_state:
        st.session_state.last_existing_vocab_file_id = None

    current_uploaded_file_id = uploaded_file.file_id if uploaded_file else None
    current_existing_vocab_file_id = existing_vocab_file.file_id if existing_vocab_file else None

    if (current_uploaded_file_id != st.session_state.last_uploaded_file_id or
        current_existing_vocab_file_id != st.session_state.last_existing_vocab_file_id):
        
        # Reset state variables
        st.session_state.conflicts = []
        st.session_state.resolutions = {}
        st.session_state.graph = None
        st.session_state.conflicts_resolved = False
        
        # Update stored file IDs
        st.session_state.last_uploaded_file_id = current_uploaded_file_id
        st.session_state.last_existing_vocab_file_id = current_existing_vocab_file_id
        
        # The app will naturally rerun due to file uploader changes,
        # so explicit rerun is not needed and causes an error in newer Streamlit versions.
        pass

if __name__ == "__main__":
    render_skos_generator_ui()
