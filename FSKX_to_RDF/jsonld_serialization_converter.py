#!/usr/bin/env python3
"""
JSON-LD Multi-Format Serialization Converter

Converts JSON-LD files to multiple RDF serialization formats including:
- RDF/XML (.rdf, .xml)
- Turtle (.ttl)
- N-Triples (.nt)
- N-Quads (.nq)
- Trig (.trig)

This script processes individual files or entire directories of JSON-LD files
and outputs them in the specified RDF serialization formats for GraphDB ingestion
and interoperability.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

try:
    from rdflib import Graph
    from rdflib.plugins.parsers.jsonld import JsonLDParser
    from rdflib.serializer import Serializer
except ImportError:
    print("Error: rdflib is required. Install with: pip install rdflib")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Namespace Definitions for RDF Serialization ---
PREFIX_MAP = {
    "fskxo": "http://semanticlookup.zbmed.de/km/fskxo/",
    "model": "https://www.ambrosia-project.eu/model/",
    "vocab": "https://www.ambrosia-project.eu/vocab/",
    "amblink": "https://www.ambrosia-project.eu/vocab/linking/",
    "schema": "https://schema.org/",
    "dcterms": "http://purl.org/dc/terms/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "qudt-unit": "http://qudt.org/vocab/unit/",
    "qk": "http://qudt.org/vocab/quantitykind/"
}

# Supported output formats with their file extensions
SUPPORTED_FORMATS = {
    'turtle': {
        'format': 'turtle',
        'extension': '.ttl',
        'description': 'Turtle (Terse RDF Triple Language)'
    },
    'rdfxml': {
        'format': 'xml',
        'extension': '.rdf',
        'description': 'RDF/XML'
    },
    'nt': {
        'format': 'nt',
        'extension': '.nt',
        'description': 'N-Triples'
    },
    'nquads': {
        'format': 'nquads',
        'extension': '.nq',
        'description': 'N-Quads'
    },
    'trig': {
        'format': 'trig',
        'extension': '.trig',
        'description': 'TriG (Turtle for Named Graphs)'
    },
    'jsonld': {
        'format': 'json-ld',
        'extension': '.jsonld',
        'description': 'JSON-LD (pretty printed)'
    }
}

class JSONLDConverter:
    """Converts JSON-LD files to various RDF serialization formats."""

    def __init__(self, output_dir: str = None, override: bool = False):
        """
        Initialize the converter.

        Args:
            output_dir: Directory for output files. If None, creates format-specific subdirs.
            override: Whether to override existing files.
        """
        self.output_dir = output_dir
        self.override = override
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'formats_created': {}
        }

    def load_jsonld_file(self, file_path: str) -> Optional[Dict]:
        """Load and validate a JSON-LD file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            logging.error(f"Error loading {file_path}: {e}", exc_info=True)
            return None

    def convert_to_format(self, jsonld_data: Dict, output_format: str) -> Optional[str]:
        """
        Convert JSON-LD data to specified RDF format.

        Args:
            jsonld_data: JSON-LD data as dictionary
            output_format: Target format (turtle, rdfxml, nt, nquads, trig, jsonld)

        Returns:
            Serialized RDF string or None if conversion failed
        """
        try:
            if output_format == 'jsonld':
                # Pretty print JSON-LD
                return json.dumps(jsonld_data, indent=2, ensure_ascii=False)

            # Create RDF graph and parse JSON-LD
            g = Graph()

            # Convert dict to JSON string for parsing
            jsonld_str = json.dumps(jsonld_data, ensure_ascii=False)
            g.parse(data=jsonld_str, format='json-ld')

            # Bind common namespaces for cleaner output
            for prefix, namespace in PREFIX_MAP.items():
                g.bind(prefix, namespace)

            # Get format configuration
            format_config = SUPPORTED_FORMATS.get(output_format)
            if not format_config:
                raise ValueError(f"Unsupported format: {output_format}")

            # Serialize to target format
            serialized = g.serialize(format=format_config['format'])

            # Handle both string and bytes return types
            if isinstance(serialized, bytes):
                return serialized.decode('utf-8')
            return serialized

        except Exception as e:
            logging.error(f"Error converting to {output_format}: {e}", exc_info=True)
            return None

    def get_output_path(self, input_file: str, output_format: str) -> str:
        """Generate output file path for given input file and format."""
        input_path = Path(input_file)
        format_config = SUPPORTED_FORMATS[output_format]

        if self.output_dir:
            # Use specified output directory
            output_dir = Path(self.output_dir)
        else:
            # Create format-specific subdirectory
            output_dir = input_path.parent / output_format

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        output_filename = input_path.stem + format_config['extension']
        return str(output_dir / output_filename)

    def convert_file(self, input_file: str, output_formats: List[str]) -> bool:
        """
        Convert a single JSON-LD file to specified formats.

        Args:
            input_file: Path to input JSON-LD file
            output_formats: List of target formats

        Returns:
            True if at least one format was successfully created
        """
        logging.info(f"Processing: {input_file}")

        # Load JSON-LD data
        jsonld_data = self.load_jsonld_file(input_file)
        if not jsonld_data:
            self.stats['failed'] += 1
            return False

        self.stats['processed'] += 1
        success_count = 0

        # Convert to each requested format
        for format_name in output_formats:
            try:
                output_path = self.get_output_path(input_file, format_name)
                if not self.override and os.path.exists(output_path):
                    logging.info(f"  -> Skipping existing file: {output_path}")
                    continue

                # Convert to target format
                serialized_data = self.convert_to_format(jsonld_data, format_name)
                if not serialized_data:
                    continue

                # Write to output file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(serialized_data)

                logging.info(f"  -> {format_name}: {output_path}")
                success_count += 1

                # Update statistics
                if format_name not in self.stats['formats_created']:
                    self.stats['formats_created'][format_name] = 0
                self.stats['formats_created'][format_name] += 1

            except Exception as e:
                logging.error(f"  -> Failed to create {format_name}: {e}", exc_info=True)

        if success_count > 0:
            self.stats['successful'] += 1
            return True
        else:
            self.stats['failed'] += 1
            return False

    def convert_directory(self, input_dir: str, output_formats: List[str], recursive: bool = True) -> None:
        """
        Convert all JSON-LD files in a directory.

        Args:
            input_dir: Directory containing JSON-LD files
            output_formats: List of target formats
            recursive: Whether to process subdirectories
        """
        input_path = Path(input_dir)
        if not input_path.is_dir():
            logging.error(f"{input_dir} is not a directory")
            return

        # Find JSON-LD files
        if recursive:
            jsonld_files = list(input_path.rglob("*.jsonld"))
        else:
            jsonld_files = list(input_path.glob("*.jsonld"))

        if not jsonld_files:
            logging.warning(f"No JSON-LD files found in {input_dir}")
            return

        logging.info(f"Found {len(jsonld_files)} JSON-LD files to process.")

        # Convert each file
        for jsonld_file in jsonld_files:
            self.convert_file(str(jsonld_file), output_formats)

    def print_statistics(self) -> None:
        """Print conversion statistics."""
        logging.info("="*50)
        logging.info("CONVERSION STATISTICS")
        logging.info("="*50)
        logging.info(f"Files processed: {self.stats['processed']}")
        logging.info(f"Successfully converted: {self.stats['successful']}")
        logging.info(f"Failed conversions: {self.stats['failed']}")

        if self.stats['formats_created']:
            logging.info("Files created by format:")
            for format_name, count in self.stats['formats_created'].items():
                description = SUPPORTED_FORMATS[format_name]['description']
                logging.info(f"  {format_name:10} ({description}): {count} files")
        logging.info("="*50)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert JSON-LD files to various RDF serialization formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported output formats:
  turtle    - Turtle (.ttl) - Terse RDF Triple Language
  rdfxml    - RDF/XML (.rdf) - W3C standard XML serialization
  nt        - N-Triples (.nt) - Simple line-based format
  nquads    - N-Quads (.nq) - N-Triples with named graphs
  trig      - TriG (.trig) - Turtle with named graphs
  jsonld    - JSON-LD (.jsonld) - Pretty printed JSON-LD

Examples:
  # Convert single file to Turtle and RDF/XML
  python jsonld_serialization_converter.py -i model.jsonld -f turtle rdfxml

  # Convert all files in directory to all formats
  python jsonld_serialization_converter.py -d ./jsonld -f all

  # Convert directory with custom output location
  python jsonld_serialization_converter.py -d ./jsonld -f turtle nt -o ./output
        """
    )

    # Default configuration when no arguments provided
    DEFAULT_INPUT_DIR = "./jsonld"
    DEFAULT_FORMATS = ['turtle', 'rdfxml']
    DEFAULT_OUTPUT_DIR = None  # Will create format-specific subdirectories

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '-i', '--input',
        help='Input JSON-LD file'
    )
    input_group.add_argument(
        '-d', '--directory',
        help=f'Directory containing JSON-LD files (default: {DEFAULT_INPUT_DIR})'
    )

    # Output options
    parser.add_argument(
        '-f', '--formats',
        nargs='+',
        choices=list(SUPPORTED_FORMATS.keys()) + ['all'],
        default=None,
        help=f'Output formats (default: {", ".join(DEFAULT_FORMATS)})'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: create format-specific subdirectories)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not process subdirectories when using -d'
    )
    parser.add_argument(
        '--list-formats',
        action='store_true',
        help='List supported formats and exit'
    )
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing output files.'
    )

    args = parser.parse_args()

    # Handle list formats option
    if args.list_formats:
        logging.info("Supported RDF serialization formats:")
        logging.info("-" * 50)
        for name, config in SUPPORTED_FORMATS.items():
            logging.info(f"{name:10} - {config['description']} ({config['extension']})")
        return

    # Apply defaults if no input is provided
    # Apply defaults if no input is provided
    if not args.input and not args.directory:
        logging.info(f"No input specified. Using default directory: {DEFAULT_INPUT_DIR}")
        args.directory = DEFAULT_INPUT_DIR
        if not os.path.exists(args.directory):
            logging.error(f"Default directory '{args.directory}' does not exist.")
            logging.error("Please specify an input file with -i or directory with -d")
            sys.exit(1)

    # Apply default formats if not specified
    if args.formats is None:
        args.formats = DEFAULT_FORMATS
        logging.info(f"Using default formats: {', '.join(DEFAULT_FORMATS)}")

    # Handle 'all' formats option
    if 'all' in args.formats:
        output_formats = list(SUPPORTED_FORMATS.keys())
    else:
        output_formats = args.formats

    # Initialize converter
    converter = JSONLDConverter(output_dir=args.output, override=args.override)

    try:
        if args.input:
            # Convert single file
            converter.convert_file(args.input, output_formats)
        else:
            # Convert directory
            recursive = not args.no_recursive
            converter.convert_directory(args.directory, output_formats, recursive)

        # Print statistics
        converter.print_statistics()

    except KeyboardInterrupt:
        print("\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
