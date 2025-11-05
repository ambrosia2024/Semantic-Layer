import json
import argparse
import logging
from pathlib import Path
from vocab_mapper import VocabularyMapper

# Configuration
INPUT_FOLDER = "./unmapped/jsonld"
OUTPUT_FOLDER = "./mapped/jsonld"
ONTOLOGY_FILE = "./fskxo.owl"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def process_files(override=False):
    """
    Process all JSON-LD files in the input folder using the VocabularyMapper
    and save them to the output folder.
    """
    logging.info("Initializing Vocabulary Mapper...")
    mapper = VocabularyMapper(ONTOLOGY_FILE)

    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    output_path.mkdir(parents=True, exist_ok=True)

    jsonld_files = sorted(input_path.glob('*.jsonld'))
    logging.info(f"Found {len(jsonld_files)} JSON-LD files to process in '{INPUT_FOLDER}'.")

    if not jsonld_files:
        logging.info("No files to process. Exiting.")
        return

    for file_path in jsonld_files:
        output_file_path = output_path / file_path.name
        if not override and output_file_path.exists():
            logging.info(f"Skipping existing file: {output_file_path}")
            continue

        logging.info(f"Processing: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mapped_data = mapper.map_object(data)

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(mapped_data, f, indent=2, ensure_ascii=False)

            logging.info(f"  -> Saved mapped file to: {output_file_path}")

        except Exception as e:
            logging.error(f"  -> ERROR processing {file_path.name}: {e}", exc_info=True)
        
        # Save mappings after each file to make new mappings available to subsequent files
        mapper.save_custom_mappings()

    logging.info("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply vocabulary mapping to JSON-LD files.")
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing output files.'
    )
    args = parser.parse_args()

    process_files(override=args.override)
