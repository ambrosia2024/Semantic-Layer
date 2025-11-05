from pyld import jsonld
import json
import os
import csv
import logging
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def validate_jsonld(jsonld_data):
    try:
        # Expand the JSON-LD to check for syntax and context issues
        expanded = jsonld.expand(jsonld_data)
        # Compact the expanded JSON-LD to ensure it can be compacted back
        compacted = jsonld.compact(expanded, {})
        return True, "JSON-LD is valid."
    except jsonld.JsonLdError as e:
        return False, f"Validation error: {str(e)}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def validate_jsonld_folder(folder_path, output_csv=None):
    """
    Validate all JSON-LD files in a folder and return/save results.

    Args:
        folder_path: Path to folder containing JSON-LD files
        output_csv: Optional path to save results as CSV (if None, prints to console)

    Returns:
        List of dictionaries containing validation results
    """
    results = []
    folder = Path(folder_path)

    # Get all .jsonld files
    jsonld_files = sorted(folder.glob('*.jsonld'))

    logging.info(f"Found {len(jsonld_files)} JSON-LD files to validate in '{folder_path}'...")

    for file_path in jsonld_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jsonld_data = json.load(f)

            is_valid, message = validate_jsonld(jsonld_data)

            result = {
                'filename': file_path.name,
                'valid': is_valid,
                'message': message,
                'file_path': str(file_path)
            }
            results.append(result)

            # Log progress
            if is_valid:
                logging.info(f"✓ VALID: {file_path.name}")
            else:
                logging.error(f"✗ INVALID: {file_path.name} - {message}")

        except Exception as e:
            result = {
                'filename': file_path.name,
                'valid': False,
                'message': f"Failed to load file: {str(e)}",
                'file_path': str(file_path)
            }
            results.append(result)
            logging.error(f"✗ FAILED_LOAD: {file_path.name} - {str(e)}", exc_info=True)

    # Save to CSV if output path provided
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'valid', 'message', 'file_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

        logging.info(f"Validation results saved to: {output_csv}")

    # Print summary
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    logging.info("="*60)
    logging.info(f"SUMMARY: {valid_count} valid, {invalid_count} invalid out of {len(results)} files")
    logging.info("="*60)

    return results

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    jsonld_folder = script_dir / "unmapped" / "jsonld"
    output_csv_path = script_dir / "validation_results.csv"

    # Validate all JSON-LD files in the folder
    results = validate_jsonld_folder(jsonld_folder, output_csv_path)
