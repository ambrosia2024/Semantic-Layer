# -*- coding: utf-8 -*-
import logging
# Conditional import for Streamlit caching
try:
    # Use Streamlit caching for the expensive model loading
    from streamlit import cache_resource
    # Set logger name for consistency if Streamlit context is available
    logger = logging.getLogger(__name__)
except ImportError:
    # Dummy decorator if Streamlit is not available (e.g., for testing)
    def cache_resource(func=None, **kwargs):
        if func: return func
        else: return lambda f: f
    # Use standard Python logger if Streamlit is not imported
    logger = logging.getLogger(__name__)
    # Configure basic logging if needed when run standalone
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# sentence_transformers will be imported locally in functions
from .reconciliation_utils import calculate_levenshtein_score # Import from utils
import time

# --- Configuration for Hybrid Score ---
# Weights for combining scores (adjust these based on experimentation)
WEIGHT_COSINE_DESC = 0.5  # Weight for cosine similarity (term vs. label + description)
WEIGHT_COSINE_LABEL = 0.2 # Weight for cosine similarity (term vs. label only)
WEIGHT_LEVENSHTEIN = 0.3 # Weight for Levenshtein similarity (term vs. label)
# Ensure weights sum to 1.0 (or normalize later)
if abs(WEIGHT_COSINE_DESC + WEIGHT_COSINE_LABEL + WEIGHT_LEVENSHTEIN - 1.0) > 1e-6:
    logger.warning("Hybrid score weights do not sum to 1.0. Normalization might be needed.")

@cache_resource # Cache the loaded model resource
def load_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads the Sentence Transformer model.
    Uses Streamlit's cache_resource for efficient loading.
    """
    logger.info(f"Loading Sentence Transformer model: {model_name}...")
    start_time = time.time()
    try:
        from sentence_transformers import SentenceTransformer # Local import
        model = SentenceTransformer(model_name)
        end_time = time.time()
        logger.info(f"Model '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")
        return model
    except Exception as e:
        logger.exception(f"Error loading Sentence Transformer model '{model_name}'. Cosine similarity disabled.")
        # Optionally: raise e # Or return None and handle in the calling function
        return None


def calculate_hybrid_scores(model, input_term: str, suggestions: list):
    """
    Calculates Levenshtein and Cosine Similarity scores (label & label+desc)
    and combines them into a hybrid score. Modifies the suggestions list in-place.

    Args:
        model: The loaded Sentence Transformer model.
        input_term (str): The original term from the input data.
        suggestions (list): List of suggestion dictionaries [{'uri':.., 'label':.., 'description':..}].
                            This list will be modified to include scores and sorted.

    Returns:
        list: The modified and sorted list of suggestions. Returns original list on error.
    """
    from sentence_transformers import util # Local import for util.cos_sim

    if not model:
        logger.error("Cannot calculate hybrid scores: Sentence Transformer model not loaded.")
        # Add levenshtein score even if model fails
        for sugg in suggestions:
             sugg['levenshtein_score'] = _calculate_levenshtein_score(input_term, sugg.get('label', ''))
             sugg['hybrid_score'] = sugg['levenshtein_score'] # Fallback
        suggestions.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
        return suggestions

    if not input_term or not suggestions:
        logger.warning("Cannot calculate hybrid scores: Empty input term or suggestions list.")
        return suggestions # Return original list if no input

    logger.info(f"Calculating hybrid scores for term '{input_term}' with {len(suggestions)} suggestions.")

    # --- 1. Calculate Levenshtein Scores ---
    for sugg in suggestions:
        label = sugg.get('label', '')
        sugg['levenshtein_score'] = _calculate_levenshtein_score(input_term, label)

    # --- 2. Prepare Texts for Cosine Calculation ---
    suggestion_labels = [s.get('label', '') for s in suggestions]
    suggestion_texts_with_desc = []
    for s in suggestions:
        label = s.get('label', '')
        desc = s.get('description', '')
        # Concatenate label and description, handle missing description
        full_text = f"{label} {desc}".strip() if desc else label
        suggestion_texts_with_desc.append(full_text if full_text else "") # Ensure non-empty string

    # --- 3. Encode Texts (Catch potential errors) ---
    try:
        logger.debug(f"Encoding input term: '{input_term}'")
        input_embedding = model.encode(input_term, convert_to_tensor=True, show_progress_bar=False)

        logger.debug(f"Encoding {len(suggestion_labels)} suggestion labels...")
        label_embeddings = model.encode(suggestion_labels, convert_to_tensor=True, show_progress_bar=False)

        logger.debug(f"Encoding {len(suggestion_texts_with_desc)} suggestion texts with descriptions...")
        desc_embeddings = model.encode(suggestion_texts_with_desc, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        logger.exception(f"Error during sentence encoding for term '{input_term}'. Cannot calculate cosine scores.")
        # Assign only levenshtein as hybrid score and sort by it
        for sugg in suggestions:
            sugg['hybrid_score'] = sugg.get('levenshtein_score', 0.0)
        suggestions.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
        return suggestions

    # --- 4. Calculate Cosine Similarities ---
    try:
        # Similarity between input term and suggestion labels
        cosine_scores_label = util.cos_sim(input_embedding, label_embeddings)[0]
        # Similarity between input term and suggestion label+description
        cosine_scores_desc = util.cos_sim(input_embedding, desc_embeddings)[0]

        # Convert scores from tensor to float list (move to CPU if necessary)
        cosine_scores_label = cosine_scores_label.cpu().tolist()
        cosine_scores_desc = cosine_scores_desc.cpu().tolist()

    except Exception as e:
        logger.exception(f"Error calculating cosine similarity for term '{input_term}'. Cannot calculate cosine scores.")
        # Assign only levenshtein as hybrid score and sort by it
        for sugg in suggestions:
            sugg['hybrid_score'] = sugg.get('levenshtein_score', 0.0)
        suggestions.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
        return suggestions

    # --- 5. Add Scores to Suggestions and Calculate Hybrid Score ---
    logger.debug("Calculating final hybrid scores...")
    for i, sugg in enumerate(suggestions):
        cos_score_lbl = cosine_scores_label[i] if i < len(cosine_scores_label) else 0.0
        cos_score_desc = cosine_scores_desc[i] if i < len(cosine_scores_desc) else 0.0
        lev_score = sugg.get('levenshtein_score', 0.0)

        sugg['cosine_score_label'] = cos_score_lbl
        sugg['cosine_score_desc'] = cos_score_desc

        # Calculate weighted hybrid score
        hybrid_score = (
            WEIGHT_COSINE_DESC * cos_score_desc +
            WEIGHT_COSINE_LABEL * cos_score_lbl +
            WEIGHT_LEVENSHTEIN * lev_score
        )
        sugg['hybrid_score'] = hybrid_score

    # --- 6. Sort Suggestions by Hybrid Score ---
    suggestions.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
    logger.info(f"Finished calculating hybrid scores for term '{input_term}'.")

    return suggestions
