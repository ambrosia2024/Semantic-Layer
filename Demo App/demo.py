import streamlit as st
import rdflib
import xarray as xr
import os
from datetime import datetime, date
import geopandas as gpd
import rioxarray
from shapely.wkt import loads
import numpy as np
from pyproj import CRS
import pandas as pd
from pathlib import Path
from rdflib import Graph
import os

# Debug logging setup
import logging

logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level to see all messages
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



# --- Configuration ---
VOCAB_DIR = Path(__file__).resolve().parent.parent / "Vocabulary"
LOCAL_DATA_DIR = Path("data")

# Validated EURO-CORDEX metadata
EXPERIMENT_OPTIONS = ["historical", "evaluation", "rcp26", "rcp45", "rcp85"]
GCM_OPTIONS = ["EC-EARTH", "HadGEM2-ES", "MPI-ESM-LR"]
RCM_OPTIONS = ["RACMO22E", "RCA4", "REMO2009"]
MEMBER_OPTIONS = ["r1i1p1", "r2i1p1"]
TIME_FREQ_OPTIONS = ["day", "mon"]

# Year range constraints based on experiment
YEAR_RANGES = {
    "historical": (1950, 2005),
    "evaluation": (1979, 2015),
    "rcp_2_6": (2006, 2100),
    "rcp_4_5": (2006, 2100),
    "rcp_8_5": (2006, 2100),
}

def validate_year_for_experiment(exp: str, year: str):
    y = int(year)
    y0, y1 = YEAR_RANGES[exp]
    if not (y0 <= y <= y1):
        raise ValueError(
            f"Year {year} not valid for experiment {exp}. Allowed range: {y0}–{y1}."
        )

@st.cache_resource
def load_knowledge_graph():
    g = rdflib.Graph()
    for filename in os.listdir(VOCAB_DIR):
        if filename.endswith(".ttl"):
            g.parse(os.path.join(VOCAB_DIR, filename), format="turtle")
    print(f"Knowledge Graph loaded with {len(g)} triples.")
    return g

@st.cache_data
def get_dropdown_choices(_graph, scheme_uris):
    query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?label ?concept_uri
        WHERE {{
            ?concept_uri skos:inScheme ?scheme .
            ?concept_uri skos:prefLabel ?label .
            FILTER(STRSTARTS(STR(?scheme), "{scheme_uris[0]}") || STRSTARTS(STR(?scheme), "{scheme_uris[1]}"))
        }}
        ORDER BY ?label
    """
    if len(scheme_uris) == 4:
         query = f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?label ?concept_uri
            WHERE {{
                ?concept_uri skos:inScheme ?scheme .
                ?concept_uri skos:prefLabel ?label .
                FILTER(STRSTARTS(STR(?scheme), "{scheme_uris[0]}") || STRSTARTS(STR(?scheme), "{scheme_uris[1]}") || STRSTARTS(STR(?scheme), "{scheme_uris[2]}") || STRSTARTS(STR(?scheme), "{scheme_uris[3]}"))
            }}
            ORDER BY ?label
        """
    results = _graph.query(query)
    choices = [(row.label.toPython(), row.concept_uri.toPython()) for row in results]
    return choices

@st.cache_data
def get_model_choices(_graph):
    query = """
        PREFIX fskxo: <http://semanticlookup.zbmed.de/km/fskxo/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?product_label ?product_uri ?hazard_label ?hazard_uri
        WHERE {
            ?model_uri a fskxo:FSKXO_0000018113 ;
                       fskxo:FSKXO_0000000007 ?product_uri ;
                       fskxo:FSKXO_0000000008 ?hazard_uri .
            ?product_uri skos:prefLabel ?product_label .
            ?hazard_uri skos:prefLabel ?hazard_label .
        }
        ORDER BY ?product_label ?hazard_label
    """
    results = _graph.query(query)
    
    product_choices = sorted(list(set((row.product_label.toPython(), row.product_uri.toPython()) for row in results)))
    hazard_choices = sorted(list(set((row.hazard_label.toPython(), row.hazard_uri.toPython()) for row in results)))
    
    return product_choices, hazard_choices

@st.cache_data
def get_nuts_choices(_graph, levels=[2]):
    if not levels:
        return [("Select a region...", None)]

    level_filters = " || ".join([f'STR(?level) = "{l}"' for l in levels])
    
    query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX nuts: <http://data.europa.eu/nuts/>
        PREFIX geo: <http://www.opengis.net/ont/geosparql#>
        SELECT ?label ?concept_uri
        WHERE {{
            ?concept_uri a skos:Concept ;
                         nuts:level ?level ;
                         skos:prefLabel ?label ;
                         geo:hasGeometry ?geom .
            ?geom geo:asWKT ?wkt .
            FILTER({level_filters})
        }}
        ORDER BY ?label
    """
    results = _graph.query(query)
    choices = [("Select a region...", None)] + [(row.label.toPython(), row.concept_uri.toPython()) for row in results]
    return choices


def wkt_to_bbox_nwse(wkt: str) -> list[float]:
    from shapely.wkt import loads as wkt_loads
    g = wkt_loads(wkt)
    minx, miny, maxx, maxy = g.bounds
    # CDS expects [N, W, S, E] in degrees
    return [maxy, minx, miny, maxx]

def pad_bbox_nwse(bbox, pad_deg):
    N, W, S, E = bbox
    return [N + pad_deg, W - pad_deg, S - pad_deg, E + pad_deg]

def sanitize_bbox_nwse(bbox):
    N, W, S, E = bbox
    N = max(min(N, 90.0), -90.0)
    S = max(min(S, 90.0), -90.0)
    W = ((W + 180) % 360) - 180
    E = ((E + 180) % 360) - 180
    if N < S:
        N, S = S, N
    if E < W:
        E, W = W, E
    return [N, W, S, E]

def ensure_min_size_bbox(bbox, min_span=0.05):
    N, W, S, E = bbox
    if (N - S) < min_span:
        mid = (N + S) / 2.0
        N, S = mid + min_span/2.0, mid - min_span/2.0
    if (E - W) < min_span:
        mid = (E + W) / 2.0
        E, W = mid + min_span/2.0, mid - min_span/2.0
    return [N, W, S, E]

def open_local(variable: str, years: list[int]) -> xr.Dataset:
    """
    Open local NetCDFs for the given variable and years from LOCAL_DATA_DIR.
    Supports patterns like tas_day_YYYY*.nc (adjust to your filenames).
    """
    patterns = []
    for y in years:
        # Adjust the pattern to match your local file naming
        patterns.append(str(LOCAL_DATA_DIR / f"{variable}_day_{y}*.nc"))
        patterns.append(str(LOCAL_DATA_DIR / f"{variable}_*.nc"))  # fallback
    files = []
    import glob
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(
            f"No local files for '{variable}' in {LOCAL_DATA_DIR}. "
            "Adjust filenames or check the `local_data_dir` variable in the script."
        )
    return xr.open_mfdataset(sorted(set(files)), combine="by_coords")



# --- Part 3: The Core Discovery Logic (MODIFIED FOR LOGGING) ---
def find_model_and_inputs(graph, product_uri, hazard_uri):
    """
    Performs discovery and returns findings AND a log of its actions.
    """
    # Initialize a list to hold our log messages
    log_messages = []
    
    query_prefix = """
        PREFIX fskxo: <http://semanticlookup.zbmed.de/km/fskxo/>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX amblink: <https://www.ambrosia-project.eu/vocab/linking/>
    """
    
    # --- Discovery Query ---
    model_discovery_query = query_prefix + f"""
        SELECT ?model_id ?model_uri
        WHERE {{
            ?model_uri a fskxo:FSKXO_0000018113 ;
                       fskxo:FSKXO_0000000007 <{product_uri}> ;
                       fskxo:FSKXO_0000000008 <{hazard_uri}> ;
                       dct:identifier ?model_id .
        }}
    """
    log_messages.append(f"🔍 **Querying for Model:** Searching for a model with product=`<{product_uri.split('/')[-1]}>` and hazard=`<{hazard_uri.split('/')[-1]}>`.")
    results = graph.query(model_discovery_query)
    
    # Handle results (same as before)
    result_list = list(results)
    model_info = None
    if len(result_list) >= 1:
        model_info = result_list[0]
    
    if not model_info:
        log_messages.append("❌ **Query Failed:** No model in the knowledge graph matched these criteria.")
        return None, None, log_messages

    model_id, model_uri = model_info
    log_messages.append(f"✅ **Model Found:** Discovered model `{model_id}`. This link is defined in `fskx-models.ttl`.")
    
    # --- Wiring Query (MODIFIED to get mapping and concept URI) ---
    data_wiring_query = query_prefix + f"""
        SELECT ?netcdf_var_name ?mapping_uri ?concept_uri
        WHERE {{
            <{model_uri}> amblink:hasInputMapping ?mapping_uri .
            ?mapping_uri amblink:sourceVariableName ?netcdf_var_name ;
                         amblink:isFulfilledBy ?concept_uri .
        }}
    """
    log_messages.append(f"🔍 **Querying for Wiring:** Searching for the `InputMapping` associated with model `{model_id}`.")
    results = graph.query(data_wiring_query)
    
    # Extract all requirements and mappings
    data_reqs = []
    for row in results:
        var_name = row.netcdf_var_name.toPython()
        data_reqs.append(var_name)
        concept_uri = row.concept_uri

        # Log the specific mapping instance found
        log_messages.append(f"✅ **Wiring Found:** The link is made through the instance `<{row.mapping_uri.split('/')[-1]}>`.")
        log_messages.append(f"   - This instance is defined in **`wiring-instances.ttl`**.")
        log_messages.append(f"   - It links to the conceptual variable `<{concept_uri.split('/')[-1]}>` via `amblink:isFulfilledBy`.")

        # --- Unit Discovery Query ---
        unit_query = query_prefix + f"""
            PREFIX qudt: <http://qudt.org/schema/qudt/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?unit_uri
            WHERE {{
                <{concept_uri}> qudt:unit ?unit_uri .
            }}
        """
        unit_results = graph.query(unit_query)
        unit_uri = next((r.unit_uri for r in unit_results), None)
        
        if unit_uri:
            # Try to get the symbol first
            symbol_query = query_prefix + f"""
                PREFIX qudt: <http://qudt.org/schema/qudt/>
                SELECT ?unit_symbol
                WHERE {{
                    <{unit_uri}> qudt:symbol ?unit_symbol .
                }}
            """
            symbol_results = graph.query(symbol_query)
            unit_symbol = next((r.unit_symbol.toPython() for r in symbol_results), None)

            if not unit_symbol:
                # If no symbol, get the local name from the URI
                unit_symbol = unit_uri.split('/')[-1].split('#')[-1]
        else:
            unit_symbol = "unknown unit"

        log_messages.append(f"   - The concept `<{concept_uri.split('/')[-1]}>` is defined in **`ambrosia-netcdf-vocab.ttl`** and specifies the unit as **`{unit_symbol}`**.")

    if not data_reqs:
        log_messages.append(f"❌ **Wiring Failed:** Model `{model_id}` was found, but it has no `InputMapping` defined.")
        return model_id, None, log_messages

    return model_id, data_reqs, log_messages

# --- Main App UI and Logic ---
st.set_page_config(layout="wide")
st.title("AMBROSIA Semantic Wiring & Geospatial Clipping Demo")
st.info("This app demonstrates semantic discovery and regional data extraction using a clipped NetCDF file.")

graph = load_knowledge_graph()

# --- UI Sidebar ---
with st.sidebar:
    st.header("Data Mode")
    st.info("Using local historical NetCDF data.")
    st.caption(f"Local data directory: `{LOCAL_DATA_DIR}`")

    experiment = "historical"
    gcm_name = ""
    rcm_name = ""
    ensemble_member = ""
    time_frequency = "day"
    min_year, max_year = YEAR_RANGES[experiment]
    selected_year = 1980
    sel = {"experiment": experiment, "year": str(selected_year)}

    st.header("Temperature Input")
    temp_input = st.radio("Pass to model", ["Polygon mean", "Raster cell set"], index=0)

    st.divider()


# --- UI Layout ---
st.subheader("1. Define Your Scenario")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Product and Hazard Selection**")
    filter_by_model = st.checkbox("Only show options with a predictive model")

    if filter_by_model:
        product_choices, hazard_choices = get_model_choices(graph)
    else:
        product_schemes = [ "https://www.ambrosia-project.eu/vocab/scheme/cereal", "https://www.ambrosia-project.eu/vocab/scheme/fruit", "https://www.ambrosia-project.eu/vocab/scheme/nut", "https://www.ambrosia-project.eu/vocab/scheme/vegetable" ]
        product_choices = get_dropdown_choices(graph, product_schemes)
        hazard_schemes = [ "https://www.ambrosia-project.eu/vocab/scheme/mycotoxin", "https://www.ambrosia-project.eu/vocab/scheme/pathogen" ]
        hazard_choices = get_dropdown_choices(graph, hazard_schemes)
    
    selected_product = st.selectbox(
        "Select a Product:",
        options=product_choices,
        format_func=lambda x: x[0],
        index=None,
        placeholder="Select a product..."
    )
    selected_product_uri = selected_product[1] if selected_product else None

    selected_hazard = st.selectbox(
        "Select a Hazard:",
        options=hazard_choices,
        format_func=lambda x: x[0],
        index=None,
        placeholder="Select a hazard..."
    )
    selected_hazard_uri = selected_hazard[1] if selected_hazard else None

with right_col:
    st.markdown("**Region Selection**")
    
    st.write("NUTS Level")
    level_cols = st.columns(4)
    levels = []
    if level_cols[0].checkbox("0", value=False): levels.append(0)
    if level_cols[1].checkbox("1", value=False): levels.append(1)
    if level_cols[2].checkbox("2", value=True): levels.append(2)
    if level_cols[3].checkbox("3", value=False): levels.append(3)

    nuts_choices = get_nuts_choices(graph, levels)
    selected_nuts = st.selectbox(
        "Select a NUTS Region:",
        options=nuts_choices,
        format_func=lambda x: x[0],
        index=0
    )
    selected_nuts_uri = selected_nuts[1] if selected_nuts else None
    selected_nuts_label = selected_nuts[0] if selected_nuts else "the entire grid"


st.subheader("2. Select a Specific Time")
exp = sel.get("experiment", "historical")  # Default to historical for offline mode

if exp in YEAR_RANGES:
    y_min, y_max = YEAR_RANGES[exp]
else:
    y_min, y_max = 1950, 2100     # safe fallback

date_min = date(y_min, 1, 1)
date_max = date(y_max, 12, 31)

# keep a sensible default (if previously selected_year exists, clamp inside allowed window)
try:
    default_year = int(sel.get("year", y_min))
except Exception:
    default_year = y_min
default_year = min(max(default_year, y_min), y_max)

date_default = date(default_year, 1, 1)

selected_date = st.date_input(
    "Date:",
    value=date_default,
    min_value=date_min,
    max_value=date_max,
    key="date_input_key",
    help=f"Allowed range for {exp.replace('_', ' ')}: {y_min}-01-01 to {y_max}-12-31",
)
selected_time = st.time_input("Time (UTC):", value=datetime(1970, 1, 1, 12, 0).time())
selected_datetime = datetime.combine(selected_date, selected_time)

st.divider()


# --- The Execution Button ---
if st.button("Discover Model and Run Regional Analysis", type="primary", use_container_width=True):

    if not selected_product_uri or not selected_hazard_uri or not selected_nuts_uri:
        st.warning("Please select a product, a hazard, AND a NUTS region.")
        st.stop()

    # --- Step 1: Model Discovery ---
    with st.spinner("Step 1/3: Searching knowledge graph for model..."):
        model_id, required_vars, log_messages = find_model_and_inputs(graph, selected_product_uri, selected_hazard_uri)

    with st.expander("Show Semantic Discovery Log", expanded=True):
        for msg in log_messages:
            st.markdown(msg, unsafe_allow_html=True)
    
    if not model_id or not required_vars:
        st.error("**Model Discovery Failed.** Check the log above for details.")
        st.stop()
    else:
        st.success(f"**Model Discovery Complete!** The correct model is `{model_id}`.")
        
    st.divider()

    # --- Step 2: Geospatial Discovery ---
    wkt_polygon = None
    with st.spinner("Step 2/3: Retrieving geospatial geometry for selected region..."):
        geo_query = f"""
            PREFIX geo: <http://www.opengis.net/ont/geosparql#>
            SELECT ?wkt
            WHERE {{
                <{selected_nuts_uri}> geo:hasGeometry ?geom .
                ?geom geo:asWKT ?wkt .
            }}
        """
        results = graph.query(geo_query)
        wkt_polygon = next((row.wkt.toPython() for row in results), None)

    with st.expander("Show Geospatial Discovery Log", expanded=True):
        st.markdown(f"🔍 **Querying for Geometry:** Searching for WKT polygon for `<{selected_nuts_uri.split('/')[-1]}>`.")
        if wkt_polygon:
            st.markdown("✅ **Geometry Found:** Retrieved WKT data from `nuts-skos-enriched-with-geometry.ttl`.")
            st.code(wkt_polygon, language="wkt")
        else:
            st.markdown("❌ **Geometry Failed:** Could not find a `geo:asWKT` literal for the selected region.")
            st.error("**Geospatial Discovery Failed.** No geometry found for the selected region.")
            st.stop()

    st.divider()

    # --- Step 3: Climate data extraction (temperature only) ---
    st.subheader("3. Climate data extraction (temperature only)")

    try:
        yr = selected_date.year
        mo = selected_date.month
        dy = selected_date.day

        ds = open_local(variable="tas", years=[yr])
        detected_experiment = ds.attrs.get("experiment_id", "historical")
        st.caption(f"Detected experiment from file: `{detected_experiment}`")
        experiment = detected_experiment  # Override the default

        # --- Select time SLICE FIRST for performance ---
        t_sel_native = ds["tas"].sel(time=np.datetime64(selected_datetime), method="nearest")

        # --- Set CRS and reproject just this slice ---
        if "rotated_pole" in ds:
            rp = ds["rotated_pole"]
            cf = {
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_longitude": float(getattr(rp, "grid_north_pole_longitude", 0.0)),
                "grid_north_pole_latitude": float(getattr(rp, "grid_north_pole_latitude", 90.0)),
                "north_pole_grid_longitude": float(getattr(rp, "north_pole_grid_longitude", 0.0)),
            }
            da_native = t_sel_native.rio.set_spatial_dims(x_dim="rlon", y_dim="rlat").rio.write_crs(CRS.from_cf(cf))
        else:
            da_native = t_sel_native
        
        da_ll = da_native.rio.reproject("EPSG:4326", nodata=np.nan)

        # --- Clip with polygon (using the TTL geometry) ---
        geom = loads(wkt_polygon)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        t_clip = da_ll.rio.clip(gdf.geometry, drop=True)

        if int(t_clip.count()) == 0:
            st.error("No intersecting temperature cells in the selected polygon/time.")
            st.stop()

        # --- Units: convert to °C if in Kelvin ---
        tas_attrs = t_clip.attrs
        if tas_attrs.get("units", "").lower() in ("k", "kelvin") or float(t_clip.mean()) > 200:
            t_clip = t_clip - 273.15
            t_clip.attrs["units"] = "°C"

        # --- Branch A/B: Polygon mean vs Raster cell set ---
        arr = t_clip.values
        if temp_input == "Polygon mean":
            if "y" in t_clip.coords:
                lat_values = t_clip["y"].values
                lat2d = np.broadcast_to(lat_values[:, None], t_clip.shape)
                weights = np.cos(np.deg2rad(lat2d))
                mean_t = float(np.nansum(arr * weights) / np.nansum(np.where(np.isfinite(arr), weights, 0)))
            else:
                st.warning("Latitude coord not found post-reprojection; using unweighted mean.")
                mean_t = float(np.nanmean(arr))
            st.success(f"Temperature passed to the model (polygon mean): **{mean_t:.2f} °C**")
        else:
            # Raster cell set: show a compact representation of all temperatures
            values = arr[np.isfinite(arr)]

            # stats + visualization
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Cells", f"{values.size}")
            colB.metric("Mean (°C)", f"{np.nanmean(values):.2f}")
            colC.metric("P10–P90 (°C)", f"{np.nanpercentile(values,10):.1f}–{np.nanpercentile(values,90):.1f}")
            colD.metric("Min–Max (°C)", f"{np.nanmin(values):.1f}–{np.nanmax(values):.1f}")

            st.caption("The model will receive the full cell temperature set shown below.")
            # Histogram
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.hist(values, bins=30)
            plt.xlabel("Temperature (°C)"); plt.ylabel("Count")
            plt.title("Distribution of cell temperatures passed to the model")
            st.pyplot(fig)

            # 2D heatmap preview
            fig2 = plt.figure()
            plt.imshow(np.where(np.isfinite(arr), arr, np.nan), origin="lower")
            plt.colorbar(label="°C")
            plt.title("Clipped temperature tile (preview)")
            st.pyplot(fig2)

            # CSV download with coordinates
            yy, xx = np.meshgrid(t_clip["y"].values, t_clip["x"].values, indexing="ij")
            flat = pd.DataFrame({
                "x": xx.flatten(),
                "y": yy.flatten(),
                "temperature_c": arr.flatten()
            })
            flat = flat[np.isfinite(flat["temperature_c"])]
            st.download_button("Download temperatures (with coords)",
                data=flat.to_csv(index=False).encode("utf-8"),
                file_name=f"temps_{yr}-{mo:02d}_{selected_nuts_label}.csv",
                mime="text/csv")
        
        # Example: handoff
        if temp_input == "Polygon mean":
            temperatures_for_model = np.array([mean_t], dtype=float)  # shape (1,)
        else:
            temperatures_for_model = flat[["temperature_c"]].values.flatten() # 1D array of floats (°C)

        # TODO: pass 'temperatures_for_model' into your predictive model selection/execution.
        st.info(f"Data prepared for model. Shape: {temperatures_for_model.shape}")

    except Exception as e:
        st.error(f"An error occurred during the data retrieval or processing: {e}")
