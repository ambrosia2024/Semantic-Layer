# AMBROSIA Semantic Data Standard (v1.0)

## 1. Purpose and scope

The **AMBROSIA Semantic Data Standard** defines a shared vocabulary and set of ontologies for representing:

- climate and weather data (including climate change scenarios),
- agricultural production contexts (farms, crops, regions),
- food safety hazards (pathogens and mycotoxins),
- predictive microbial risk models, and  
- user requests and results in the AMBROSIA dashboard.

The goal is to provide a **common semantic layer** across the AMBROSIA platform so that:

- data from different sources (NetCDF climate data, predictive models, hazard vocabularies, NUTS regions) can be combined consistently,
- the back end can query and reason over this data in a uniform way (SPARQL over RDF), and
- the front end receives clear, structured JSON/JSON-LD representations of questions such as:

> *“For crop X in region Y and time period Z, under scenario S, how does climate change affect the occurrence of hazard H?”*

The standard is realised as a **suite of OWL and SKOS ontologies in Turtle (`.ttl`)**, versioned under stable `w3id.org` IRIs, and is designed to be extended over time while remaining backward compatible.  
Version 1.0 underpins **Milestone MS6 – Interoperability standard**.

### 1.1 Supporting tools and applications

To support the **creation, population and maintenance** of the semantic layer, AMBROSIA provides several user-friendly tools and applications that build on this standard:

* A **reconciliation service** that matches project-specific terms to external ontologies and vocabularies, returning stable URIs and annotations to keep vocabularies aligned.
* A **SKOS generator**, which takes structured input (e.g. Excel with term/URI combinations) and produces SKOS-compliant Turtle vocabularies that can be versioned and extended.
* An **FSKX to RDF generator**, which converts existing FSKX model files into RDF/FSKXO so that predictive microbial models originally provided in `.fskx` format can be brought into the AMBROSIA knowledge graph.

These tools do not primarily expose the semantic layer to end users, but rather **populate and maintain** the underlying semantic resources (vocabularies and model instances) that the platform relies on.
More detailed technical descriptions are available here: [https://deepwiki.com/ambrosia2024/Semantic-Layer/](https://deepwiki.com/ambrosia2024/Semantic-Layer/)

---

## 2. Design principles and architecture

### 2.1 Core principles

The AMBROSIA semantic standard follows these principles:

- **Reuse before reinventing**  
  Reuse well-known cross-domain ontologies and vocabularies wherever possible (SOSA/SSN, GeoSPARQL, Time, QUDT, PROV, FOAF, SAREF4Agri, NUTS, FoodEx2, CHEBI, FSKXO, SKOS).
- **Modular and layered**  
  Split the model into focused modules:
  - a *core ontology* that glues everything together,
  - *domain vocabularies* for plants, pathogens and climate variables,
  - *external reference vocabularies* (NUTS + geometry),
  - *instance data* (predictive models, requests, observations).
- **Implementation-ready**  
  All modules are provided as Turtle files that can be loaded into a triple store and directly queried with SPARQL.
- **Alignment to existing modelling practice**  
  Predictive models are represented via FSKXO and aligned with AMBROSIA concepts, so existing FSKX models can be reused without changing their core structure.

### 2.2 Layers

Conceptually, the AMBROSIA semantic layer mirrors the layered approach used by other agricultural information models:

1. **Cross-domain layer (reused external ontologies)**  
   - W3C/OGC **SOSA/SSN** for observations and features of interest  
   - **Time** ontology for temporal concepts  
   - **GeoSPARQL** and **WGS84** for spatial features and geometries  
   - **QUDT** for quantity kinds, units and numeric values  
   - **PROV-O** for provenance (activities, agents, generatedBy)  
   - **FOAF** for basic agent/person concepts  
   - **SAREF4Agri** for farms, parcels, and agricultural roles  
   - **FSKXO** for predictive microbial models and their metadata  
   - **SKOS** for controlled vocabularies (crops, pathogens, risk levels, etc.)

2. **Domain layer (agri-food & climate vocabularies)**  
   Implemented as dedicated SKOS vocabularies:
   - Plant vocabulary (`ambrosia-plant-vocab.ttl`)  
   - Pathogen & mycotoxin vocabulary (`ambrosia-pathogen-vocab.ttl`)  
   - NetCDF / climate vocabulary (`ambrosia-netcdf-vocab.ttl`)  
   - NUTS + geometry (`nuts-skos-enriched-with-geometry.ttl`)

3. **Platform / pilot-specific layer (AMBROSIA ontology)**  
   Implemented in `ambrosia-ontology.ttl`, with AMBROSIA-specific modules for:
   - model and data linking,
   - dashboard context,
   - agents/farms/parcels,
   - observations.

4. **Instance data layer (models, requests, observations)**  
   - FSKX-derived predictive models in RDF (e.g. `ac398182-01ab-48f0-b25c-ca432631b018.ttl`),  
   - future AMBROSIA instance data (user requests, farm profiles, observations, results).

---

## 3. Files and namespaces

### 3.1 Files

The semantic layer is currently defined by the following Turtle files:

- `ambrosia-ontology.ttl` – AMBROSIA core ontology (v1.0)  
- `ambrosia-netcdf-vocab.ttl` – climate / NetCDF vocabulary  
- `ambrosia-plant-vocab.ttl` – plant / crop vocabulary  
- `ambrosia-pathogen-vocab.ttl` – pathogen / mycotoxin vocabulary  
- `nuts-skos-enriched-with-geometry.ttl` – NUTS regions with geometry  
- `ac398182-01ab-48f0-b25c-ca432631b018.ttl` – example predictive microbial model (FSKX → RDF)

### 3.2 Namespaces

Key namespaces used across the files:

```ttl
PREFIX amblink:  <https://w3id.org/ambrosia/linking#>
PREFIX ambdash:  <https://w3id.org/ambrosia/dashboard#>
PREFIX ambag:    <https://w3id.org/ambrosia/agent#>
PREFIX amobs:    <https://w3id.org/ambrosia/observation#>
PREFIX vocab:    <https://w3id.org/ambrosia/vocab#>

PREFIX ambplant: <https://w3id.org/ambrosia/plant-vocab#>
PREFIX ambpath:  <https://w3id.org/ambrosia/pathogen-vocab#>
PREFIX ambnc:    <https://w3id.org/ambrosia/netcdf-vocab#>
PREFIX nuts:     <http://data.europa.eu/nuts/>

PREFIX sosa:     <http://www.w3.org/ns/sosa/>
PREFIX ssn:      <http://www.w3.org/ns/ssn/>
PREFIX time:     <http://www.w3.org/2006/time#>
PREFIX geo:      <http://www.opengis.net/ont/geosparql#>
PREFIX wgs84:    <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX qudt:     <http://qudt.org/schema/qudt/>
PREFIX unit:     <http://qudt.org/vocab/unit/>
PREFIX prov:     <http://www.w3.org/ns/prov#>
PREFIX foaf:     <http://xmlns.com/foaf/0.1/>
PREFIX skos:     <http://www.w3.org/2004/02/skos/core#>
````

Very roughly, the AMBROSIA-specific modules cover:

* `amblink:` – predictive models, input/output mappings, risk semantics
* `ambdash:` – dashboard and user request context
* `ambag:`   – agents, farms and parcels
* `amobs:`   – climate, hazard and yield observations
* `vocab:`   – model parameter vocabulary
* `ambplant:` – crop concepts (SKOS)
* `ambpath:`  – pathogen / mycotoxin concepts (SKOS)
* `ambnc:`    – NetCDF/climate variables
* `nuts:`     – spatial reference (EU NUTS regions)

All AMBROSIA URIs are stable under `https://w3id.org/ambrosia/...`.

---

## 4. Core ontology (`ambrosia-ontology.ttl`)

### 4.1 Ontology overview

* **Ontology IRI:** `https://w3id.org/ambrosia/ontology`
* **Version:** `owl:versionInfo "1.0"`

The ontology:

* imports SKOS, SOSA/SSN, Time, QUDT, PROV, FOAF, GeoSPARQL, SAREF4Agri and FSKXO,
* defines the AMBROSIA-specific modules: `amblink`, `ambdash`, `ambag`, `amobs`, `vocab`,
* binds these to the domain vocabularies (`ambplant`, `ambpath`, `ambnc`) and NUTS.

### 4.2 Linking models and data (`amblink:`)

**Main classes**

* `amblink:PredictiveModel`

  * Predictive model used in AMBROSIA.
  * Subclass of `sosa:Procedure` and `prov:Plan`.
  * `owl:equivalentClass` to the FSKXO predictive model class, so FSKX models are reused directly.
* `amblink:InputMapping` / `amblink:OutputMapping`

  * Define how model parameters are linked to data variables and outputs.
* `amblink:ObservedProperty`

  * Observable quantity (e.g. concentration, dose, risk), subclass of `sosa:ObservableProperty` and `skos:Concept`.
* `amblink:Endpoint`, `amblink:ExposureRoute`, `amblink:Population`

  * Controlled concepts describing health endpoints, exposure routes and target populations.
* `amblink:FoodSafetyIncident`, `amblink:MitigationMeasure`

  * Food safety incidents/alerts and mitigation measures.

**Key properties**

* `amblink:hasInputMapping` / `amblink:hasOutputMapping`

  * Domain: `amblink:PredictiveModel`
  * Range: `amblink:InputMapping` / `amblink:OutputMapping`
* `amblink:mapsParameter`

  * `InputMapping` / `OutputMapping` → `vocab:ModelParameter`.
* `amblink:isFulfilledBy`

  * Links an input mapping to the data variable or concept that fulfils it (e.g. a climate variable from `ambnc:`).
* `amblink:expectsObservedProperty`

  * `vocab:ModelParameter` → `amblink:ObservedProperty`.
* `amblink:expectsQuantityKind`, `amblink:expectsUnit`

  * `vocab:ModelParameter` → quantity kind and unit (QUDT).
* Concept schemes for:

  * observed properties, health endpoints, exposure routes, target populations, risk levels, statistics.

This module provides the semantic glue between:

* FSKX/FSKXO predictive model metadata,
* climate and other input data, and
* AMBROSIA’s risk-oriented outputs.

### 4.3 Dashboard and user request context (`ambdash:`)

**Main classes**

* `ambdash:FarmProfile`

  * Profile of a farm used to drive climate and food safety queries.
  * Subclass of `sosa:FeatureOfInterest`, `wgs84:SpatialThing` and `ambag:Farm`.
* `ambdash:UserRequest`

  * A concrete query coming from the dashboard.
* `ambdash:TimeSelection`

  * Selected time window and temporal resolution (e.g. period 2030–2050, annual).
* `ambdash:ScenarioSelection`

  * Selected climate scenario(s).

**Key properties**

* `ambdash:hasNUTSRegion`

  * `FarmProfile` → NUTS region (`nuts:`), subproperty of `dct:spatial`.
* `ambdash:hasCrop`

  * `UserRequest` → crop concept (`ambplant:`).
* `ambdash:hasPathogen` / `ambdash:hasHazardType`

  * `UserRequest` → specific pathogen/mycotoxin (`ambpath:`) and more general hazard category.
* `ambdash:hasTimeSelection`

  * `UserRequest` → `TimeSelection`.
* `ambdash:startDate`, `ambdash:endDate`, `ambdash:hasTimePeriod`, `ambdash:hasTimeScale`

  * Properties of `TimeSelection`.
* `ambdash:hasFarmProfile`

  * `UserRequest` → `FarmProfile`.
* `ambdash:usesModel`

  * `UserRequest` → predictive model (`amblink:PredictiveModel`).

This module encodes exactly what the dashboard lets users specify:

* who is asking (role),
* where (NUTS-based location),
* what crop and hazard,
* and for which time period and climate scenario.

### 4.4 Agents, farms and parcels (`ambag:`)

**Main classes**

* `ambag:Agent`, `ambag:Person`, `ambag:Organization`
* `ambag:Farmer`, `ambag:Advisor`, `ambag:PolicyMaker`, `ambag:FoodProcessor`
* `ambag:Farm` – farm as spatial feature and management unit (aligned with SAREF4Agri).
* `ambag:Parcel` – field or parcel.

These classes let the platform represent the human and organisational actors behind requests and data, and link farms/parcels to user profiles and contexts.

### 4.5 Observations (`amobs:`)

**Main classes**

* `amobs:CropInRegionContext`

  * A feature of interest that combines a crop concept and a NUTS region, optionally linked to a `FarmProfile`.
* `amobs:AmbrosiaObservation`

  * Base class for AMBROSIA observations; subclass of `sosa:Observation`.
* `amobs:ClimateObservation`, `amobs:HazardObservation`, `amobs:YieldObservation`

  * Specialisations for climate, hazard and yield.

**Key properties**

* `amobs:hasCropConcept`

  * `CropInRegionContext` → crop concept (`ambplant:`).
* `amobs:hasRegion`

  * `CropInRegionContext` → NUTS region (`nuts:`); subproperty of `dct:spatial`.
* `amobs:hasFarmProfile`

  * Optionally links context to `ambdash:FarmProfile`.
* `amobs:hasContext`

  * `AmbrosiaObservation` → `CropInRegionContext`; subproperty of `sosa:hasFeatureOfInterest`.
* `amobs:hasQuantityResult`

  * `AmbrosiaObservation` → `qudt:QuantityValue`; subproperty of `sosa:hasResult`.
* `amobs:forRequest`

  * `AmbrosiaObservation` → `ambdash:UserRequest`; subproperty of `prov:wasGeneratedBy`.

This module provides a unified representation for time series of:

* **climate variables**,
* **hazard indicators**, and
* **yield**,

always tied to a **crop + region (+ optional farm)** and to the user request that triggered computations.

### 4.6 Model parameters (`vocab:`)

* `vocab:ModelParameter`

  * Model input or output parameter.
  * Linked from `amblink:InputMapping` / `amblink:OutputMapping`.
  * Parameter semantics (observable, unit, quantity kind) are attached via:

    * `amblink:expectsObservedProperty`,
    * `amblink:expectsQuantityKind`,
    * `amblink:expectsUnit`.

---

## 5. Domain vocabularies and reference data

### 5.1 Climate / NetCDF vocabulary (`ambrosia-netcdf-vocab.ttl`, `ambnc:`)

The NetCDF vocabulary defines a controlled list of NetCDF variables and related concepts used in AMBROSIA:

* coordinate variables: `.../latitude`, `.../longitude`, `.../time`,
* climate variables such as temperature and precipitation,
* vertical context (e.g. separate height variable, reference level).

Each concept is represented as a `skos:Concept` (and often `nc:Variable`) with:

* NetCDF metadata: `nc:hasDataType`, `nc:axis`, `nc:long_name`, `nc:standard_name`, `nc:units`,
* QUDT annotations: `qudt:unit` and `qudt:quantityKind`,
* optional links to external resources (e.g. Wikidata).

These concepts are what `amblink:InputMapping` / `amblink:isFulfilledBy` will typically point to when a predictive model expects specific climate inputs.

### 5.2 Plant vocabulary (`ambrosia-plant-vocab.ttl`, `ambplant:`)

The plant vocabulary is a SKOS concept scheme (with sub-schemes for cereals, fruits, nuts, vegetables). Each crop or commodity concept:

* is a `skos:Concept` with labels and definitions,
* is aligned to external codes via `skos:exactMatch`, notably:

  * **FoodEx2** IDs,
  * **FSKXO** identifiers,
  * **Wikidata** entries.

These concepts are used for:

* dashboard configuration (`ambdash:hasCrop`),
* observation contexts (`amobs:hasCropConcept`),
* model scoping (which commodities a model applies to).

### 5.3 Pathogen & mycotoxin vocabulary (`ambrosia-pathogen-vocab.ttl`, `ambpath:`)

The pathogen vocabulary defines two SKOS concept schemes:

* a mycotoxin scheme (e.g. aflatoxin B1, B2, etc.),
* a pathogen scheme (relevant microbial hazards).

Each concept has labels and definitions and is aligned with external references:

* **CHEBI** for chemical entities,
* **FSKXO** for hazard identifiers in models,
* **NCBI Taxon** for organisms (where relevant),
* **Wikidata** entities.

These concepts are used in:

* dashboard requests (`ambdash:hasPathogen`, `ambdash:hasHazardType`),
* hazard observations,
* model semantics.

### 5.4 NUTS + geometry (`nuts-skos-enriched-with-geometry.ttl`, `nuts:`)

This file provides EU NUTS regions as SKOS concepts, enriched with:

* NUTS codes (`dct:identifier`),
* labels and descriptions,
* geometry information through GeoSPARQL / LOCN.

AMBROSIA uses NUTS regions as canonical spatial units:

* `ambdash:hasNUTSRegion` for farm profiles,
* `amobs:hasRegion` for crop-in-region contexts.

This allows climate and hazard observations to be aggregated and visualised per NUTS region.

### 5.5 FSKX-derived predictive model example (`ac398182-01ab-48f0-b25c-ca432631b018.ttl`)

This file is an RDF representation of a predictive microbial model created from an FSKX file using the project’s FSKX→RDF generator. It:

* uses the **FSKXO** vocabulary to represent all FSKX sections (general info, scope, data, model, etc.),
* serves as a realistic example of how existing FSKX models appear in RDF.

Through `amblink:PredictiveModel` being equivalent to the FSKXO predictive model class, this instance can be treated as an AMBROSIA predictive model and linked to:

* `amblink:InputMapping` / `amblink:OutputMapping`,
* dashboard requests (`ambdash:usesModel`),
* observations generated by the model.

---

## 6. End-to-end flow: dashboard scenario

A typical AMBROSIA dashboard interaction looks like this:

> A farmer (or advisor) enters a **location**, **crop**, **hazard** and a **future time period** to see how climate change affects hazard occurrence.

In semantic terms:

1. **User and farm**

   * The user is represented as an `ambag:Farmer` (or `Advisor`/`PolicyMaker`), linked to an `ambag:Farm`.
   * The farm is represented as an `ambdash:FarmProfile`.

2. **Location**

   * The farm profile is linked to a NUTS region via `ambdash:hasNUTSRegion` → `nuts:...`.
   * More detailed geometry can be added via GeoSPARQL/WGS84 if needed.

3. **Crop**

   * The user selects a crop (e.g. “winter wheat”) → a `skos:Concept` in `ambplant:`.
   * The request uses `ambdash:hasCrop` → selected crop concept.

4. **Hazard**

   * The user selects a pathogen or mycotoxin from `ambpath:`.
   * The request uses `ambdash:hasPathogen` and optionally `ambdash:hasHazardType`.

5. **Time window**

   * The user selects a future time window and time scale.
   * Represented as a `ambdash:TimeSelection` instance linked via `ambdash:hasTimeSelection`, with:

     * `ambdash:startDate` / `ambdash:endDate`,
     * `ambdash:hasTimePeriod` (time interval),
     * `ambdash:hasTimeScale` (aggregation level, e.g. monthly, yearly).

All of this is bundled into a `ambdash:UserRequest`.

6. **Context and data selection**

   * A `amobs:CropInRegionContext` is created or reused:

     * `amobs:hasCropConcept` = selected crop,
     * `amobs:hasRegion` = NUTS region,
     * optionally `amobs:hasFarmProfile` = farm profile.
   * Climate data is selected as `amobs:ClimateObservation` tied to this context and time period, using climate variables from `ambnc:`.

7. **Model selection and execution**

   * Suitable predictive models are selected as `amblink:PredictiveModel` / FSKXO instances whose input parameters and scope match:

     * the crop and hazard (via `ambplant:` / `ambpath:` and FSKXO metadata),
     * the needed observed properties, quantity kinds and units (`vocab:ModelParameter` + `amblink:expects*`).
   * Input and output mappings (`amblink:InputMapping` / `amblink:OutputMapping`) describe how model parameters are fulfilled by climate and other data.

8. **Hazard predictions**

   * Model outputs are stored as `amobs:HazardObservation`:

     * `amobs:hasContext` → `CropInRegionContext`,
     * `amobs:hasQuantityResult` → `qudt:QuantityValue` (numeric value + unit),
     * optionally classified with risk level concepts from `amblink:` (e.g. low/medium/high risk).
   * Observations are linked to the originating request via `amobs:forRequest`.

9. **Presentation to the user**

   * The backend queries this RDF graph (via SPARQL) and projects the results into JSON/JSON-LD.
   * The frontend displays:

     * time series for climate variables and hazard predictions,
     * maps by NUTS region,
     * model metadata and risk levels.

---

## 7. Example SPARQL queries

The following example queries illustrate how the semantic standard is intended to be used from the backend. They can be adapted to concrete triple-store and API needs.

### 7.1 List crops and hazards used in requests for a given NUTS region

```sparql
PREFIX ambdash:  <https://w3id.org/ambrosia/dashboard#>
PREFIX ambplant: <https://w3id.org/ambrosia/plant-vocab#>
PREFIX ambpath:  <https://w3id.org/ambrosia/pathogen-vocab#>
PREFIX skos:     <http://www.w3.org/2004/02/skos/core#>
PREFIX nuts:     <http://data.europa.eu/nuts/>
PREFIX dct:      <http://purl.org/dc/terms/>

SELECT DISTINCT ?crop ?cropLabel ?hazard ?hazardLabel
WHERE {
  ?request a ambdash:UserRequest ;
           ambdash:hasCrop ?crop ;
           ambdash:hasPathogen ?hazard ;
           ambdash:hasFarmProfile ?farmProfile .

  ?farmProfile ambdash:hasNUTSRegion ?region .
  ?region dct:identifier ?nutsCode .

  FILTER(?nutsCode = "DE40")  # example NUTS code

  ?crop   skos:prefLabel ?cropLabel .
  ?hazard skos:prefLabel ?hazardLabel .

  FILTER(LANG(?cropLabel)   = "en")
  FILTER(LANG(?hazardLabel) = "en")
}
ORDER BY ?cropLabel ?hazardLabel
```

### 7.2 Retrieve climate time series for a crop-in-region context

```sparql
PREFIX amobs:    <https://w3id.org/ambrosia/observation#>
PREFIX sosa:     <http://www.w3.org/ns/sosa/>
PREFIX qudt:     <http://qudt.org/schema/qudt/>
PREFIX ambplant: <https://w3id.org/ambrosia/plant-vocab#>
PREFIX nuts:     <http://data.europa.eu/nuts/>
PREFIX skos:     <http://www.w3.org/2004/02/skos/core#>
PREFIX dct:      <http://purl.org/dc/terms/>

SELECT ?time ?value ?unit
WHERE {
  # Identify context for a given crop and NUTS region
  ?context a amobs:CropInRegionContext ;
           amobs:hasCropConcept ?crop ;
           amobs:hasRegion ?region .

  ?crop   skos:prefLabel "Winter wheat"@en .
  ?region dct:identifier "DE40" .

  # Climate observations for that context
  ?obs a amobs:ClimateObservation ;
       amobs:hasContext ?context ;
       sosa:resultTime ?time ;
       amobs:hasQuantityResult ?q .

  ?q qudt:numericValue ?value ;
     qudt:unit ?unit .
}
ORDER BY ?time
```

### 7.3 List predictive models and their expected inputs

```sparql
PREFIX amblink: <https://w3id.org/ambrosia/linking#>
PREFIX vocab:   <https://w3id.org/ambrosia/vocab#>
PREFIX skos:    <http://www.w3.org/2004/02/skos/core#>
PREFIX qudt:    <http://qudt.org/schema/qudt/>

SELECT ?model ?parameter ?paramLabel ?observedPropertyLabel ?quantityKind ?unit
WHERE {
  ?model a amblink:PredictiveModel ;
         amblink:hasInputMapping ?mapping .

  ?mapping amblink:mapsParameter ?parameter .

  ?parameter a vocab:ModelParameter ;
             rdfs:label ?paramLabel ;
             amblink:expectsObservedProperty ?obsProp ;
             amblink:expectsQuantityKind ?quantityKind ;
             amblink:expectsUnit ?unit .

  ?obsProp a amblink:ObservedProperty ;
           skos:prefLabel ?observedPropertyLabel .

  FILTER(LANG(?paramLabel)            = "en")
  FILTER(LANG(?observedPropertyLabel) = "en")
}
ORDER BY ?model ?parameter
```

### 7.4 Retrieve hazard predictions for a specific user request

```sparql
PREFIX amobs:  <https://w3id.org/ambrosia/observation#>
PREFIX ambdash: <https://w3id.org/ambrosia/dashboard#>
PREFIX sosa:   <http://www.w3.org/ns/sosa/>
PREFIX qudt:   <http://qudt.org/schema/qudt/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>

SELECT ?time ?value ?unit ?riskLabel
WHERE {
  BIND(<https://w3id.org/ambrosia/request/123> AS ?request)

  ?obs a amobs:HazardObservation ;
       amobs:forRequest ?request ;
       sosa:resultTime ?time ;
       amobs:hasQuantityResult ?q .

  ?q qudt:numericValue ?value ;
     qudt:unit ?unit .

  OPTIONAL {
    ?obs skos:broader ?riskConcept .
    ?riskConcept skos:prefLabel ?riskLabel .
    # riskConcept can be constrained to the RiskLevel scheme if needed
  }
}
ORDER BY ?time
```
