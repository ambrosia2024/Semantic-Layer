import React, { useMemo, useState } from 'react';
import * as Reactodia from '@reactodia/workspace';
import * as N3 from 'n3';

const TTL_URLS = [
  'https://raw.githubusercontent.com/ambrosia2024/Semantic-Layer/main/Vocabulary/ambrosia-ontology.ttl',
  'https://raw.githubusercontent.com/ambrosia2024/Semantic-Layer/main/Vocabulary/ambrosia-plant-vocab.ttl',
  'https://raw.githubusercontent.com/ambrosia2024/Semantic-Layer/main/Vocabulary/ambrosia-pathogen-vocab.ttl',
  'https://raw.githubusercontent.com/ambrosia2024/Semantic-Layer/main/Vocabulary/ambrosia-netcdf-vocab.ttl',
];

const INTERNAL_NS = 'https://w3id.org/ambrosia/';
const FSKXO_NS = 'http://semanticlookup.zbmed.de/km/fskxo/';

const Layouts = Reactodia.defineLayoutWorker(() =>
  new Worker(new URL('@reactodia/workspace/layout.worker', import.meta.url))
);

// IRIs
const RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type';
const RDFS_SUBCLASS_OF = 'http://www.w3.org/2000/01/rdf-schema#subClassOf';
const RDFS_SUBPROPERTY_OF = 'http://www.w3.org/2000/01/rdf-schema#subPropertyOf';
const RDFS_DOMAIN = 'http://www.w3.org/2000/01/rdf-schema#domain';
const RDFS_RANGE = 'http://www.w3.org/2000/01/rdf-schema#range';

const OWL_CLASS = 'http://www.w3.org/2002/07/owl#Class';
const OWL_OBJECT_PROPERTY = 'http://www.w3.org/2002/07/owl#ObjectProperty';
const OWL_DATATYPE_PROPERTY = 'http://www.w3.org/2002/07/owl#DatatypeProperty';
const OWL_ANNOTATION_PROPERTY = 'http://www.w3.org/2002/07/owl#AnnotationProperty';
const OWL_INVERSE_OF = 'http://www.w3.org/2002/07/owl#inverseOf';
const OWL_EQUIVALENT_CLASS = 'http://www.w3.org/2002/07/owl#equivalentClass';
const OWL_EQUIVALENT_PROPERTY = 'http://www.w3.org/2002/07/owl#equivalentProperty';
const SKOS_EXACT_MATCH = 'http://www.w3.org/2004/02/skos/core#exactMatch';

// --- Tiny inline SVG icons as data URLs (no loaders needed)
function svgDataUrl(svg: string) {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

const ICON_CLASS = svgDataUrl(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="currentColor" d="M4 4h16v4H4V4zm0 6h10v4H4v-4zm0 6h16v4H4v-4z"/>
</svg>`);

const ICON_OBJPROP = svgDataUrl(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="currentColor" d="M7 7h10v10H7V7zm-3 5h2v0H4zm14 0h2v0h-2zM12 4h0v2h0V4zm0 14h0v2h0v-2z"/>
</svg>`);

const ICON_DATAPROP = svgDataUrl(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <path fill="currentColor" d="M4 6h16v3H4V6zm0 5h16v3H4v-3zm0 5h10v3H4v-3z"/>
</svg>`);

interface BridgePair {
  amb: string;
  ext: string;
  pred: string;
}

export default function App() {
  const { defaultLayout } = Reactodia.useWorker(Layouts);
  const [fskxoBridgePairs, setFskxoBridgePairs] = useState<BridgePair[]>([]);

  const { onMount } = Reactodia.useLoadedWorkspace(async ({ context, signal }) => {
    const { model, performLayout } = context;

    // 1) Load + parse all TTLs
    const allQuads: N3.Quad[] = [];
    const dataProvider = new Reactodia.RdfDataProvider({ acceptBlankNodes: false });

    for (const url of TTL_URLS) {
      try {
        const res = await fetch(url, { signal });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const text = await res.text();
        const quads = new N3.Parser().parse(text);
        allQuads.push(...quads);
        dataProvider.addGraph(quads);
      } catch (e) {
        console.warn('Load/Parse failed for', url, e);
      }
    }

    // 2) Collect types & schema relations
    const classes = new Set<string>();
    const objectProps = new Set<string>();
    const domains = new Map<string, Set<string>>();
    const ranges = new Map<string, Set<string>>();

    const addToMapSet = (map: Map<string, Set<string>>, key: string, value: string) => {
      const set = map.get(key) ?? new Set<string>();
      set.add(value);
      map.set(key, set);
    };

    const propertyIsInteresting = new Set<string>();
    const classIsInteresting = new Set<string>();
    const bridgePairs: BridgePair[] = [];

    allQuads.forEach(q => {
      const s = q.subject.termType === 'NamedNode' ? q.subject.value : null;
      const p = q.predicate.value;
      const o = q.object.termType === 'NamedNode' ? q.object.value : null;
      if (!s) return;

      if (p === RDF_TYPE && o === OWL_CLASS) classes.add(s);
      if (p === RDF_TYPE && o === OWL_OBJECT_PROPERTY) objectProps.add(s);

      if (p === RDFS_DOMAIN && o) {
        addToMapSet(domains, s, o);
        propertyIsInteresting.add(s);
        classIsInteresting.add(o);
      }
      if (p === RDFS_RANGE && o) {
        addToMapSet(ranges, s, o);
        propertyIsInteresting.add(s);
        classIsInteresting.add(o);
      }
      if ((p === OWL_INVERSE_OF || p === RDFS_SUBPROPERTY_OF) && o) {
        propertyIsInteresting.add(s);
        if (o.startsWith('http')) propertyIsInteresting.add(o);
      }
      if (p === RDFS_SUBCLASS_OF && o) {
        classIsInteresting.add(s);
        classIsInteresting.add(o);
      }

      // Collect Bridge Pairs (AMB <-> FSKXO)
      if (o) {
        const isBridgePred = p === OWL_EQUIVALENT_CLASS || p === OWL_EQUIVALENT_PROPERTY || p === SKOS_EXACT_MATCH;
        if (isBridgePred) {
          const sIsAmb = s.startsWith(INTERNAL_NS);
          const oIsAmb = o.startsWith(INTERNAL_NS);
          const sIsFsk = s.startsWith(FSKXO_NS);
          const oIsFsk = o.startsWith(FSKXO_NS);

          if (sIsAmb && oIsFsk) bridgePairs.push({ amb: s, ext: o, pred: p });
          else if (oIsAmb && sIsFsk) bridgePairs.push({ amb: o, ext: s, pred: p });
        }
      }
    });

    setFskxoBridgePairs(bridgePairs);

    // 3) Schema projection for internal classes
    const { namedNode, quad } = N3.DataFactory;
    const internalObjectPropsArray = Array.from(objectProps).filter(p => p.startsWith(INTERNAL_NS));
    internalObjectPropsArray.forEach(pIri => {
      const ds = domains.get(pIri);
      const rs = ranges.get(pIri);
      if (ds && rs) {
        ds.forEach(d => {
          rs.forEach(r => {
            if (d.startsWith(INTERNAL_NS) && r.startsWith(INTERNAL_NS)) {
              dataProvider.addGraph([quad(namedNode(d), namedNode(pIri), namedNode(r))]);
            }
          });
        });
      }
    });

    // 4) Create diagram
    await model.createNewDiagram({ dataProvider, signal });

    // 5) Decide what to render initial
    const nodesToShow = new Set<string>();
    classes.forEach(c => { if (c.startsWith(INTERNAL_NS)) nodesToShow.add(c); });
    classIsInteresting.forEach(c => { if (c.startsWith(INTERNAL_NS)) nodesToShow.add(c); });
    propertyIsInteresting.forEach(p => { if (p.startsWith(INTERNAL_NS)) nodesToShow.add(p); });

    nodesToShow.forEach(iri => model.createElement(iri));

    await model.requestData();
    await model.requestLinks();

    // 6) Initial link visibility
    [RDFS_SUBCLASS_OF, RDFS_DOMAIN, RDFS_RANGE, OWL_INVERSE_OF, RDFS_SUBPROPERTY_OF, ...internalObjectPropsArray]
      .forEach(pred => model.setLinkVisibility(pred, 'visible'));

    await performLayout({ signal });
  }, []);

  return (
    <div style={{ height: '100vh', width: '100vw' }}>
      <Reactodia.Workspace
        ref={onMount}
        defaultLayout={defaultLayout}
        typeStyleResolver={(types) => {
          if (types.includes(OWL_CLASS) || types.includes('http://www.w3.org/2000/01/rdf-schema#Class')) {
            return { icon: ICON_CLASS, iconMonochrome: true };
          }
          if (types.includes(OWL_OBJECT_PROPERTY)) {
            return { icon: ICON_OBJPROP, iconMonochrome: true, color: '#2e7d32' };
          }
          if (types.includes(OWL_DATATYPE_PROPERTY)) {
            return { icon: ICON_DATAPROP, iconMonochrome: true, color: '#1565c0' };
          }
          if (types.includes(OWL_ANNOTATION_PROPERTY)) {
            return { color: '#6a1b9a' };
          }
          return undefined;
        }}
      >
        <Reactodia.DefaultWorkspace
          menu={
            <Reactodia.ToolbarAction
              title="Show FSKXO bridge"
              onSelect={async () => {
                const workspace = (onMount as any).contextValue;
                if (!workspace) return;
                const { model, performLayout } = workspace;
                fskxoBridgePairs.forEach(pair => {
                  model.createElement(pair.amb);
                  model.createElement(pair.ext);
                  model.setLinkVisibility(pair.pred, 'visible');
                });
                await model.requestData();
                await model.requestLinks();
                await performLayout();
              }}
            >
              Show FSKXO bridge
            </Reactodia.ToolbarAction>
          }
          canvas={{
            elementTemplateResolver: (types) => {
              if (types.includes(OWL_OBJECT_PROPERTY) || types.includes(OWL_DATATYPE_PROPERTY) || types.includes(OWL_ANNOTATION_PROPERTY)) {
                return PropertyTemplate;
              }
              return undefined;
            },
            linkTemplateResolver: (linkType) => linkType ? FancyLinkTemplate : undefined,
          }}
        />
      </Reactodia.Workspace>
    </div>
  );
}

const PropertyTemplate: Reactodia.ElementTemplate = {
  ...Reactodia.RoundTemplate,
  renderElement: (props) => <Reactodia.RoundEntity {...props} />,
};

const FancyLinkTemplate: Reactodia.LinkTemplate = {
  markerTarget: {
    fill: '#4b4a67', stroke: '#4b4a67', width: 20, height: 12,
    d: 'm 20,5.88 -10.3,-5.95 0,5.6 -9.7,-5.6 0,11.82 9.7,-5.53 0,5.6 z',
  },
  renderLink: (props) => <Reactodia.StandardRelation {...props} pathProps={{ strokeWidth: 2 }} />,
};
