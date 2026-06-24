import sys, json, pathlib
import pdfplumber
for pdf in sys.argv[1:]:
    p = pathlib.Path(pdf)
    out = pathlib.Path('tmp/bacillus_example_extractions') / (p.stem + '.txt')
    table_out = pathlib.Path('tmp/bacillus_example_extractions') / (p.stem + '.tables.txt')
    with pdfplumber.open(str(p)) as doc:
        texts=[]
        table_text=[]
        for i,page in enumerate(doc.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ''
            texts.append(f'\n--- PAGE {i} ---\n{text}')
            try:
                tables=page.extract_tables()
            except Exception as e:
                tables=[]
            if tables:
                table_text.append(f'\n--- PAGE {i} TABLES ---')
                for tnum, table in enumerate(tables, start=1):
                    table_text.append(f'TABLE {tnum}')
                    for row in table:
                        table_text.append('\t'.join('' if c is None else str(c).replace('\n',' | ') for c in row))
        out.write_text('\n'.join(texts), encoding='utf-8')
        table_out.write_text('\n'.join(table_text), encoding='utf-8')
        print(json.dumps({'pdf': str(p), 'pages': len(doc.pages), 'text': str(out), 'tables': str(table_out)}))
