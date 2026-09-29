"""Download the map layers for central Saint Petersburg from Overture Maps.

Overture publishes OpenStreetMap-derived data as GeoParquet in a public S3
bucket. Each file's row groups carry bbox statistics, so only the row groups
that intersect our area are read.

    python scripts/fetch_overture.py segment   # roads   -> data/segment.parquet
    python scripts/fetch_overture.py water     # water   -> data/water.parquet
"""
import os, sys, time, concurrent.futures as cf
import pyarrow as pa, pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc

RELEASE = '2026-09-23.1'
BBOX = (30.08, 59.84, 30.52, 60.05)  # xmin, ymin, xmax, ymax (lon/lat)

LAYERS = {
    'segment': ('transportation', 'segment', ['id', 'subtype', 'class', 'subclass', 'road_flags', 'geometry', 'bbox']),
    'water':   ('base', 'water', ['id', 'subtype', 'class', 'is_salt', 'names', 'geometry', 'bbox']),
}

s3 = fs.S3FileSystem(anonymous=True, region='us-west-2')

def row_groups_in_bbox(path):
    md = pq.ParquetFile(path, filesystem=s3).metadata
    names = [md.row_group(0).column(j).path_in_schema for j in range(md.num_columns)]
    ix = {n: names.index(n) for n in ('bbox.xmin', 'bbox.xmax', 'bbox.ymin', 'bbox.ymax')}
    hits = []
    for i in range(md.num_row_groups):
        st = {k: md.row_group(i).column(j).statistics for k, j in ix.items()}
        if any(s is None or not s.has_min_max for s in st.values()):
            hits.append(i)
        elif (st['bbox.xmin'].min <= BBOX[2] and st['bbox.xmax'].max >= BBOX[0] and
              st['bbox.ymin'].min <= BBOX[3] and st['bbox.ymax'].max >= BBOX[1]):
            hits.append(i)
    return path, hits

def read_filtered(path, groups, columns):
    tbl = pq.ParquetFile(path, filesystem=s3).read_row_groups(groups, columns=columns)
    b = tbl.column('bbox')
    f = lambda k: pc.struct_field(b, k)
    keep = pc.and_(pc.and_(pc.less_equal(f('xmin'), BBOX[2]), pc.greater_equal(f('xmax'), BBOX[0])),
                   pc.and_(pc.less_equal(f('ymin'), BBOX[3]), pc.greater_equal(f('ymax'), BBOX[1])))
    return tbl.filter(keep)

def fetch(layer):
    theme, typ, columns = LAYERS[layer]
    out = f'data/{layer}.parquet'
    t = time.time()
    base = f'overturemaps-us-west-2/release/{RELEASE}/theme={theme}/type={typ}/'
    paths = [i.path for i in s3.get_file_info(fs.FileSelector(base)) if i.path.endswith('.parquet')]
    with cf.ThreadPoolExecutor(16) as ex:
        todo = [(p, h) for p, h in ex.map(row_groups_in_bbox, paths) if h]
    print(f'{theme}/{typ}: {sum(len(h) for _, h in todo)} row groups in {len(todo)} of {len(paths)} files', flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        tables = list(ex.map(lambda ph: read_filtered(ph[0], ph[1], columns), todo))
    tbl = pa.concat_tables(tables)
    os.makedirs('data', exist_ok=True)
    pq.write_table(tbl, out)
    print(f'-> {out}: {tbl.num_rows} features, {time.time() - t:.1f}s')

if __name__ == '__main__':
    for layer in sys.argv[1:] or LAYERS:
        fetch(layer)
