"""Project Overture layers to Web Mercator and pack them for the renderer."""
import pickle
import numpy as np, shapely, pyarrow.parquet as pq

R = 6378137.0
def merc(lon, lat):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

# area kept for rendering (lon/lat), a bit larger than the widest frame
KEEP = (30.10, 59.85, 30.50, 60.04)
kx0, ky0 = merc(KEEP[0], KEEP[1]); kx1, ky1 = merc(KEEP[2], KEEP[3])

def to_merc(geom):
    return shapely.transform(geom, lambda c: np.column_stack(merc(c[:, 0], c[:, 1])))

def lines_of(g):
    if g.is_empty: return []
    if g.geom_type == 'LineString': return [np.asarray(g.coords)]
    if g.geom_type in ('MultiLineString', 'GeometryCollection'):
        return [a for p in g.geoms for a in lines_of(p)]
    return []

def rings_of(g):
    if g.is_empty: return []
    if g.geom_type == 'Polygon':
        return [np.asarray(g.exterior.coords)] + [np.asarray(i.coords) for i in g.interiors]
    if g.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [a for p in g.geoms for a in rings_of(p)]
    return []

out = {'merc_keep': (float(kx0), float(ky0), float(kx1), float(ky1))}

# ---- roads -------------------------------------------------------------
seg = pq.read_table('data/segment.parquet', columns=['subtype', 'class', 'road_flags', 'geometry']).to_pylist()
roads = {}
for r in seg:
    cls = r['class'] if r['subtype'] == 'road' else ('rail_' + str(r['class']) if r['subtype'] == 'rail' else None)
    if cls is None: continue
    flags = set()
    for rf in r['road_flags'] or []:
        flags.update(rf['values'] or [])
    if 'is_tunnel' in flags or 'is_underground' in flags:
        cls += '_tunnel'
    g = to_merc(shapely.from_wkb(r['geometry']))
    g = shapely.clip_by_rect(g, kx0, ky0, kx1, ky1)
    for ln in lines_of(g):
        if len(ln) >= 2:
            roads.setdefault(cls, []).append(ln.astype(np.float64))
out['roads'] = roads
print({k: len(v) for k, v in sorted(roads.items(), key=lambda kv: -len(kv[1]))})

# ---- water -------------------------------------------------------------
wat = pq.read_table('data/water.parquet', columns=['subtype', 'class', 'names', 'geometry']).to_pylist()
polys = []
for r in wat:
    if r['subtype'] in ('physical', 'human_made', 'spring'): continue
    if r['class'] in ('swimming_pool', 'wastewater', 'fountain'): continue
    g = shapely.from_wkb(r['geometry'])
    if g.geom_type not in ('Polygon', 'MultiPolygon'): continue
    polys.append(to_merc(g))
water = shapely.union_all(shapely.make_valid(np.array(polys)))
water = shapely.clip_by_rect(water, kx0, ky0, kx1, ky1)
out['water'] = rings_of(water)
print('water rings', len(out['water']), 'area km2 (merc)', water.area / 1e6)

# ---- points of interest (where the dots may sit) ---------------------------
POI_CATEGORIES = {
    'restaurant', 'bar', 'casual_eatery', 'coffee_shop', 'cafe', 'historic_site',
    'park', 'arts_and_entertainment', 'museum', 'music_venue', 'art_gallery',
    'theatre_venue', 'monument', 'dance_club', 'movie_theater', 'stadium_arena',
}
pl = pq.read_table('data/place.parquet', columns=['basic_category', 'confidence', 'geometry']).to_pylist()
pois = []
for r in pl:
    if r['basic_category'] not in POI_CATEGORIES or (r['confidence'] or 0) < 0.6:
        continue
    g = shapely.from_wkb(r['geometry'])
    if g.geom_type != 'Point':
        continue
    x, y = merc(g.x, g.y)
    if kx0 <= x <= kx1 and ky0 <= y <= ky1:
        pois.append((float(x), float(y)))
out['pois'] = np.array(pois)
print('pois', len(pois))

with open('data/prepared.pkl', 'wb') as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
print('saved')
