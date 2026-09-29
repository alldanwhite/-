"""Render the 'Лучшие места города' Saint Petersburg map as a seamless loop.

    python scripts/render.py video video/spb_best_places.mp4 [--crf=17]
    python scripts/render.py gif   video/spb_best_places.gif [--size=720] [--fps=25] [--colors=128]
    python scripts/render.py still 2.0 frame.png
    python scripts/render.py sheet 0,1,2,3,4,5 sheet.png

Run from the repository root (reads data/prepared.pkl and fonts/).
The map is static; every dot twinkles a whole number of times per LOOP
seconds, so the last frame flows straight into the first one.
"""
import math, os, pickle, shutil, sys, subprocess, time
import multiprocessing as mp
import numpy as np, skia, uharfbuzz as hb
import shapely
import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------- config
SIZE = 1080            # output px (square)
SS = 2                 # supersampling factor
FPS = 60
LOOP = 6.0             # seconds, one seamless cycle
R = 6378137.0

CENTER = (30.307, 59.9512)   # lon, lat of the frame centre
GROUND_W = 14200.0          # ground metres across the frame

WHITE = skia.Color(255, 255, 255)
INK = (10, 10, 10)
WATER = skia.Color(238, 238, 238)
RED = (255, 45, 32)

def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

CX, CY = merc(*CENTER)
MW = GROUND_W / math.cos(math.radians(CENTER[1]))   # mercator metres across frame
K = SIZE / MW

def to_px(xy):
    xy = np.asarray(xy, float)
    return np.column_stack(((xy[:, 0] - CX) * K + SIZE / 2, (CY - xy[:, 1]) * K + SIZE / 2))

def lonlat_px(lon, lat):
    x, y = merc(lon, lat)
    return (x - CX) * K + SIZE / 2, (CY - y) * K + SIZE / 2

# ---------------------------------------------------------------- dots
# landmarks get the big dots with wide waves; the rest of the dots sit on real
# cafes, bars, museums, galleries, theatres, parks... (Overture places)
LANDMARKS = [
    (30.3159, 59.9391), (30.3062, 59.9341), (30.3288, 59.9400), (30.3165, 59.9502),
    (30.2892, 59.9290), (30.3957, 59.9490), (30.2207, 59.9729), (30.3355, 59.9446),
    (30.3880, 59.9213), (30.2415, 59.9247), (30.3710, 59.9460), (30.2513, 59.9318),
]
DOT_SPACING = 37.5       # min distance between dots, px
DOT_COUNT = 110          # dots in total; extra ones are thinned out at random
LANDMARK_SPACING = 40.0  # keep small dots a bit further from the big ones
DOT_AREA = (16, 208, SIZE - 16, SIZE - 16)   # below the title
CREDIT_BOX = (SIZE - 232, SIZE - 44)         # no dots under the credit

# ---------------------------------------------------------------- map paths
ROAD_STYLE = [  # class, width (base px), alpha
    ('footway',        0.22, 0.20),
    ('path',           0.22, 0.14),
    ('steps',          0.22, 0.14),
    ('cycleway',       0.22, 0.20),
    ('pedestrian',     0.45, 0.45),
    ('service',        0.30, 0.35),
    ('track',          0.30, 0.25),
    ('living_street',  0.50, 0.75),
    ('unknown',        0.45, 0.60),
    ('unclassified',   0.55, 0.80),
    ('residential',    0.60, 0.85),
    ('rail_standard_gauge', 0.45, 0.35),
    ('tertiary',       0.85, 0.95),
    ('secondary',      1.05, 1.00),
    ('primary',        1.35, 1.00),
    ('trunk',          1.60, 1.00),
    ('motorway',       1.70, 1.00),
]

def build_paths(data):
    roads = {}
    for cls, _, _ in ROAD_STYLE:
        p = skia.Path()
        for ln in data['roads'].get(cls, []):
            q = to_px(ln)
            p.addPoly([skia.Point(float(x), float(y)) for x, y in q], False)
        roads[cls] = p
    water = skia.Path()
    water.setFillType(skia.PathFillType.kEvenOdd)
    for ring in data['water']:
        q = to_px(ring)
        water.addPoly([skia.Point(float(x), float(y)) for x, y in q], True)
    return roads, water

def water_shape(data):
    """Water as a shapely geometry in frame px, to keep dots on land."""
    rings = [to_px(r) for r in data['water']]
    polys = [shapely.Polygon(r) for r in rings if len(r) >= 4]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    # even-odd rings: xor them together (islands inside rivers stay land)
    acc = shapely.Polygon()
    for p in polys:
        acc = shapely.symmetric_difference(acc, p)
    shapely.prepare(acc)
    return acc

# ---------------------------------------------------------------- text
class Title:
    def __init__(self, font_path, text, size, tracking=0.0):
        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        font = hb.Font(face)
        buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
        hb.shape(font, buf, {'kern': True, 'liga': True})
        scale = size / face.upem
        self.glyphs, self.pos = [], []
        x = 0.0
        for info, p in zip(buf.glyph_infos, buf.glyph_positions):
            self.glyphs.append(info.codepoint)
            self.pos.append((x + p.x_offset * scale, -p.y_offset * scale))
            x += p.x_advance * scale + tracking * size
        self.width = x - tracking * size
        self.font = skia.Font(skia.Typeface.MakeFromFile(font_path), size)
        self.font.setEdging(skia.Font.Edging.kAntiAlias)
        self.font.setSubpixel(True)
        self.font.setHinting(skia.FontHinting.kNone)

    def draw(self, canvas, x, y, paint):
        builder = skia.TextBlobBuilder()
        builder.allocRunPos(self.font, self.glyphs, [skia.Point(px, py) for px, py in self.pos])
        canvas.drawTextBlob(builder.make(), x, y, paint)

# ---------------------------------------------------------------- helpers
def clamp(v, a=0.0, b=1.0): return a if v < a else b if v > b else v
def ease_out_cubic(t): t = clamp(t); return 1 - (1 - t) ** 3

def color4(rgb, a):
    return skia.Color4f(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a)

TITLE_TEXT = 'Лучшие места города'
CREDIT_TEXT = '© OpenStreetMap contributors'   # ODbL attribution for the map data
TITLE_SIZE = 76
TITLE_BASELINE = 112
TITLE_ALIGN = 'center'   # or 'left'
TITLE_MARGIN = 56        # left margin when left-aligned
FADE_SOLID, FADE_END = 146, 196  # white band under the title, then a short fade into the map (px)
BLINK_OUT, BLINK_OFF, BLINK_IN = 0.25, 0.30, 0.25   # seconds: fade out, stay dark, come back

# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, data):
        self.data = data
        self.dots = self.pick_dots(data)
        self.base = self.render_base(data)

    # ---- dot layout: landmarks first, then a poisson-disk pick of real places
    def pick_dots(self, data):
        rng = np.random.default_rng(11)
        water = water_shape(data)
        x0, y0, x1, y1 = DOT_AREA

        def ok(x, y):
            if not (x0 <= x <= x1 and y0 <= y <= y1): return False
            if x >= CREDIT_BOX[0] and y >= CREDIT_BOX[1]: return False
            return not shapely.contains_xy(water, x, y)

        cell = DOT_SPACING
        grid = {}
        def near(x, y, d):
            gx, gy = int(x // cell), int(y // cell)
            r = int(math.ceil(d / cell))
            for i in range(gx - r, gx + r + 1):
                for j in range(gy - r, gy + r + 1):
                    for (px, py, pd) in grid.get((i, j), ()):
                        if (px - x) ** 2 + (py - y) ** 2 < max(d, pd) ** 2:
                            return True
            return False
        def add(x, y, d):
            grid.setdefault((int(x // cell), int(y // cell)), []).append((x, y, d))

        dots = []
        for lo, la in LANDMARKS:
            x, y = lonlat_px(lo, la)
            if ok(x, y):
                dots.append(dict(x=x, y=y, kind='big'))
                add(x, y, LANDMARK_SPACING)
        pts = to_px(data['pois'])
        for k in rng.permutation(len(pts)):
            x, y = pts[k]
            if ok(x, y) and not near(x, y, DOT_SPACING):
                dots.append(dict(x=float(x), y=float(y), kind='small'))
                add(x, y, DOT_SPACING)
        # thin out at random, keeping the landmarks: the centre stays denser
        big = [d for d in dots if d['kind'] == 'big']
        rest = [d for d in dots if d['kind'] != 'big']
        keep = sorted(rng.permutation(len(rest))[:max(0, DOT_COUNT - len(big))])
        dots = big + [rest[i] for i in keep]
        # a quarter of the small dots become medium ones with their own waves
        for d in dots:
            if d['kind'] == 'small' and rng.random() < 0.25:
                d['kind'] = 'mid'
        # rhythm: whole number of flashes per loop -> seamless
        for d in dots:
            cycles = {'big': (1,), 'mid': (1, 2), 'small': (1, 1, 2)}[d['kind']]
            d['n'] = int(rng.choice(cycles))
            d['phase'] = float(rng.random())
        return dots

    # ---- static layer: map, title veil, title, credit (drawn once at SSx)
    def render_base(self, data):
        roads, water = build_paths(data)
        surf = skia.Surface(SIZE * SS, SIZE * SS)
        with surf as c:
            c.clear(WHITE)
            c.scale(SS, SS)
            c.drawPath(water, skia.Paint(AntiAlias=True, Color=WATER))
            sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style,
                            StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
            for cls, w, a in ROAD_STYLE:
                sp.setStrokeWidth(w)
                sp.setColor(color4(INK, a))
                c.drawPath(roads[cls], sp)
            g = skia.Paint()
            g.setShader(skia.GradientShader.MakeLinear(
                [skia.Point(0, 0), skia.Point(0, FADE_END)],
                [color4((255, 255, 255), 1), color4((255, 255, 255), 1), color4((255, 255, 255), 0)],
                [0.0, FADE_SOLID / FADE_END, 1.0]))
            c.drawRect(skia.Rect(0, 0, SIZE, FADE_END), g)
            title = Title('fonts/InterDisplay-700-cyr.ttf', TITLE_TEXT, TITLE_SIZE, tracking=-0.012)
            tx = TITLE_MARGIN if TITLE_ALIGN == 'left' else (SIZE - title.width) / 2
            title.draw(c, tx, TITLE_BASELINE, skia.Paint(AntiAlias=True, Color=skia.Color(*INK)))
            credit = Title('fonts/Inter-400-latin.ttf', CREDIT_TEXT, 13, tracking=0.005)
            w = credit.width
            xr, yb = SIZE - 14, SIZE - 12
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect(xr - w - 14, yb - 21, xr, yb), 6, 6),
                        skia.Paint(AntiAlias=True, Color4f=color4((255, 255, 255), 0.88)))
            credit.draw(c, xr - w - 7, yb - 6.5, skia.Paint(AntiAlias=True, Color4f=color4((120, 120, 120), 1)))
        return surf.makeImageSnapshot()

    # ---- one dot at loop time t
    def draw_dot(self, c, d, t, paints):
        """A dot is lit, blinks off for a moment and comes back with a small pop.

        Between blinks the dot is perfectly still, which keeps the GIF small.
        """
        fill, ring, glow = paints
        x, y, kind = d['x'], d['y'], d['kind']
        period = LOOP / d['n']
        cyc = d['n'] * t / LOOP - d['phase']
        since = (cyc % 1.0) * period                  # seconds since this blink began
        dark = BLINK_OUT + BLINK_OFF
        if since < BLINK_OUT:
            v = math.cos(0.5 * math.pi * since / BLINK_OUT) ** 2
        elif since < dark:
            v = 0.0
        elif since < dark + BLINK_IN:
            v = math.sin(0.5 * math.pi * (since - dark) / BLINK_IN) ** 2
        else:
            v = 1.0
        back = since - dark                           # seconds since it started coming back
        pop = 1.0 + 0.22 * math.exp(-((back - 0.20) / 0.14) ** 2) if back > 0 else 1.0

        r0 = {'big': 6.6, 'mid': 4.6, 'small': 3.6}[kind]
        if kind != 'small' and back > 0:
            life, reach = (1.8, 42.0) if kind == 'big' else (1.3, 18.0)
            waves = (0.0, 0.35) if kind == 'big' else (0.0,)
            for delay in waves:
                q = (back - delay) / life
                if not 0.0 < q < 1.0:
                    continue
                rr = r0 + reach * ease_out_cubic(q)
                a = (1.0 - q) ** 2
                if kind == 'big' and delay == 0.0:
                    fill.setColor(color4(RED, 0.12 * a))
                    c.drawCircle(x, y, rr, fill)
                ring.setStrokeWidth(1.5 if kind == 'big' else 1.2)
                ring.setColor(color4(RED, 0.65 * a))
                c.drawCircle(x, y, rr, ring)

        if v <= 0.0:
            return
        k = pop * (0.55 + 0.45 * v)
        glow.setColor(color4(RED, 0.22 * v))
        c.drawCircle(x, y, r0 * 2.0 * k, glow)
        fill.setColor(color4((255, 255, 255), v))
        c.drawCircle(x, y, (r0 + 1.5) * k, fill)
        fill.setColor(color4(RED, v))
        c.drawCircle(x, y, r0 * k, fill)

    def draw(self, c, t, size):
        c.clear(WHITE)
        c.save()
        c.scale(size / SIZE, size / SIZE)
        c.drawImageRect(self.base, skia.Rect(0, 0, SIZE, SIZE),
                        skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear))
        fill = skia.Paint(AntiAlias=True)
        ring = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
        glow = skia.Paint(AntiAlias=True)
        glow.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 3.0))
        for d in self.dots:
            self.draw_dot(c, d, t, (fill, ring, glow))
        c.restore()

def load_scene():
    with open('data/prepared.pkl', 'rb') as f:
        data = pickle.load(f)
    return Scene(data)

def render_frame(scene, t, size, big, small):
    """Draw at SS x size, then downsample to size."""
    with big as c:
        scene.draw(c, t, size * SS)
    img = big.makeImageSnapshot()
    with small as c:
        c.drawImageRect(img, skia.Rect(0, 0, size, size), skia.SamplingOptions(skia.CubicResampler.Mitchell()))
    return small.makeImageSnapshot()

# ---------------------------------------------------------------- parallel rendering
WORKERS = 3
_W = {}

def _worker_init(size):
    _W['scene'] = load_scene()
    _W['size'] = size
    _W['big'], _W['small'] = skia.Surface(size * SS, size * SS), skia.Surface(size, size)

def _worker_frame(t):
    img = render_frame(_W['scene'], t, _W['size'], _W['big'], _W['small'])
    return img.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType).tobytes()

def encode(out, size, fps, codec_args):
    n = int(round(LOOP * fps))
    ff = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error',
                           '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{size}x{size}', '-r', str(fps), '-i', '-']
                          + codec_args + [out], stdin=subprocess.PIPE)
    t0 = time.time()
    with mp.Pool(WORKERS, initializer=_worker_init, initargs=(size,)) as pool:
        for i, buf in enumerate(pool.imap(_worker_frame, [i / fps for i in range(n)], chunksize=2)):
            ff.stdin.write(buf)
            if i % 120 == 0:
                print(f'frame {i}/{n}  {time.time() - t0:.0f}s', flush=True)
    ff.stdin.close(); ff.wait()

def arg(name, default):
    return next((a.split('=', 1)[1] for a in sys.argv if a.startswith(f'--{name}=')), default)

if __name__ == '__main__':
    mode = sys.argv[1]
    t0 = time.time()
    if mode in ('still', 'sheet'):
        scene = load_scene()
        print(f'scene built {time.time() - t0:.1f}s, {len(scene.dots)} dots', flush=True)
        big, small = skia.Surface(SIZE * SS, SIZE * SS), skia.Surface(SIZE, SIZE)
    if mode == 'still':
        render_frame(scene, float(sys.argv[2]), SIZE, big, small).save(sys.argv[3], skia.kPNG)
    elif mode == 'sheet':   # contact sheet of several moments
        times = [float(v) for v in sys.argv[2].split(',')]
        cols = 3; rows = (len(times) + cols - 1) // cols; cell = 540
        sheet = skia.Surface(cols * cell, rows * cell)
        with sheet as sc:
            sc.clear(skia.Color(128, 128, 128))
            for i, t in enumerate(times):
                img = render_frame(scene, t, SIZE, big, small)
                sc.drawImageRect(img, skia.Rect.MakeXYWH((i % cols) * cell + 2, (i // cols) * cell + 2, cell - 4, cell - 4),
                                 skia.SamplingOptions(skia.CubicResampler.Mitchell()))
        sheet.makeImageSnapshot().save(sys.argv[3], skia.kPNG)
    elif mode == 'video':   # silent seamless mp4 (what messengers call a "GIF")
        encode(sys.argv[2], SIZE, FPS, [
            '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', arg('crf', '17'), '-profile:v', 'high', '-level', '4.2',
            # keep the key frame's quality close to the other frames so the loop point doesn't pop
            '-x264-params', 'mbtree=0:ipratio=1.0',
            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
            '-an', '-movflags', '+faststart'])
    elif mode == 'gif':     # real GIF, infinite loop, one global palette
        size, fps = int(arg('size', '720')), int(arg('fps', '25'))
        out = sys.argv[2]
        raw = out + '.raw.gif'
        encode(raw, size, fps, [
            '-filter_complex', f'split[a][b];[a]palettegen=max_colors={arg("colors", "128")}:stats_mode=full:reserve_transparent=1[p];'
                               '[b][p]paletteuse=dither=none:diff_mode=rectangle',
            '-loop', '0'])
        if shutil.which('gifsicle'):   # store only what changes between frames
            # lossless on purpose: lossy GIF errors pile up and the loop point would jump
            lossy = int(arg('lossy', '0'))
            subprocess.run(['gifsicle', '-O3'] + ([f'--lossy={lossy}'] if lossy else []) + [raw, '-o', out], check=True)
            os.remove(raw)
        else:
            os.replace(raw, out)
    print('done', round(time.time() - t0, 1), 's')
