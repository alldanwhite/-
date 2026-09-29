"""Render the 'Лучшие места города' Saint Petersburg map animation.

    python scripts/render.py video video/spb_best_places.mp4 [--crf=19] [--range=4:6]
    python scripts/render.py still 7.0 frame.png [--labels]
    python scripts/render.py sheet 0.5,2,6,11.9 sheet.png [--labels]

Run from the repository root (reads data/prepared.pkl and fonts/).
"""
import math, pickle, sys, subprocess, time
import multiprocessing as mp
import numpy as np, skia, uharfbuzz as hb
import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------- config
SIZE = 1080            # output px (square)
SS = 2                 # supersampling factor
FPS = 60
DURATION = 12.0        # seconds
R = 6378137.0

CENTER = (30.307, 59.9512)   # lon, lat of the base frame centre
GROUND_W = 14200.0          # ground metres across the base frame

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

# ---------------------------------------------------------------- places
PLACES = [  # name, lon, lat, major
    ('Дворцовая площадь',        30.3159, 59.9391, False),
    ('Исаакиевский собор',       30.3062, 59.9341, False),
    ('Спас на Крови',            30.3288, 59.9400, False),
    ('Петропавловская крепость', 30.3165, 59.9502, True),
    ('Новая Голландия',          30.2892, 59.9290, True),
    ('Смольный собор',           30.3957, 59.9490, False),
    ('Газпром Арена',            30.2207, 59.9729, True),
    ('Летний сад',               30.3355, 59.9446, False),
    ('Александро-Невская лавра', 30.3880, 59.9213, False),
    ('Севкабель Порт',           30.2415, 59.9247, True),
    ('Таврический сад',          30.3710, 59.9460, False),
    ('Эрарта',                   30.2513, 59.9318, False),
    ('Аптекарский огород',       30.3238, 59.9700, False),
    ('Лофт Проект Этажи',        30.3562, 59.9219, False),
    ('Нарвские ворота',          30.2744, 59.9008, False),
    ('Ткачи',                    30.3413, 59.9154, False),
    ('Морской фасад',            30.2095, 59.9535, False),
    ('Финляндский вокзал',       30.3558, 59.9557, False),
]

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

# ---------------------------------------------------------------- text
class Title:
    def __init__(self, font_path, text, size, tracking=0.0):
        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        font = hb.Font(face)
        upem = face.upem
        buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
        hb.shape(font, buf, {'kern': True, 'liga': True})
        scale = size / upem
        self.glyphs, self.pos = [], []
        x = 0.0
        for info, p in zip(buf.glyph_infos, buf.glyph_positions):
            self.glyphs.append(info.codepoint)
            self.pos.append((x + p.x_offset * scale, -p.y_offset * scale))
            x += p.x_advance * scale + tracking * size
        self.width = x - tracking * size
        tf = skia.Typeface.MakeFromFile(font_path)
        self.font = skia.Font(tf, size)
        self.font.setEdging(skia.Font.Edging.kAntiAlias)
        self.font.setSubpixel(True)
        self.font.setHinting(skia.FontHinting.kNone)

    def draw(self, canvas, x, y, paint):
        builder = skia.TextBlobBuilder()
        builder.allocRunPos(self.font, self.glyphs, [skia.Point(px, py) for px, py in self.pos])
        canvas.drawTextBlob(builder.make(), x, y, paint)

# ---------------------------------------------------------------- easing
def clamp(v, a=0.0, b=1.0): return a if v < a else b if v > b else v
def smooth(t): t = clamp(t); return t * t * (3 - 2 * t)
def ease_out_cubic(t): t = clamp(t); return 1 - (1 - t) ** 3
def ease_in_out_sine(t): t = clamp(t); return 0.5 - 0.5 * math.cos(math.pi * t)
def ease_out_back(t, s=1.7): t = clamp(t) - 1; return 1 + t * t * ((s + 1) * t + s)

# ---------------------------------------------------------------- scene
FOCUS = (30.312, 59.9395)       # camera push-in target
REVEAL_T = 2.6
# moments when one place 'lights up' with a big double wave
BURSTS = [(5.0, 'Новая Голландия'), (7.2, 'Дворцовая площадь'), (9.3, 'Севкабель Порт'), (10.2, 'Смольный собор')]
BURST_T = 1.7
TITLE_TEXT = 'Лучшие места города'
CREDIT_TEXT = '© OpenStreetMap contributors'   # ODbL attribution for the map data
TITLE_SIZE = 70
TITLE_BASELINE = 128
FADE_SOLID, FADE_END = 150, 285  # white veil under the title (px)

def color4(rgb, a):
    return skia.Color4f(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a)

class Scene:
    def __init__(self, data, font_path, debug_labels=False):
        self.roads, self.water = build_paths(data)
        self.title = Title(font_path, TITLE_TEXT, TITLE_SIZE, tracking=-0.012)
        self.credit = Title('fonts/Inter-400-latin.ttf', CREDIT_TEXT, 13, tracking=0.005)
        self.debug_labels = debug_labels
        self.focus = lonlat_px(*FOCUS)
        rng = np.random.default_rng(7)
        order = rng.permutation(len(PLACES))
        self.dots = []
        for k, idx in enumerate(order):
            name, lo, la, major = PLACES[idx]
            x, y = lonlat_px(lo, la)
            self.dots.append(dict(
                name=name, x=x, y=y, major=major,
                appear=1.25 + 0.16 * k,                  # staggered pop-in
                period=float(rng.uniform(2.1, 3.0)),     # own rhythm per dot
                phase=float(rng.uniform(0.0, 1.0)),
            ))

    # camera: slow continuous push-in, zoom 1.00 -> 1.12
    def camera(self, t):
        return 1.0 + 0.12 * ease_in_out_sine(t / DURATION)

    def to_screen(self, x, y, z):
        fx, fy = self.focus
        return fx + (x - fx) * z, fy + (y - fy) * z

    def draw_map(self, canvas, z):
        fx, fy = self.focus
        canvas.save()
        canvas.translate(fx, fy); canvas.scale(z, z); canvas.translate(-fx, -fy)
        paint = skia.Paint(AntiAlias=True, Color=WATER)
        canvas.drawPath(self.water, paint)
        sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style,
                        StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
        for cls, w, a in ROAD_STYLE:
            sp.setStrokeWidth(w)
            sp.setColor(color4(INK, a))
            canvas.drawPath(self.roads[cls], sp)
        canvas.restore()

    def draw_reveal_mask(self, canvas, t):
        # radial reveal from the focus point during the intro
        u = ease_in_out_sine(t / REVEAL_T)
        if u >= 1.0:
            return
        fx, fy = self.focus
        feather = 260.0
        rad = 110 + u * (1080 + feather)
        inner = max(rad - feather, 0.0)
        g = skia.GradientShader.MakeRadial(
            skia.Point(fx, fy), rad,
            [color4((0, 0, 0), 1.0), color4((0, 0, 0), 1.0), color4((0, 0, 0), 0.0)],
            [0.0, inner / rad, 1.0])
        p = skia.Paint(Shader=g, BlendMode=skia.BlendMode.kDstIn)
        canvas.drawRect(skia.Rect(0, 0, SIZE, SIZE), p)

    def draw_dot(self, canvas, d, t, z):
        age = t - d['appear']
        if age <= 0:
            return
        x, y = self.to_screen(d['x'], d['y'], z)
        major = d['major']
        r0 = 7.0 if major else 5.5          # red core radius
        rmax = 46.0 if major else 32.0      # ripple reach
        ring_w = 2.2 if major else 1.8      # white outline
        pop = ease_out_back(age / 0.55, 2.2)
        vis = smooth(age / 0.18)

        p = skia.Paint(AntiAlias=True)
        ps = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.5)
        # ripples: two waves per cycle (half a period apart); the first one
        # starts exactly at pop-in, so the motion is continuous
        period = d['period']
        for k in range(2):
            ak = age - 0.5 * k * period
            if ak <= 0: continue
            ph = (ak / period) % 1.0
            rr = r0 + (rmax - r0) * ease_out_cubic(ph)
            a = (1.0 - ph) ** 2
            p.setColor(color4(RED, 0.16 * a * vis))
            canvas.drawCircle(x, y, rr, p)
            ps.setColor(color4(RED, 0.60 * a * vis))
            canvas.drawCircle(x, y, rr, ps)

        # occasional big event: a wide double wave + strong flash
        burst = 0.0
        for bt, bname in BURSTS:
            if bname != d['name']: continue
            for j in range(2):
                q = (t - bt - 0.22 * j) / BURST_T
                if 0.0 < q < 1.0:
                    rr = r0 + (92.0 - r0) * ease_out_cubic(q)
                    a = (1.0 - q) ** 1.6
                    ps2 = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.0 - 0.8 * q)
                    ps2.setColor(color4(RED, 0.75 * a))
                    canvas.drawCircle(x, y, rr, ps2)
                    if j == 0:
                        p.setColor(color4(RED, 0.10 * a))
                        canvas.drawCircle(x, y, rr, p)
            q = (t - bt) / 0.9
            burst = max(burst, math.exp(-((q - 0.12) / 0.16) ** 2))

        # blink: the core flashes at the start of every cycle (wrap-safe)
        ph0 = (age / period) % 1.0
        dph = (ph0 - 0.04 + 0.5) % 1.0 - 0.5
        flash = math.exp(-(dph / 0.07) ** 2) * smooth((age - 0.45) / 0.3)
        flash = max(flash, 1.6 * burst)
        glow = skia.Paint(AntiAlias=True, Color4f=color4(RED, (0.18 + 0.32 * flash) * vis))
        glow.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 5.0))
        canvas.drawCircle(x, y, (r0 + 3.5) * pop, glow)

        s = pop * (1.0 + 0.16 * flash)
        p.setColor(color4((255, 255, 255), vis))
        canvas.drawCircle(x, y, (r0 + ring_w) * s, p)
        p.setColor(color4(RED, vis))
        canvas.drawCircle(x, y, r0 * s, p)
        if self.debug_labels:
            lp = skia.Paint(AntiAlias=True, Color=skia.Color(0, 90, 255))
            canvas.drawString(d['name'], x + 9, y + 4, skia.Font(skia.Typeface('DejaVu Sans'), 11), lp)

    def draw_title(self, canvas, t):
        g = skia.Paint()
        g.setShader(skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(0, FADE_END)],
            [color4((255, 255, 255), 1), color4((255, 255, 255), 1), color4((255, 255, 255), 0)],
            [0.0, FADE_SOLID / FADE_END, 1.0]))
        canvas.drawRect(skia.Rect(0, 0, SIZE, FADE_END), g)
        # per-glyph fade/rise
        x0 = (SIZE - self.title.width) / 2
        font = self.title.font
        for i, (gid, (gx, gy)) in enumerate(zip(self.title.glyphs, self.title.pos)):
            u = ease_out_cubic((t - 0.35 - 0.035 * i) / 0.75)
            if u <= 0: continue
            p = skia.Paint(AntiAlias=True, Color4f=color4(INK, u))
            b = skia.TextBlobBuilder()
            b.allocRunPos(font, [gid], [skia.Point(gx, gy)])
            canvas.drawTextBlob(b.make(), x0, TITLE_BASELINE + 14 * (1 - u), p)

    def draw_credit(self, canvas, t):
        u = smooth((t - 1.2) / 1.2)
        if u <= 0: return
        w = self.credit.width
        x1, y1 = SIZE - 14, SIZE - 12
        box = skia.RRect.MakeRectXY(skia.Rect(x1 - w - 14, y1 - 21, x1, y1), 6, 6)
        canvas.drawRRect(box, skia.Paint(AntiAlias=True, Color4f=color4((255, 255, 255), 0.88 * u)))
        self.credit.draw(canvas, x1 - w - 7, y1 - 6.5, skia.Paint(AntiAlias=True, Color4f=color4((120, 120, 120), u)))

    def draw(self, canvas, t):
        canvas.clear(WHITE)
        canvas.save()
        canvas.scale(SS, SS)
        z = self.camera(t)
        intro = t < REVEAL_T
        if intro:
            canvas.saveLayer(None, None)
        self.draw_map(canvas, z)
        if intro:
            self.draw_reveal_mask(canvas, t)
            canvas.restore()
        for d in self.dots:
            self.draw_dot(canvas, d, t, z)
        self.draw_title(canvas, t)
        self.draw_credit(canvas, t)
        canvas.restore()

def load_scene(debug_labels=False):
    with open('data/prepared.pkl', 'rb') as f:
        data = pickle.load(f)
    return Scene(data, 'fonts/InterDisplay-700-cyr.ttf', debug_labels)

def render_frame(scene, t, big, small):
    with big as c:
        scene.draw(c, t)
    img = big.makeImageSnapshot()
    with small as c:
        c.drawImageRect(img, skia.Rect(0, 0, SIZE, SIZE), skia.SamplingOptions(skia.CubicResampler.Mitchell()))
    return small.makeImageSnapshot()


# ---------------------------------------------------------------- parallel video
WORKERS = 3
_W = {}

def _worker_init():
    _W['scene'] = load_scene()
    _W['big'], _W['small'] = skia.Surface(SIZE * SS, SIZE * SS), skia.Surface(SIZE, SIZE)

def _worker_frame(i):
    img = render_frame(_W['scene'], i / FPS, _W['big'], _W['small'])
    return img.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType).tobytes()

if __name__ == '__main__':
    mode = sys.argv[1]
    t0 = time.time()
    if mode != 'video':
        scene = load_scene(debug_labels=('--labels' in sys.argv))
        print('scene built', round(time.time() - t0, 2), 's', flush=True)
        big, small = skia.Surface(SIZE * SS, SIZE * SS), skia.Surface(SIZE, SIZE)
    if mode == 'still':
        t = float(sys.argv[2]); out = sys.argv[3]
        t1 = time.time()
        img = render_frame(scene, t, big, small)
        print('frame', round(time.time() - t1, 3), 's')
        img.save(out, skia.kPNG)
    elif mode == 'sheet':   # contact sheet of several moments
        times = [float(v) for v in sys.argv[2].split(',')]; out = sys.argv[3]
        cols = 3; rows = (len(times) + cols - 1) // cols; cell = 540
        sheet = skia.Surface(cols * cell, rows * cell)
        with sheet as sc:
            sc.clear(skia.Color(128, 128, 128))
            for i, t in enumerate(times):
                img = render_frame(scene, t, big, small)
                sc.drawImageRect(img, skia.Rect.MakeXYWH((i % cols) * cell + 2, (i // cols) * cell + 2, cell - 4, cell - 4),
                                 skia.SamplingOptions(skia.CubicResampler.Mitchell()))
                sc.drawString(f't={t:.2f}s', (i % cols) * cell + 10, (i // cols) * cell + 20,
                              skia.Font(skia.Typeface('DejaVu Sans'), 14), skia.Paint(Color=skia.Color(0, 90, 255)))
        sheet.makeImageSnapshot().save(out, skia.kPNG)
    elif mode == 'video':
        out = sys.argv[2]
        crf = next((a.split('=')[1] for a in sys.argv if a.startswith('--crf=')), '19')
        rng_arg = next((a.split('=')[1] for a in sys.argv if a.startswith('--range=')), None)
        f0, f1 = 0, int(round(DURATION * FPS))
        if rng_arg:   # render only part of the timeline, in seconds: --range=4:6
            s0, s1 = (float(v) for v in rng_arg.split(':'))
            f0, f1 = int(round(s0 * FPS)), int(round(s1 * FPS))
        n = f1 - f0
        ff = subprocess.Popen([
            FFMPEG, '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{SIZE}x{SIZE}', '-r', str(FPS), '-i', '-',
            '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', crf, '-profile:v', 'high', '-level', '4.2',
            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
            '-movflags', '+faststart', out], stdin=subprocess.PIPE)
        with mp.Pool(WORKERS, initializer=_worker_init) as pool:
            for i, buf in enumerate(pool.imap(_worker_frame, range(f0, f1), chunksize=2)):
                ff.stdin.write(buf)
                if i % 120 == 0:
                    print(f'frame {i}/{n}  {time.time() - t0:.0f}s', flush=True)
        ff.stdin.close(); ff.wait()
        print('done', out, round(time.time() - t0, 1), 's')
