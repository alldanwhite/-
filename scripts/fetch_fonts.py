"""Download Inter (Google Fonts, SIL OFL 1.1) and cut the static instances we need.

* fonts/InterDisplay-700-cyr.ttf - cyrillic subset, opsz=32 ("Display", made
  for headlines), wght=700 - the title.
* fonts/Inter-400-latin.ttf      - latin subset, opsz=14, wght=400 - map credit.
"""
import os, re, urllib.request
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
CSS = 'https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,100..900'

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def subset_url(css, name):
    return re.search(r'/\* %s \*/\s*@font-face \{.*?url\((.*?)\)' % name, css, re.S).group(1)

os.makedirs('fonts', exist_ok=True)
css = get(CSS).decode()
for subset, axes, out in (('cyrillic', {'opsz': 32, 'wght': 700}, 'fonts/InterDisplay-700-cyr.ttf'),
                          ('latin', {'opsz': 14, 'wght': 400}, 'fonts/Inter-400-latin.ttf')):
    src = f'fonts/InterVar-{subset}.woff2'
    open(src, 'wb').write(get(subset_url(css, subset)))
    font = TTFont(src)
    font.flavor = None
    instancer.instantiateVariableFont(font, axes).save(out)
    print(out)
