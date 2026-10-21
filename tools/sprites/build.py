"""Bake the pixel-art sheets (survivor x 12 tools, 27 creatures, and an atlas of trees, rocks, bushes, ore and buildings) and embed them into ../../index.html.

    python3 tools/sprites/build.py          # needs numpy + pillow
    python3 tools/sprites/build.py --icons  # just the interface icons (tools/sprites/icons.py), the other sheets stay

Each sheet is 8 directions (rows: S, SW, W, NW, N, NE, E, SE) x frames (cols per animation are listed in each sheet's meta); creature sheets keep only the first five rows and the game mirrors them.
The survivor's jacket and the mountain king's fur are painted in a marker colour; the game swaps it for the real colours at load time.
"""
import base64, io, json, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R
import animals  # noqa: F401  (registers the animals)
import bosses   # noqa: F401  (the zone bosses, the new night bosses, the creatures of the new islands)
import objects

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, '..', '..', 'index.html')

import icons

def icon_sheet():
    atlas, idx, cols, rows = icons.build()
    buf = io.BytesIO(); atlas.save(buf, 'WEBP', lossless=True, quality=100, method=6)
    print(f"icons     {atlas.size[0]}x{atlas.size[1]}  {len(idx)} emoji  {len(buf.getvalue()) // 1024} KB")
    return dict(map=idx, cols=cols, rows=rows, n=icons.N, img=base64.b64encode(buf.getvalue()).decode(), mime='image/webp')

def embed(sheets):
    blob = json.dumps(sheets, separators=(',', ':'), default=int)      # (numpy integers in the metadata)
    s = open(INDEX, encoding='utf-8').read()
    a = s.index('<script id="sprites">') + len('<script id="sprites">'); b = s.index('</script>', a)
    open(INDEX, 'w', encoding='utf-8').write(s[:a] + 'window.SPRITES = ' + blob + ';' + s[b:])
    print('embedded', len(blob) // 1024, 'KB into index.html')

if '--icons' in sys.argv:                  # only the interface icons: keep every baked sheet as it is
    s = open(INDEX, encoding='utf-8').read()
    a = s.index('window.SPRITES = ', s.index('<script id="sprites">')) + len('window.SPRITES = '); b = s.index(';</script>', a)
    sheets = json.loads(s[a:b]); sheets['icons'] = icon_sheet(); embed(sheets); sys.exit()

sheets = {}
LOOKS = [('survivor', ['hand', 'axe', 'spear', 'pickaxe', 'gun', 'sword', 'hammer', 'wrench', 'sickle', 'bombpick', 'bow', 'shotgun'])] + [(k, [None]) for k in ('zombie', 'brute', 'zombielord', 'wolf', 'wolfking', 'boar', 'bear', 'mountainking', 'rabbit', 'bat', 'golem', 'batqueen', 'plaguegiant', 'frostgiant',
                                                        'boarking', 'mole', 'spiderqueen', 'spider', 'yeti', 'crocking', 'lavagiant', 'magma', 'firebat', 'snake', 'ninetail', 'ghostcaptain', 'turtle')]
for look, tools in LOOKS:
    for tool in tools:
        key = tool if look == 'survivor' else look
        img, meta = R.build_sheet(look, None if tool == 'hand' else tool, half=look != 'survivor')       # creatures keep five directions, the game mirrors the other three
        buf = io.BytesIO(); img.save(buf, 'WEBP', lossless=True, quality=100, method=6)               # exact colours (a few dozen per sheet), no blur
        sheets[key] = dict(meta, img=base64.b64encode(buf.getvalue()).decode(), mime='image/webp')
        print(f"{key:9s} {img.size[0]}x{img.size[1]}  frame {meta['w']}x{meta['h']}  {len(buf.getvalue()) // 1024} KB")

atlas, place = objects.build_atlas(objects.objects_list() + objects.building_list() + bosses.extra_objects(), width=720)      # trees, rocks, bushes, ore, buildings: one image, a rectangle and an anchor for each
buf = io.BytesIO(); atlas.convert('RGBA').save(buf, 'WEBP', lossless=True, quality=100, method=6)
sheets['objects'] = dict(items={k: [int(v) for v in vals] for k, vals in place.items()}, img=base64.b64encode(buf.getvalue()).decode(), mime='image/webp')
print(f"objects   {atlas.size[0]}x{atlas.size[1]}  {len(place)} pieces  {len(buf.getvalue()) // 1024} KB")

sheets['icons'] = icon_sheet()
embed(sheets)
