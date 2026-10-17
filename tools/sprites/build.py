"""Bake the pixel-art sheets (survivor x 5 tools, 11 creatures, and an atlas of trees, rocks, bushes, ore and buildings) and embed them into ../../index.html.

    python3 tools/sprites/build.py          # needs numpy + pillow

Each sheet is 8 directions (rows: S, SW, W, NW, N, NE, E, SE) x frames (cols per animation are listed in each sheet's meta).
The survivor's jacket and the mountain king's fur are painted in a marker colour; the game swaps it for the real colours at load time.
"""
import base64, io, json, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R
import animals  # noqa: F401  (registers the animals)
import objects

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, '..', '..', 'index.html')

sheets = {}
LOOKS = [('survivor', ['hand', 'axe', 'spear', 'pickaxe', 'gun'])] + [(k, [None]) for k in ('zombie', 'brute', 'zombielord', 'wolf', 'wolfking', 'boar', 'bear', 'mountainking', 'rabbit', 'bat', 'golem')]
for look, tools in LOOKS:
    for tool in tools:
        key = tool if look == 'survivor' else look
        img, meta = R.build_sheet(look, None if tool == 'hand' else tool)
        out = img.convert('RGBA').quantize(colors=96, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)     # the art uses a few dozen colours
        buf = io.BytesIO(); out.save(buf, 'PNG', optimize=True)
        sheets[key] = dict(meta, png=base64.b64encode(buf.getvalue()).decode())
        print(f"{key:9s} {img.size[0]}x{img.size[1]}  frame {meta['w']}x{meta['h']}  {len(buf.getvalue()) // 1024} KB")

atlas, place = objects.build_atlas(objects.objects_list() + objects.building_list(), width=720)      # trees, rocks, bushes, ore, buildings: one image, a rectangle and an anchor for each
out = atlas.convert('RGBA').quantize(colors=96, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
buf = io.BytesIO(); out.save(buf, 'PNG', optimize=True)
sheets['objects'] = dict(items={k: [int(v) for v in vals] for k, vals in place.items()}, png=base64.b64encode(buf.getvalue()).decode())
print(f"objects   {atlas.size[0]}x{atlas.size[1]}  {len(place)} pieces  {len(buf.getvalue()) // 1024} KB")

blob = json.dumps(sheets, separators=(',', ':'))
s = open(INDEX, encoding='utf-8').read()
a = s.index('<script id="sprites">') + len('<script id="sprites">'); b = s.index('</script>', a)
open(INDEX, 'w', encoding='utf-8').write(s[:a] + 'window.SPRITES = ' + blob + ';' + s[b:])
print('embedded', len(blob) // 1024, 'KB into index.html')
