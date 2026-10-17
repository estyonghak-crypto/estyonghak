"""Bake the pixel-art sheets (survivor x 5 tools, zombie, brute) and embed them into ../../index.html.

    python3 tools/sprites/build.py          # needs numpy + pillow

Each sheet is 8 directions (rows: S, SW, W, NW, N, NE, E, SE) x frames (cols: idle 4, walk 8, attack 4, hurt 1, die 5 for the monsters).
The survivor's jacket is painted in a marker colour; the game swaps it for the armour colours at load time.
"""
import base64, io, json, os, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, '..', '..', 'index.html')

sheets = {}
for look, tools in (('survivor', ['hand', 'axe', 'spear', 'pickaxe', 'gun']), ('zombie', [None]), ('brute', [None])):
    for tool in tools:
        key = tool if look == 'survivor' else look
        img, meta = R.build_sheet(look, None if tool == 'hand' else tool)
        out = img.convert('RGBA').quantize(colors=96, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)     # the art uses a few dozen colours
        buf = io.BytesIO(); out.save(buf, 'PNG', optimize=True)
        sheets[key] = dict(meta, png=base64.b64encode(buf.getvalue()).decode())
        print(f"{key:9s} {img.size[0]}x{img.size[1]}  frame {meta['w']}x{meta['h']}  {len(buf.getvalue()) // 1024} KB")

blob = json.dumps(sheets, separators=(',', ':'))
s = open(INDEX, encoding='utf-8').read()
a = s.index('<script id="sprites">') + len('<script id="sprites">'); b = s.index('</script>', a)
open(INDEX, 'w', encoding='utf-8').write(s[:a] + 'window.SPRITES = ' + blob + ';' + s[b:])
print('embedded', len(blob) // 1024, 'KB into index.html')
