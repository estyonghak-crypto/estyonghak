"""Pixel icons for the interface: every emoji the game prints gets a 20 x 20 drawing in the same palette as the world.

Each icon is a few flat shapes, painted from the back to the front. Every shape is shaded by itself (a lighter rim up and to the left,
a darker one down and to the right, the way the sun lights the world) and the whole icon gets a dark outline, like the sprites.

    python3 tools/sprites/icons.py preview.png     # a contact sheet at 4x, for looking at them
"""
import math, sys
import numpy as np
from PIL import Image, ImageDraw

N = 20                                     # a cell; the drawing keeps to the inner 18 x 18, the outline takes the last pixel
OUTLINE = (24, 17, 13, 255)

def hexc(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
def ramp(*c): return [hexc(x) for x in c]
PAL = {
    'wood':   ramp('#4a2e16', '#7a4e26', '#a8743c', '#d29e5c'), 'bark': ramp('#3a2414', '#5c3a1e', '#7e5430', '#a07446'),
    'ring':   ramp('#8a6438', '#c09458', '#ddb878', '#f0d49a'),
    'stone':  ramp('#3e4248', '#61666d', '#8a9097', '#b8bec4'), 'dark': ramp('#16181b', '#25282c', '#383c42', '#50555c'),
    'red':    ramp('#5a0f12', '#a01e24', '#d83a36', '#ff7a66'), 'meat': ramp('#6a1a1e', '#b0343a', '#dc5a5a', '#f59a90'),
    'fat':    ramp('#b8a890', '#e0d2b8', '#f4ead6', '#ffffff'), 'bone': ramp('#9a8e78', '#d6ccb4', '#efe8d6', '#ffffff'),
    'roast':  ramp('#4a220c', '#84401a', '#b8662a', '#e0924a'),
    'brass':  ramp('#6a4a12', '#b08428', '#e0b444', '#fff0a0'), 'steel': ramp('#3a4450', '#66727e', '#98a4b0', '#d8e2ea'),
    'leather': ramp('#4a2c16', '#80522c', '#a87444', '#cc9a64'),
    'glass':  ramp('#5a7a86', '#9cc0cc', '#d4ecf2', '#ffffff'),
    'cyan':   ramp('#12546a', '#2a92b0', '#5cc8e0', '#b8f2ff'), 'blue': ramp('#14286a', '#2a4ab0', '#4c78e0', '#a4c4ff'),
    'orange': ramp('#6a2a08', '#c45a14', '#f08a2c', '#ffc070'), 'yellow': ramp('#7a5a08', '#d0a018', '#f4d04a', '#fff4a8'),
    'gold':   ramp('#6a4a08', '#b88a18', '#ecc23c', '#fff0a0'), 'copper': ramp('#5a2a10', '#a05024', '#d4783c', '#f8b07c'),
    'green':  ramp('#1a4012', '#2e7024', '#52a03c', '#a0d870'), 'leaf': ramp('#183a1c', '#2c6430', '#4a9446', '#8cc870'),
    'purple': ramp('#2e1450', '#5a2c8e', '#8a52c8', '#c8a0f4'), 'pink': ramp('#7a1e4a', '#d0508a', '#f08cb8', '#ffd0e4'),
    'ice':    ramp('#3a6a8a', '#7ab8dc', '#b4e2f8', '#f0fcff'), 'white': ramp('#8a8a90', '#c8c8cc', '#e8e8ea', '#ffffff'),
    'cloth':  ramp('#8a7a5a', '#cfc0a0', '#ebe0c6', '#fff8e6'), 'tea': ramp('#2c4a14', '#4c7a22', '#78a840', '#b4d878'),
    'fish':   ramp('#2a4458', '#4c7490', '#7aa4c0', '#c0dcec'), 'cream': ramp('#9a8a6a', '#d8c8a0', '#f0e4c4', '#fffaea'),
    'rice':   ramp('#a8a49a', '#e4e0d6', '#f6f4ee', '#ffffff'), 'soupr': ramp('#7a2a10', '#b8501c', '#e07a36', '#ffb070'),
    'gray':   ramp('#2e3034', '#55595e', '#80858b', '#b0b5ba'), 'zomb': ramp('#2a3a1c', '#4a6a2c', '#729a44', '#a8cc78'),
    'wolf':   ramp('#2a2c30', '#4c5058', '#767b84', '#a8adb5'), 'boar': ramp('#3a2414', '#64402a', '#8c6040', '#b48a64'),
    'croc':   ramp('#1c3412', '#30561e', '#4c7c2e', '#80a858'), 'fur': ramp('#7a7a80', '#b8b8c0', '#dcdce2', '#ffffff'),
    'skin':   ramp('#8a5434', '#c8845a', '#e8aa7c', '#ffd0a8'), 'night': ramp('#1a2048', '#2c3878', '#4c5cae', '#8898e0'),
    'fire':   ramp('#a02008', '#e85a10', '#ffa020', '#fff070'), 'smoke': ramp('#4a4e56', '#767c86', '#a4aab4', '#d4d8de'),
}

class Icon:
    def __init__(self): self.layers = []
    def _layer(self, draw):
        im = Image.new('L', (N, N), 0); draw(ImageDraw.Draw(im)); return np.array(im) > 0
    def shape(self, pal, draw, shade=True, hl=None):
        """a solid part in one material: shaded at its own edges; hl: pixels for the shine"""
        self.layers.append(('s', self._layer(draw), PAL[pal] if isinstance(pal, str) else pal, shade, hl or []))
        return self
    def ink(self, col, draw):
        """a flat detail (eyes, cracks, letters): one colour, no shading"""
        self.layers.append(('i', self._layer(draw), hexc(col) if isinstance(col, str) else col, False, []))
        return self
    def cut(self, draw):
        """a hole: clears what is under it (the outline then runs round its edge)"""
        self.layers.append(('c', self._layer(draw), None, False, []))
        return self
    def px(self, col, pts):
        m = np.zeros((N, N), bool)
        for x, y in pts: m[y, x] = True
        self.layers.append(('i', m, hexc(col) if isinstance(col, str) else col, False, []))
        return self
    def image(self):
        img = np.zeros((N, N, 4), np.uint8); solid = np.zeros((N, N), bool)
        for kind, m, pal, shade, hl in self.layers:
            if kind == 'c':
                img[m] = 0; solid &= ~m; continue
            if kind == 'i':
                img[m] = (*pal, 255); solid |= m; continue
            base = np.zeros((N, N, 3), np.uint8); base[:] = pal[1]
            if shade:
                p = np.pad(m, 1)
                up, left, down, right = ~p[:-2, 1:-1], ~p[1:-1, :-2], ~p[2:, 1:-1], ~p[1:-1, 2:]
                base[m & (up | left)] = pal[2]
                base[m & (down | right) & ~(up | left)] = pal[0]
            img[m, :3] = base[m]; img[m, 3] = 255
            for x, y in hl: img[y, x] = (*pal[3], 255)
            solid |= m
        p = np.pad(solid, 1)
        edge = ~solid & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
        img[edge] = OUTLINE
        return Image.fromarray(img, 'RGBA')

def poly(*pts): return lambda d: d.polygon(list(pts), fill=255)
def rect(x0, y0, x1, y1): return lambda d: d.rectangle([x0, y0, x1, y1], fill=255)
def ell(x0, y0, x1, y1): return lambda d: d.ellipse([x0, y0, x1, y1], fill=255)
def line(pts, w=1): return lambda d: d.line(pts, fill=255, width=w)
def many(*fs): return lambda d: [f(d) for f in fs]
def star(cx, cy, r0, r1, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r1 if i % 2 == 0 else r0; a = rot + i * math.pi / n; pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return poly(*pts)

ICONS = {}
def icon(*keys):
    def reg(f):
        for k in keys: ICONS[k] = f
        return f
    return reg

# ------------------------------------------------------------------ materials and things
@icon('🪵')
def _wood():
    return (Icon().shape('bark', poly((2, 12), (12, 2), (17, 7), (7, 17)))
            .shape('ring', ell(10, 2, 17, 9), hl=[(12, 3)]).ink('#8a6438', ell(12, 4, 15, 7)).px('#5c3a1e', [(13, 5), (14, 5)])
            .ink('#3a2414', line([(4, 13), (11, 6)])))
@icon('🪨')
def _stone(): return Icon().shape('stone', poly((2, 14), (4, 7), (9, 3), (15, 5), (18, 11), (16, 17), (6, 17)), hl=[(8, 5), (9, 5), (7, 6)]).ink('#3e4248', line([(10, 9), (13, 13)]))
@icon('🍒')
def _berry():
    return (Icon().shape('leaf', poly((10, 2), (16, 1), (14, 5))).ink('#3a5a1c', line([(10, 3), (6, 10)])).ink('#3a5a1c', line([(10, 3), (13, 11)]))
            .shape('red', ell(1, 9, 9, 17), hl=[(3, 11), (4, 11)]).shape('red', ell(9, 10, 17, 18), hl=[(11, 12), (12, 12)]))
@icon('🥩')
def _meat():
    return (Icon().shape('fat', poly((2, 9), (6, 3), (14, 2), (18, 7), (17, 14), (10, 18), (4, 16)))
            .shape('meat', poly((4, 9), (7, 5), (13, 4), (16, 8), (15, 13), (10, 16), (5, 14)), hl=[(8, 6), (9, 6)])
            .ink('#f4ead6', line([(7, 9), (12, 8)])).ink('#f4ead6', line([(8, 12), (13, 11)])))
@icon('🍖')
def _cooked():
    return (Icon().shape('bone', many(rect(11, 11, 15, 14), ell(13, 13, 18, 18), ell(14, 10, 18, 14)))
            .shape('roast', ell(1, 1, 13, 13), hl=[(4, 3), (5, 3), (3, 4)]).ink('#5a2a0c', line([(5, 8), (8, 6)])))
@icon('🔸')
def _ammo():
    def bullet(x):
        return Icon.shape
    ic = Icon()
    for x in (3, 9):
        ic.shape('brass', rect(x, 8, x + 4, 17)).shape('copper', poly((x, 8), (x + 2, 3), (x + 4, 8)), hl=[(x + 2, 4)]).ink('#6a4a12', line([(x, 15), (x + 4, 15)]))
    return ic
@icon('🪶')
def _arrow():
    return (Icon().ink('#6a4a24', line([(3, 17), (16, 4)], 2)).shape('steel', poly((13, 3), (18, 1), (16, 7)))
            .shape('white', poly((2, 13), (5, 12), (7, 15), (6, 18), (3, 18))).shape('red', poly((5, 15), (7, 14), (8, 16), (6, 17))))
@icon('🔩')
def _scrap():
    return (Icon().shape('steel', rect(8, 7, 11, 18)).ink('#3a4450', many(line([(8, 10), (11, 9)]), line([(8, 13), (11, 12)]), line([(8, 16), (11, 15)])))
            .shape('steel', poly((5, 4), (8, 1), (12, 1), (15, 4), (12, 7), (8, 7)), hl=[(8, 2), (9, 2)]))
@icon('🟫')
def _hide(): return Icon().shape('leather', poly((4, 3), (8, 4), (12, 2), (16, 4), (15, 9), (17, 14), (13, 17), (10, 15), (6, 17), (3, 13), (5, 8)), hl=[(6, 5), (7, 5)]).ink('#5a3418', many(line([(7, 9), (12, 9)]), line([(8, 12), (11, 12)])))
@icon('🧪')
def _heal():
    return (Icon().shape('leather', rect(8, 1, 11, 4)).shape('glass', rect(7, 4, 12, 7))
            .shape('red', ell(3, 6, 16, 18), hl=[(6, 9), (6, 10), (7, 8)]).ink('#ffe0d8', many(rect(9, 10, 10, 15), rect(7, 12, 12, 13))))
@icon('💨')
def _swift():
    ic = Icon()
    for (y, x0, x1) in ((4, 2, 14), (9, 5, 17), (14, 1, 12)):
        ic.shape('cyan', many(line([(x0, y), (x1, y)], 2), ell(x1 - 2, y - 3, x1 + 2, y + 1)))
    return ic.ink('#b8f2ff', many(line([(3, 4), (8, 4)]), line([(6, 9), (11, 9)])))
@icon('💥')
def _power():
    return (Icon().shape('red', star(10, 10, 4.5, 9.5, 8)).shape('orange', star(10, 10, 3, 6.5, 8)).shape('yellow', ell(7, 7, 13, 13), hl=[(9, 8)]))
@icon('💠', '💎')
def _core():
    return (Icon().shape('cyan', poly((10, 1), (18, 8), (10, 18), (2, 8))).shape('blue', poly((10, 8), (18, 8), (10, 18)))
            .shape('ice', poly((6, 3), (10, 1), (14, 3), (10, 8)), hl=[(9, 3), (10, 3)]).ink('#12546a', line([(2, 8), (18, 8)])))
@icon('🔺')
def _tstone():
    return (Icon().shape('red', poly((10, 1), (18, 17), (2, 17)), hl=[(9, 4), (8, 6)]).shape('pink', poly((10, 4), (14, 14), (10, 12)))
            .ink('#5a0f12', line([(10, 12), (2, 17)])))
def _ore(vein):
    return (Icon().shape('stone', poly((2, 13), (4, 6), (9, 3), (15, 4), (18, 10), (16, 17), (5, 17)))
            .shape(vein, ell(5, 7, 9, 11), hl=[(6, 8)]).shape(vein, ell(11, 6, 14, 9), hl=[(12, 7)]).shape(vein, ell(10, 11, 15, 15), hl=[(11, 12)]))
@icon('🟠')
def _copper(): return _ore('copper')
@icon('⚪')
def _iron(): return _ore('white')
@icon('🟡')
def _gold(): return _ore('gold')
def _bar(m):
    return (Icon().shape(m, poly((1, 12), (5, 6), (19, 6), (15, 12))).shape(m, poly((1, 12), (15, 12), (15, 16), (1, 16)))
            .shape(m, poly((15, 12), (19, 6), (19, 10), (15, 16))).px(PAL[m][3], [(6, 7), (7, 7), (8, 7)]))
@icon('🟧')
def _cubar(): return _bar('copper')
@icon('🟦')
def _febar(): return _bar('steel')
@icon('🟨')
def _aubar(): return _bar('gold')
@icon('🩹')
def _bandage():
    return (Icon().shape('cloth', poly((2, 12), (12, 2), (17, 7), (7, 17)), hl=[(11, 4)]).shape('cream', poly((6, 8), (8, 6), (12, 10), (10, 12)))
            .px('#8a7a5a', [(8, 9), (9, 10), (10, 8), (9, 8), (8, 11)]))
@icon('🌿', '🌱')
def _herb():
    ic = Icon().ink('#2a4a18', line([(10, 18), (10, 5)], 2))
    for (x0, y0, x1, y1) in ((2, 7, 10, 12), (10, 4, 18, 9), (3, 12, 10, 16), (10, 10, 17, 15), (7, 1, 13, 6)):
        ic.shape('leaf', ell(x0, y0, x1, y1), hl=[(x0 + 2, y0 + 1)])
    return ic
@icon('🍵')
def _tea():
    return (Icon().ink('#a4aab4', many(line([(7, 1), (6, 4)]), line([(11, 1), (12, 4)])))
            .shape('cream', poly((2, 7), (16, 7), (14, 16), (4, 16))).shape('cream', many(ell(14, 8, 19, 13))).ink('#24180e', ell(16, 9, 17, 11))
            .shape('tea', rect(3, 7, 15, 8)).ink('#9a8a6a', line([(4, 12), (14, 12)])))
@icon('🧃')
def _tonic():
    return (Icon().ink('#e8e8ea', line([(12, 1), (12, 5)], 2)).shape('yellow', rect(5, 5, 15, 18), hl=[(6, 6), (6, 7)])
            .shape('orange', rect(5, 9, 15, 14)).ink('#fff4a8', many(line([(8, 11), (12, 11)]))))
@icon('🐟')
def _fish():
    return (Icon().shape('fish', poly((1, 5), (5, 9), (1, 14))).shape('fish', ell(3, 4, 18, 15), hl=[(9, 5), (10, 5)])
            .shape('cream', ell(6, 10, 15, 14)).ink('#141414', rect(14, 7, 15, 8)).ink('#2a4458', line([(11, 5), (11, 13)])))
@icon('🍢')
def _grilled():
    return (Icon().ink('#8a6438', line([(2, 18), (17, 2)], 2)).shape('roast', poly((3, 9), (8, 5), (14, 9), (12, 13), (6, 14)), hl=[(7, 6)])
            .shape('roast', poly((9, 13), (14, 9), (17, 12), (12, 16)))
            .ink('#4a220c', many(line([(6, 9), (9, 11)]), line([(9, 7), (12, 9)]))))
@icon('🍲')
def _stew():
    return (Icon().ink('#a4aab4', many(line([(6, 1), (5, 4)]), line([(10, 0), (11, 4)]), line([(14, 1), (13, 4)])))
            .shape('dark', many(rect(1, 7, 3, 9), rect(16, 7, 18, 9))).shape('gray', poly((2, 8), (17, 8), (15, 17), (4, 17)))
            .shape('soupr', rect(3, 6, 16, 8), hl=[(5, 6)]).px('#ffd070', [(7, 7), (12, 7)]))
@icon('🥣')
def _soup():
    return (Icon().ink('#a4aab4', many(line([(8, 1), (7, 5)]), line([(12, 1), (13, 5)])))
            .ink('#c8c8cc', line([(13, 9), (18, 3)], 2)).shape('cream', poly((1, 9), (18, 9), (15, 15), (11, 17), (8, 17), (4, 15)))
            .shape('soupr', rect(2, 8, 17, 9), hl=[(4, 8)]))
@icon('🍱')
def _feast():
    return (Icon().shape('red', rect(1, 4, 18, 17)).shape('dark', rect(3, 6, 16, 15)).shape('rice', rect(3, 6, 9, 15), hl=[(4, 7)])
            .shape('roast', rect(11, 6, 16, 10)).shape('green', rect(11, 12, 16, 15)).px('#5a0f12', [(10, 6), (10, 15)]))

# ------------------------------------------------------------------ conditions and effects
@icon('🩸')
def _bleed(): return Icon().shape('red', many(poly((10, 1), (16, 11), (4, 11)), ell(4, 6, 16, 18)), hl=[(7, 10), (7, 11), (8, 9)])
@icon('🦠')
def _infect():
    ic = Icon()
    for i in range(8):
        a = i * math.pi / 4; x, y = 10 + 7.5 * math.cos(a), 10 + 7.5 * math.sin(a)
        ic.shape('green', ell(x - 1.6, y - 1.6, x + 1.6, y + 1.6))
    return ic.shape('green', ell(4, 4, 16, 16), hl=[(7, 6), (6, 7)]).ink('#1a4012', many(ell(7, 8, 9, 10), ell(11, 11, 13, 13), ell(12, 6, 13, 7)))
@icon('☠', '💀')
def _skull():
    return (Icon().shape('bone', many(line([(2, 13), (17, 18)], 3), line([(2, 18), (17, 13)], 3)))
            .shape('bone', many(ell(3, 1, 16, 13), rect(6, 10, 13, 15)), hl=[(6, 3), (7, 3), (5, 4)])
            .ink('#24180e', many(ell(5, 6, 8, 9), ell(11, 6, 14, 9), rect(9, 10, 10, 11), line([(7, 14), (12, 14)]))))
@icon('🧊')
def _ice():
    return (Icon().shape('ice', poly((3, 6), (10, 2), (17, 6), (17, 14), (10, 18), (3, 14)), hl=[(6, 5), (7, 5), (5, 6)])
            .shape('cyan', poly((10, 10), (17, 6), (17, 14), (10, 18))).ink('#f0fcff', line([(3, 6), (10, 10), (17, 6)])).ink('#f0fcff', line([(10, 10), (10, 18)])))
@icon('😴', '💤')
def _zzz():
    ic = Icon()
    for (x, y, s) in ((2, 9, 8), (10, 4, 6), (14, 0, 4)):
        ic.shape('ice', many(line([(x, y), (x + s, y)], 2), line([(x + s, y), (x, y + s)], 2), line([(x, y + s), (x + s, y + s)], 2)))
    return ic
@icon('😵')
def _dizzy():
    return (Icon().shape('yellow', ell(2, 2, 17, 17), hl=[(6, 4), (5, 5)])
            .ink('#4a2a08', many(line([(4, 5), (8, 9)], 2), line([(8, 5), (4, 9)], 2), line([(11, 5), (15, 9)], 2), line([(15, 5), (11, 9)], 2), ell(7, 12, 12, 16))))
@icon('🔥')
def _fire():
    return (Icon().shape('fire', poly((10, 1), (13, 6), (16, 5), (17, 11), (14, 17), (6, 17), (3, 12), (4, 7), (7, 9), (8, 4)))
            .shape('orange', poly((10, 7), (13, 11), (12, 16), (8, 16), (7, 12))).shape('yellow', poly((10, 11), (11.5, 14), (10, 16), (8.5, 14))))
@icon('⚡')
def _bolt(): return Icon().shape('yellow', poly((12, 1), (4, 11), (9, 11), (7, 18), (16, 7), (11, 7), (14, 1)), hl=[(11, 3), (10, 4)])
@icon('✨')
def _sparkle():
    return (Icon().shape('yellow', star(8, 9, 1.6, 7.5, 4, 0)).shape('yellow', star(15, 4, .9, 3.6, 4, 0)).shape('white', star(15, 15, .8, 3, 4, 0)))
@icon('⚠')
def _warn():
    return (Icon().shape('yellow', poly((10, 1), (19, 17), (1, 17)), hl=[(9, 4), (8, 6)]).ink('#2a1a06', many(rect(9, 6, 10, 12), rect(9, 14, 10, 15))))
@icon('🏆')
def _trophy():
    return (Icon().shape('gold', many(ell(1, 3, 7, 10), ell(12, 3, 18, 10))).cut(many(ell(3, 5, 5, 8), ell(14, 5, 16, 8)))
            .shape('gold', poly((4, 2), (15, 2), (14, 9), (11, 12), (8, 12), (5, 9)), hl=[(6, 3), (6, 4)]).shape('gold', rect(9, 12, 10, 14))
            .shape('bark', rect(5, 15, 14, 18)))
@icon('👑')
def _crown():
    return (Icon().shape('gold', poly((2, 15), (2, 5), (6, 10), (10, 3), (14, 10), (18, 5), (18, 15)), hl=[(4, 8), (9, 6)]).shape('gold', rect(2, 14, 18, 17))
            .shape('red', ell(9, 13, 12, 16)).shape('cyan', ell(4, 14, 6, 16)).shape('cyan', ell(14, 14, 16, 16)))
@icon('🛡')
def _shield():
    return (Icon().shape('steel', poly((2, 3), (10, 1), (18, 3), (17, 11), (10, 18), (3, 11)), hl=[(5, 4), (4, 5)])
            .shape('blue', poly((5, 5), (10, 4), (15, 5), (14, 10), (10, 15), (6, 10))).shape('gold', rect(9, 5, 11, 14)).shape('gold', rect(6, 8, 14, 10)))
@icon('🔊')
def _speaker():
    return (Icon().shape('white', many(rect(2, 7, 6, 12), poly((6, 7), (11, 3), (11, 16), (6, 12))), hl=[(3, 8)])
            .ink('#e8e8ea', many(line([(13, 7), (14, 9), (13, 12)]), line([(16, 4), (18, 9), (16, 15)]))))
@icon('🔇')
def _muted():
    return (Icon().shape('gray', many(rect(1, 7, 5, 12), poly((5, 7), (10, 3), (10, 16), (5, 12))), hl=[(2, 8)])
            .shape('red', many(line([(12, 6), (18, 13)], 2), line([(18, 6), (12, 13)], 2))))
@icon('🎯')
def _target():
    return (Icon().shape('white', ell(1, 1, 18, 18)).shape('red', ell(3, 3, 16, 16)).shape('white', ell(6, 6, 13, 13)).shape('red', ell(8, 8, 11, 11))
            .ink('#6a4a24', line([(11, 9), (18, 2)], 2)).shape('steel', poly((15, 1), (19, 0), (18, 4))))

# ------------------------------------------------------------------ events
@icon('📦')
def _crate():
    return (Icon().shape('wood', rect(2, 4, 17, 17), hl=[(3, 5), (4, 5)]).ink('#4a2e16', many(rect(2, 4, 17, 5), rect(2, 10, 17, 10), rect(2, 16, 17, 17)))
            .ink('#4a2e16', line([(3, 6), (16, 15)])).px('#d29e5c', [(9, 7), (10, 7)]))
@icon('☄')
def _meteor():
    return (Icon().shape('fire', poly((1, 2), (13, 10), (9, 14))).shape('orange', poly((3, 4), (12, 10), (10, 12)))
            .shape('dark', ell(8, 8, 18, 18), hl=[(11, 10)]).ink('#ff7a20', many(line([(12, 13), (15, 15)]), line([(14, 11), (16, 12)]))))
@icon('🐗')
def _boar():
    return (Icon().shape('boar', many(poly((3, 4), (7, 2), (13, 2), (17, 4), (18, 12), (14, 17), (6, 17), (2, 12)), poly((2, 1), (6, 4), (3, 6)), poly((18, 1), (14, 4), (17, 6))), hl=[(7, 4), (8, 4)])
            .shape('skin', ell(6, 10, 14, 16)).ink('#24180e', many(rect(8, 12, 8, 13), rect(11, 12, 11, 13), rect(6, 7, 7, 8), rect(12, 7, 13, 8)))
            .shape('bone', many(poly((5, 13), (3, 10), (5, 11)), poly((15, 13), (17, 10), (15, 11)))))
@icon('🏕')
def _camp():
    return (Icon().shape('green', poly((10, 2), (18, 15), (2, 15)), hl=[(9, 5), (8, 7)]).shape('dark', poly((10, 7), (13, 15), (7, 15)))
            .ink('#5c3a1e', line([(1, 17), (18, 17)], 2)).px('#ffa020', [(10, 13), (10, 12), (9, 14), (11, 14)]))
@icon('⛈')
def _storm():
    return (Icon().ink('#7ab8dc', many(line([(4, 13), (3, 17)]), line([(8, 14), (7, 18)]), line([(15, 13), (14, 17)])))
            .shape('yellow', poly((11, 9), (8, 14), (11, 14), (9, 19), (14, 12), (11, 12), (13, 9)))
            .shape('smoke', many(ell(1, 5, 9, 12), ell(5, 1, 15, 10), ell(11, 4, 19, 12)), hl=[(7, 3), (8, 3)]))
@icon('🥶')
def _cold():
    return (Icon().shape('white', many(rect(8, 1, 11, 13), ell(5, 11, 14, 19)), hl=[(9, 2)]).shape('blue', many(rect(9, 7, 10, 13), ell(7, 13, 12, 18)))
            .ink('#c8c8cc', many(line([(12, 3), (14, 3)]), line([(12, 6), (14, 6)]), line([(12, 9), (14, 9)]))))
@icon('🐺')
def _wolf():
    return (Icon().shape('wolf', many(poly((2, 1), (7, 5), (3, 8)), poly((18, 1), (13, 5), (17, 8)), poly((3, 6), (10, 3), (17, 6), (16, 12), (10, 18), (4, 12))), hl=[(6, 5), (7, 5)])
            .shape('fur', poly((7, 11), (10, 9), (13, 11), (10, 17))).ink('#141414', many(rect(10, 15, 10, 16), rect(9, 15, 11, 15)))
            .ink('#f4d04a', many(rect(6, 8, 7, 9), rect(12, 8, 13, 9))))
@icon('🧟')
def _zombie():
    return (Icon().shape('zomb', many(ell(3, 2, 17, 17)), hl=[(6, 4), (5, 5)]).shape('dark', poly((3, 7), (4, 2), (10, 1), (16, 2), (17, 7), (13, 5), (8, 6)))
            .ink('#e8e8a0', many(rect(6, 9, 8, 10), rect(12, 9, 14, 10))).ink('#2a1a10', many(rect(7, 9, 7, 10), rect(13, 9, 13, 10)))
            .ink('#2a3a1c', many(line([(7, 14), (13, 14)]), rect(8, 15, 8, 15), rect(11, 15, 11, 15))).px('#8a2020', [(5, 12), (5, 13), (15, 6)]))
@icon('🌫')
def _fog():
    ic = Icon()
    for (y, x0, x1) in ((4, 4, 17), (8, 1, 14), (12, 5, 18), (16, 2, 13)):
        ic.shape('smoke', many(line([(x0, y), (x1, y)], 3)))
    return ic
@icon('🌕')
def _fullmoon():
    return (Icon().shape('cream', ell(1, 1, 18, 18), hl=[(5, 4), (4, 5), (6, 3)]).ink('#b8a880', many(ell(5, 7, 8, 10), ell(11, 4, 13, 6), ell(10, 11, 14, 15))))
@icon('🌙')
def _moon(): return Icon().shape('yellow', poly((9, 1), (5, 4), (3, 9), (5, 15), (10, 18), (16, 17), (18, 14), (13, 14), (9, 11), (8, 6)), hl=[(6, 5), (5, 7)])
@icon('☀')
def _sun():
    ic = Icon()
    for i in range(8):
        a = i * math.pi / 4; ic.shape('orange', poly((10 + 9.4 * math.cos(a), 10 + 9.4 * math.sin(a)), (10 + 5 * math.cos(a + .4), 10 + 5 * math.sin(a + .4)), (10 + 5 * math.cos(a - .4), 10 + 5 * math.sin(a - .4))))
    return ic.shape('yellow', ell(4, 4, 15, 15), hl=[(7, 6), (6, 7)])
@icon('❄')
def _snowflake():
    ic = Icon()
    for i in range(3):
        a = i * math.pi / 3; dx, dy = 8.5 * math.cos(a), 8.5 * math.sin(a)
        ic.shape('ice', line([(10 - dx, 10 - dy), (10 + dx, 10 + dy)], 2))
    for i in range(6):
        a = i * math.pi / 3; x, y = 10 + 6 * math.cos(a), 10 + 6 * math.sin(a)
        for s in (1, -1): ic.shape('ice', line([(x, y), (x + 2.6 * math.cos(a + s * 2.3), y + 2.6 * math.sin(a + s * 2.3))]))
    return ic
@icon('🌲')
def _pine():
    return (Icon().shape('bark', rect(9, 14, 11, 18)).shape('leaf', poly((10, 7), (17, 15), (3, 15))).shape('leaf', poly((10, 4), (15, 11), (5, 11)), hl=[(9, 6)])
            .shape('leaf', poly((10, 1), (14, 7), (6, 7)), hl=[(9, 3)]))
@icon('🌸')
def _plum():
    ic = Icon()
    for i in range(5):
        a = -math.pi / 2 + i * 2 * math.pi / 5; x, y = 10 + 5 * math.cos(a), 10 + 5 * math.sin(a)
        ic.shape('pink', ell(x - 4.2, y - 4.2, x + 4.2, y + 4.2), hl=[(int(x) - 1, int(y) - 2)])
    return ic.shape('yellow', ell(7, 7, 12, 12)).px('#d0508a', [(10, 6), (13, 9), (12, 13), (7, 13), (6, 9)])

# ------------------------------------------------------------------ tools, skills, perks
@icon('🪓')
def _axe():
    return (Icon().shape('wood', line([(4, 18), (14, 3)], 3)).shape('steel', poly((9, 2), (16, 0), (19, 6), (16, 9), (12, 6)), hl=[(15, 2), (16, 3)]))
@icon('⛏')
def _pick():
    return (Icon().shape('wood', line([(3, 18), (13, 6)], 3)).shape('steel', poly((2, 6), (8, 2), (14, 1), (19, 4), (19, 9), (16, 5), (11, 4), (5, 6)), hl=[(10, 2), (11, 2)]))
@icon('⚔')
def _swords():
    ic = Icon()
    for flip in (False, True):
        f = (lambda x: 19 - x) if flip else (lambda x: x)
        ic.shape('steel', line([(f(3), 16), (f(16), 3)], 2)).shape('brass', line([(f(2), 12), (f(7), 17)], 2)).shape('leather', line([(f(1), 18), (f(4), 15)], 2))
    return ic
@icon('🗡')
def _dagger():
    return (Icon().shape('steel', poly((7, 11), (16, 1), (18, 3), (9, 13)), hl=[(14, 3)]).shape('brass', line([(4, 10), (10, 16)], 2)).shape('leather', line([(2, 17), (6, 13)], 3)))
@icon('🔨')
def _hammer(): return Icon().shape('wood', line([(4, 18), (12, 8)], 3)).shape('steel', poly((7, 5), (12, 0), (19, 7), (14, 12)), hl=[(11, 2), (10, 3)])
@icon('🔧')
def _wrench():
    return (Icon().shape('steel', many(line([(4, 16), (14, 6)], 4), ell(10, 1, 19, 10)), hl=[(12, 3), (13, 2)]).cut(poly((14, 0), (19, 0), (19, 5), (16, 6), (13, 3))))
@icon('🛠')
def _tools():
    return (Icon().shape('steel', many(line([(4, 3), (15, 15)], 3), ell(1, 0, 8, 7))).cut(poly((1, 0), (5, 0), (4, 3), (1, 3)))
            .shape('wood', line([(3, 18), (12, 9)], 3)).shape('steel', poly((9, 7), (14, 2), (19, 7), (14, 12))))
@icon('🏹')
def _bow():
    return (Icon().ink('#e8e0c8', line([(4, 3), (4, 17)])).shape('wood', many(line([(4, 1), (10, 4), (13, 10), (10, 16), (4, 19)], 2)))
            .ink('#6a4a24', line([(1, 10), (16, 10)])).shape('steel', poly((15, 8), (19, 10), (15, 12))).shape('red', poly((1, 8), (4, 10), (1, 12))))
@icon('🔫')
def _gun():
    return (Icon().shape('wood', poly((1, 9), (6, 8), (8, 11), (5, 16), (1, 16))).shape('steel', rect(5, 7, 19, 10), hl=[(7, 8), (8, 8)])
            .shape('dark', poly((8, 10), (11, 10), (10, 14), (8, 14))).ink('#3a4450', rect(15, 6, 16, 6)))
@icon('🧍')
def _person():
    return (Icon().shape('skin', ell(7, 1, 13, 7), hl=[(9, 2)]).shape('leather', rect(6, 7, 14, 13)).shape('blue', many(rect(7, 13, 9, 18), rect(11, 13, 13, 18)))
            .shape('leather', many(rect(4, 8, 6, 13), rect(14, 8, 16, 13))))
@icon('👊')
def _fist():
    return (Icon().shape('skin', many(rect(3, 6, 16, 15), ell(3, 12, 13, 19)), hl=[(5, 7), (6, 7)]).shape('skin', many(rect(14, 9, 18, 14)))
            .ink('#8a5434', many(line([(6, 6), (6, 10)]), line([(9, 6), (9, 10)]), line([(12, 6), (12, 10)]))))
@icon('🧱')
def _bricks():
    ic = Icon().shape('red', rect(1, 3, 18, 17), hl=[(2, 4), (3, 4)])
    return ic.ink('#e0d2b8', many(*[line([(1, y), (18, y)]) for y in (7, 12)], *[line([(x, y0), (x, y1)]) for x, y0, y1 in ((6, 3, 7), (13, 3, 7), (9, 7, 12), (16, 7, 12), (4, 12, 17), (11, 12, 17))]))
@icon('🐾')
def _paws():
    ic = Icon().shape('boar', poly((4, 14), (7, 10), (13, 10), (16, 14), (14, 18), (6, 18)), hl=[(7, 11), (8, 11)])
    for (x, y) in ((3, 7), (7, 3), (13, 3), (17, 7)): ic.shape('boar', ell(x - 2, y - 2, x + 2, y + 2), hl=[(x - 1, y - 1)])
    return ic
@icon('💉')
def _syringe():
    return (Icon().shape('glass', poly((5, 10), (12, 3), (16, 7), (9, 14)), hl=[(10, 5)]).shape('red', poly((7, 10), (11, 6), (13, 8), (9, 12)))
            .shape('steel', line([(5, 14), (1, 18)], 1)).shape('steel', many(line([(13, 2), (17, 6)], 2), line([(16, 1), (18, 3)], 2))))
@icon('🏰')
def _castle():
    return (Icon().shape('stone', many(rect(3, 7, 16, 18), rect(2, 3, 5, 9), rect(14, 3, 17, 9), rect(8, 4, 11, 9)), hl=[(4, 8), (3, 4)])
            .ink('#3e4248', many(rect(3, 3, 3, 4), rect(16, 3, 16, 4), rect(9, 4, 9, 5))).shape('wood', many(rect(8, 12, 11, 18), ell(8, 10, 11, 14))))
@icon('⚙')
def _gear():
    ic = Icon()
    for i in range(8):
        a = i * math.pi / 4; ic.shape('steel', ell(10 + 7.5 * math.cos(a) - 2, 10 + 7.5 * math.sin(a) - 2, 10 + 7.5 * math.cos(a) + 2, 10 + 7.5 * math.sin(a) + 2))
    return ic.shape('steel', ell(3, 3, 16, 16), hl=[(6, 5), (5, 6)]).cut(ell(7, 7, 12, 12))
@icon('🧨')
def _dynamite():
    return (Icon().ink('#c8c8cc', line([(12, 4), (15, 1)])).px('#ffa020', [(15, 0), (16, 1), (16, 0)])
            .shape('red', many(rect(3, 5, 7, 18), rect(8, 5, 12, 18)), hl=[(4, 6), (9, 6)]).shape('red', rect(13, 6, 17, 18), hl=[(14, 7)]).shape('bark', rect(3, 10, 17, 12)))
@icon('👟')
def _boot():
    return (Icon().shape('leather', poly((4, 2), (10, 2), (10, 11), (17, 13), (18, 17), (2, 17), (3, 9)), hl=[(5, 3), (5, 4)]).shape('dark', rect(2, 16, 18, 18))
            .shape('white', many(poly((1, 4), (4, 6), (1, 8)), poly((1, 9), (3, 10), (1, 11)))))
@icon('🍳')
def _pan():
    return (Icon().shape('dark', many(ell(1, 3, 15, 17), line([(13, 13), (19, 18)], 3)), hl=[(4, 5)]).shape('white', ell(4, 6, 12, 14)).shape('yellow', ell(7, 8, 10, 11), hl=[(8, 8)]))
@icon('🎣')
def _rod():
    return (Icon().shape('wood', line([(2, 18), (15, 2)], 2)).ink('#e8e8ea', many(line([(15, 2), (18, 4), (18, 12)]))).shape('red', ell(16, 11, 19, 15)).shape('white', rect(16, 14, 19, 15)))
@icon('🦷')
def _tusk(): return Icon().shape('bone', poly((4, 18), (2, 10), (5, 4), (11, 1), (17, 3), (12, 4), (8, 8), (8, 14), (10, 18)), hl=[(6, 5), (5, 7)])
@icon('🕸')
def _web():
    ic = Icon()
    for i in range(8):
        a = i * math.pi / 4; ic.ink('#e8e8ea', line([(10, 10), (10 + 9 * math.cos(a), 10 + 9 * math.sin(a))]))
    for r in (3, 6, 9):
        pts = [(10 + r * math.cos(i * math.pi / 4), 10 + r * math.sin(i * math.pi / 4)) for i in range(9)]
        ic.ink('#c8c8cc', line(pts))
    return ic
@icon('🧥')
def _coat():
    return (Icon().shape('fur', poly((6, 1), (14, 1), (18, 6), (17, 18), (3, 18), (2, 6)), hl=[(7, 2), (6, 3)]).shape('white', many(rect(1, 6, 4, 15), rect(16, 6, 19, 15)))
            .ink('#7a7a80', line([(10, 3), (10, 18)])).shape('fur', poly((6, 1), (10, 6), (14, 1))))
@icon('🐊')
def _croc():
    return (Icon().shape('croc', poly((1, 8), (8, 5), (19, 6), (19, 9), (8, 10)), hl=[(4, 7)]).shape('croc', poly((1, 11), (8, 11), (19, 12), (18, 15), (8, 16), (2, 14)))
            .ink('#ffffff', many(*[line([(x, 10), (x + 1, 11)]) for x in (6, 9, 12, 15)])).shape('yellow', ell(7, 2, 11, 6)).ink('#141414', rect(9, 3, 9, 5)))
@icon('💍')
def _ring(): return Icon().shape('gold', many(ell(3, 7, 16, 19)), hl=[(5, 10), (6, 9)]).cut(ell(6, 10, 13, 16)).shape('cyan', poly((9.5, 1), (13, 4), (9.5, 8), (6, 4)), hl=[(9, 3)])
@icon('📿')
def _beads():
    ic = Icon()
    for i in range(10):
        a = math.pi * .1 + i * math.pi * .08 + math.pi * .1; x, y = 10 - 7 * math.cos(a * 2.2), 4 + 9 * math.sin(a * 1.1)
    for k, (x, y) in enumerate(((3, 3), (3, 7), (4, 11), (6, 14), (13, 14), (15, 11), (16, 7), (16, 3))):
        ic.shape('wood', ell(x - 1.6, y - 1.6, x + 1.6, y + 1.6))
    return ic.shape('purple', ell(7, 13, 12, 19), hl=[(8, 14)])
@icon('🧿')
def _charm():
    return (Icon().ink('#c8c8cc', line([(10, 0), (10, 3)])).shape('blue', ell(2, 3, 17, 18), hl=[(5, 6), (6, 5)]).shape('white', ell(5, 6, 14, 15)).shape('cyan', ell(7, 8, 12, 13)).ink('#141414', ell(9, 10, 10, 11)))

ORDER = list(dict.fromkeys(ICONS.keys()))

def build(cols=12):
    """the atlas and, for each emoji, its cell index"""
    seen, cells = {}, []
    for k in ORDER:
        f = ICONS[k]
        if f not in seen: seen[f] = len(cells); cells.append(f().image())
    rows = (len(cells) + cols - 1) // cols
    atlas = Image.new('RGBA', (cols * N, rows * N), (0, 0, 0, 0))
    for i, im in enumerate(cells): atlas.paste(im, ((i % cols) * N, (i // cols) * N))
    return atlas, {k: seen[ICONS[k]] for k in ORDER}, cols, rows

if __name__ == '__main__':
    atlas, idx, cols, rows = build()
    out = sys.argv[1] if len(sys.argv) > 1 else 'icons_preview.png'
    big = atlas.resize((atlas.width * 4, atlas.height * 4), Image.NEAREST)
    bg = Image.new('RGBA', big.size, (40, 52, 48, 255)); bg.alpha_composite(big); bg.save(out)
    print(len(idx), 'emoji,', cols * rows, 'cells', atlas.size)
