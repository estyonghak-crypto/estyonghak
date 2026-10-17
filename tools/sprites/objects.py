"""Still things for the sprite pipeline: trees, rocks, bushes, ore and (later) buildings.

Every object is a pile of boxes rendered once, from the same iso camera as the creatures, and packed into one atlas.
The camera looks at the model the way the game's world axes look on screen: model +z is world +x (down-right on screen),
model +x is world -y (up-right), so a box that is long in z lies along the world x axis.
"""
import math, random
import numpy as np
from PIL import Image
import render as R
from render import Node, box, rx, ry, rz, ramp, MAT

FACE = 45          # the facing (degrees) that lines model axes up with the world axes on screen

MAT.update({
    'trunk':  ramp('#241810', '#392616', '#503620', '#6a482b'),
    'leafA':  ramp('#16301f', '#264a2e', '#46693f', '#7e9962'), 'leafB': ramp('#222f18', '#385225', '#59733a', '#8f9d62'), 'leafC': ramp('#122e2f', '#224944', '#406e5d', '#759683'),
    'leafA2': ramp('#1d3d26', '#31603a', '#548049', '#92b072'), 'leafB2': ramp('#2e3d1d', '#4a6a2a', '#6b8844', '#a3b46c'), 'leafC2': ramp('#173b3a', '#2a5c52', '#4c836f', '#86ab98'),
    'snowL':  ramp('#1f3a42', '#325861', '#587e85', '#8aabb0'), 'snowL2': ramp('#243f39', '#3b635a', '#648a7f', '#93ada3'), 'snowL3': ramp('#2a3b4a', '#415a70', '#6c8399', '#96aab9'),
    'snow':   ramp('#8fa6bc', '#b9cbdb', '#dfeaf3', '#ffffff'),
    'stone':  ramp('#3f4348', '#5a5f65', '#7a7f86', '#9fa4ab'), 'moss': ramp('#2c4426', '#41622f', '#5f8442', '#88a85e'),
    'bushg':  ramp('#1a3320', '#2a4b2d', '#416a3d', '#678c54'), 'berry': ramp('#5a0f1c', '#8e1a2c', '#c4283e', '#e8607a'),
    'orerock': ramp('#2a2e32', '#3b4046', '#4f555c', '#666d75'),
    'copper': ramp('#7a3f12', '#a85a22', '#d9803f', '#ffc58a'), 'iron': ramp('#5c6b7d', '#7f92a8', '#9fb2c7', '#e3eefa'), 'goldore': ramp('#8a6a10', '#bf9a1f', '#f2c84b', '#fff2a8'),
})
MAT.update({
    'plank':  ramp('#3e2a16', '#5e4224', '#80592f', '#a07840'), 'plank2': ramp('#2e1f10', '#46301a', '#604424', '#7c5a30'),
    'thatch': ramp('#4a3a20', '#6a5430', '#8c7242', '#ad9258'),
    'wallst': ramp('#555b60', '#767c81', '#9a9fa4', '#bcc1c5'), 'walliron': ramp('#2f3841', '#47515b', '#69757f', '#8e9ba6'),
    'sail':   ramp('#8a8272', '#b3ab98', '#d9d0b8', '#f6eed8'), 'sail2': ramp('#6a7384', '#8a94a8', '#aab4c6', '#cdd6e4'),
    'red':    ramp('#7a1f14', '#a82e1c', '#d9472f', '#ff7a60'), 'blue': ramp('#1c2f6a', '#2c4a9a', '#3b6fd6', '#7fa6f2'),
    'furn':   ramp('#4a4e53', '#686d72', '#868b90', '#a5abb0'), 'brickd': ramp('#3a3430', '#524a44', '#6b625a', '#857a70'),
    'exitl':  ramp('#a88a50', '#d4b06a', '#f2d48c', '#fff0c0'), 'hull': ramp('#3a2610', '#573a1c', '#7d552b', '#a8723a'), 'hull2': ramp('#2a2420', '#3f3730', '#5a5048', '#766a60'),
    'ash':    ramp('#1c1c1e', '#2a2a2d', '#3a3a3e', '#4c4c50'),
})
MAT.update({
    'logw': ramp('#3f2a14', '#5f4120', '#83592b', '#a87a40'), 'logw2': ramp('#4d3419', '#71502a', '#9a6b36', '#c28d4d'), 'coal': ramp('#101012', '#1a1a1d', '#26262a', '#34343a'),
})
ORE_MAT = {'copper': 'copper', 'iron': 'iron', 'gold': 'goldore'}

def seeded(name): return random.Random(sum(ord(c) * (i + 3) for i, c in enumerate(name)))

def _canopy(b, root, r, lumps, pal, size, cell=.34):
    """a rounded crown made of small cubes: the union of some ellipsoids, only the cubes you can see, two leaf tones"""
    cells = set()
    for (cx, cy, cz, rx_, ry_, rz_) in lumps:
        n = int(max(rx_, ry_, rz_) / cell) + 2
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    x, y, z = i * cell, j * cell, k * cell
                    d = (x / rx_) ** 2 + (y / ry_) ** 2 + (z / rz_) ** 2
                    if d <= 1 + r.uniform(-.28, .22):
                        cells.add((round((cx + x) / cell), round((cy + y) / cell), round((cz + z) / cell)))
    top = max(c[1] for c in cells)
    for (i, j, k) in sorted(cells):
        if all(n in cells for n in ((i + 1, j, k), (i - 1, j, k), (i, j + 1, k), (i, j - 1, k), (i, j, k + 1), (i, j, k - 1))): continue      # buried inside the crown
        m = pal + '2' if r.random() < (.18 + .22 * (j - top + 4) / 4 if j >= top - 3 else .1) else pal
        box(b, root, (i * cell * size, j * cell * size, k * cell * size), (cell * size * 1.04,) * 3, m, 'leaf')

def tree(i, size=1.0, stump=False):
    r = seeded('tree%d' % i); shape = i % 3; pal = ('leafA', 'leafB', 'leafC')[i // 3]
    b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    th = (1.5 if shape != 1 else 1.2) * size
    if stump: return _stump(size)
    box(b, root, (0, th / 2, 0), (.30 * size, th, .30 * size), 'trunk', 'trunk'); box(b, root, (0, .1, 0), (.46 * size, .2, .46 * size), 'trunk', 'root')
    if shape == 0:                                                     # round crown
        _canopy(b, root, r, [(0, 2.05, 0, 1.05, .9, 1.05), (.55, 1.8, .2, .7, .6, .7), (-.5, 1.85, -.2, .7, .6, .7)], pal, size)
        box(b, Node(root, (.18 * size, 1.2 * size, 0), rz(-.7)), (0, .0, 0), (.08 * size, .6 * size, .08 * size), 'trunk', 'branch')
    elif shape == 1:                                                   # tall, narrow crown
        _canopy(b, root, r, [(0, 2.3, 0, .8, 1.25, .8), (.3, 1.7, .2, .75, .55, .75), (-.25, 2.9, -.1, .55, .55, .55)], pal, size)
    elif shape == 2:                                                   # a pine: stacked, narrowing tiers, ragged at the bottom
        for k in range(6):
            w = (1.6 - k * .27) * size; y = (1.0 + k * .38) * size
            n = Node(root, (r.uniform(-.04, .04) * size, y, r.uniform(-.04, .04) * size), ry(k * .7)); box(b, n, (0, .2 * size, 0), (w, .42 * size, w), pal if k % 2 == 0 else pal + '2', 'tier%d' % k)
        box(b, root, (0, 3.35 * size, 0), (.26 * size, .5 * size, .26 * size), pal, 'tip')
    return b

def _stump(size):
    b = []; root = Node(None, (0, 0, 0), ry(.4))
    box(b, root, (0, .22 * size, 0), (.34 * size, .44 * size, .34 * size), 'trunk', 'stump'); box(b, root, (0, .02, 0), (.5 * size, .08, .5 * size), 'trunk', 'root')
    return b

def snow_pine(i, size=1.0):
    r = seeded('sp%d' % i); pal = ('snowL', 'snowL2', 'snowL3')[i]
    b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    box(b, root, (0, .6 * size, 0), (.30 * size, 1.2 * size, .30 * size), 'trunk', 'trunk')
    for k in range(4):
        w = (1.55 - k * .36) * size; y = (.95 + k * .5) * size
        n = Node(root, (0, y, 0), ry(k * .8)); box(b, n, (0, .27 * size, 0), (w, .56 * size, w), pal, 'tier%d' % k)
        box(b, n, (0, .58 * size, 0), (w * .9, .1 * size, w * .9), 'snow', 'cap%d' % k)
    box(b, root, (0, 2.95 * size, 0), (.3 * size, .5 * size, .3 * size), 'snow', 'tip')
    return b

def rock(i, size=1.0):
    r = seeded('rock%d' % i); b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    for k in range(3):
        s = r.uniform(.5, .85) * size * (1 if k == 0 else .72)
        n = Node(root, (r.uniform(-.28, .28) * size, s * .3, r.uniform(-.25, .25) * size), ry(r.uniform(0, 3)) @ rx(r.uniform(-.3, .3)) @ rz(r.uniform(-.3, .3)))
        box(b, n, (0, 0, 0), (s * 1.15, s * .8, s), 'stone', 'rock%d' % k)
    if i % 2 == 0:
        box(b, Node(root, (.0, .62 * size, .05), ry(.5)), (0, 0, 0), (.45 * size, .06, .36 * size), 'moss', 'moss')
    return b

def bush(i, berries=3):
    r = seeded('bush%d' % i); b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    spots = [(-.28, .26, .0, .6), (.3, .24, .06, .56), (0, .3, -.28, .52), (.04, .24, .3, .5), (0, .55, 0, .5)]
    for k, (x, y, z, s) in enumerate(spots):
        n = Node(root, (x, y, z), ry(r.uniform(0, 1.5)) @ rx(r.uniform(-.15, .15))); box(b, n, (0, 0, 0), (s, s * .85, s), 'bushg', 'b%d' % k)
    pos = [(-.38, .42, .26), (.4, .34, -.1), (.05, .72, .12)]
    for k in range(berries): box(b, root, pos[k], (.17, .17, .17), 'berry', 'berry%d' % k)
    return b

def ore(kind, i, rubble=False):
    r = seeded('ore%s%d' % (kind, i)); b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    for k in range(3):
        s = r.uniform(.5, .8) * (1 if k == 0 else .7)
        n = Node(root, (r.uniform(-.3, .3), s * .28, r.uniform(-.25, .25)), ry(r.uniform(0, 3)) @ rx(r.uniform(-.25, .25)) @ rz(r.uniform(-.25, .25)))
        box(b, n, (0, 0, 0), (s * 1.1, s * .75, s), 'orerock', 'rk%d' % k)
    for k in range(2 if rubble else 4):
        n = Node(root, (r.uniform(-.35, .35), .5 if not rubble else .1, r.uniform(-.3, .3)), ry(r.uniform(0, 3)) @ rx(r.uniform(-.4, .4)) @ rz(r.uniform(-.4, .4)))
        h = r.uniform(.42, .78) if not rubble else .14; w = .18 if not rubble else .12
        box(b, n, (0, h / 2, 0), (w, h, w), ORE_MAT[kind], 'cr%d' % k)
    return b

# ---------------------------------------------------------------- atlas
PPU = {'tree': 22, 'stump': 22, 'snowpine': 22, 'rock': 21, 'bush': 22, 'ore': 22, 'rubble': 22, 'wreck': 24}      # nature is drawn a little smaller than people-made things (26)

def objects_list():
    out = []
    for i in range(9): out += [('tree%d' % i, tree(i)), ('tree%d_s' % i, tree(i, .78)), ('stump%d' % i, tree(i, 1.0, True))]
    for i in range(3): out += [('snowpine%d' % i, snow_pine(i)), ('snowpine%d_s' % i, snow_pine(i, .78))]
    for i in range(6): out += [('rock%d' % i, rock(i)), ('rock%d_m' % i, rock(i, .72)), ('rock%d_s' % i, rock(i, .5))]
    for i in range(4):
        for n in range(4): out.append(('bush%d_%d' % (i, n), bush(i, n)))
    for kind in ('copper', 'iron', 'gold'):
        for i in range(3): out.append(('ore_%s%d' % (kind, i), ore(kind, i)))
        out.append(('rubble_' + kind, ore(kind, 0, True)))
    return [(n, b, next((v for k, v in PPU.items() if n.startswith(k)), 26)) for n, b in out]

def render_static(boxes, ppu=26.0):
    k, info = R.render(boxes, math.radians(FACE), ppu)
    return R.shade_image(k, info)

def build_atlas(items, ppu=26.0, width=640):
    tiles = []
    for item in items:
        name, boxes = item[0], item[1]; im = render_static(boxes, item[2] if len(item) > 2 else ppu); a = im[..., 3]
        ys, xs = np.nonzero(a); x0, x1, y0, y1 = xs.min() - 1, xs.max() + 2, ys.min() - 1, ys.max() + 2
        tiles.append((name, Image.fromarray(im).crop((x0, y0, x1, y1)), R.ANCHOR[0] - x0, R.ANCHOR[1] - y0))
    x = y = rowh = 0; place = {}
    for name, t, ax, ay in sorted(tiles, key=lambda t: -t[1].height):
        if x + t.width > width: x = 0; y += rowh; rowh = 0
        place[name] = [x, y, t.width, t.height, ax, ay]; x += t.width; rowh = max(rowh, t.height)
    atlas = Image.new('RGBA', (width, y + rowh), (0, 0, 0, 0))
    for name, t, ax, ay in tiles: atlas.alpha_composite(t, tuple(place[name][:2]))
    return atlas, place


# ---------------------------------------------------------------- things people build (axes: +z = world +x, -x = world +y)
def fire_pit(burnt=False):
    b = []; root = Node(None, (0, 0, 0), ry(0))
    for k in range(7):
        a = k / 7 * 6.283; box(b, Node(root, (math.cos(a) * .46, .07, math.sin(a) * .46), ry(-a)), (0, 0, 0), (.2, .14, .17), 'stone', 'ring%d' % k)
    box(b, root, (0, .01, 0), (.78, .02, .78), 'ash', 'ash')
    for sg, a in ((1, .6), (-1, -.6)):
        box(b, Node(root, (0, .15, 0), ry(a) @ rz(.12 * sg)), (0, 0, 0), (.95, .13, .15), 'ash' if burnt else 'trunk', 'log%d' % sg)
    return b

def shelter():                                                          # a small log cabin: six rows of logs, a doorway in the front wall, a pitched shingle roof and a stone chimney
    b = []; root = Node(None, (0, 0, 0), ry(0))
    for row in range(6):
        y = .11 + row * .2; long_, short = (2.0, 1.55) if row % 2 == 0 else (1.55, 2.0); m = 'logw' if row % 2 else 'logw2'
        for sg in (1, -1):
            if sg == 1 and 1 <= row <= 4:                               # the doorway
                for sx in (1, -1): box(b, root, (sx * .62, y, .72), (.62, .2, .22), m, 'fx%d_%d' % (row, sx))
            else: box(b, root, (0, y, sg * .72), (long_, .2, .22), m, 'fx%d_%d' % (row, sg))
            box(b, root, (sg * .72, y, 0), (.22, .2, short), m, 'fz%d_%d' % (row, sg))
    box(b, root, (0, .55, .62), (.5, .9, .08), 'dark', 'door')
    for sg in (1, -1):
        n = Node(root, (0, 1.58, sg * .46), rx(sg * .7)); box(b, n, (0, 0, 0), (2.4, .1, 1.3), 'plank', 'roof%d' % sg)
        for k in range(6): box(b, n, (0, .06, -.55 + k * .22), (2.42, .03, .1), 'plank2' if k % 2 == 0 else 'plank', 'shingle%d_%d' % (sg, k))
    box(b, root, (0, 1.95, 0), (2.4, .1, .14), 'plank2', 'ridge'); box(b, root, (.7, 2.2, -.3), (.34, .7, .34), 'stone', 'chimney')
    box(b, root, (-.72 - .12, .6, .0), (.04, .3, .3), 'dark', 'window'); box(b, root, (.0, .06, .95), (.9, .12, .3), 'stone', 'step')
    return b

def furnace(on=False):                                                  # a forge: stone hearth with a bed of coals under a hood and chimney, an anvil on a stump, bellows behind
    b = []; root = Node(None, (0, 0, 0), ry(0))
    box(b, root, (-.2, .3, 0), (1.2, .6, 1.0), 'furn', 'hearth'); box(b, root, (-.2, .62, 0), (1.0, .08, .8), 'glow' if on else 'coal', 'coals')
    for sx in (1, -1):
        for sz in (1, -1): box(b, root, (-.2 + sx * .5, 1.0, sz * .4), (.12, .8, .12), 'plank2', 'post')
    box(b, root, (-.2, 1.42, 0), (1.2, .12, 1.0), 'walliron', 'hood'); box(b, root, (-.2, 1.9, -.2), (.34, 1.0, .34), 'brickd', 'chimney')
    box(b, root, (.95, .2, .2), (.34, .4, .34), 'trunk', 'stump'); box(b, root, (.95, .46, .2), (.5, .12, .24), 'metal', 'anvil'); box(b, root, (.95, .54, .2), (.3, .08, .18), 'metal', 'anvilTop'); box(b, root, (1.18, .5, .2), (.18, .06, .1), 'metal', 'horn')
    box(b, Node(root, (-.2, .4, -.7), rx(.25)), (0, 0, 0), (.5, .5, .12), 'plank', 'bellows'); box(b, root, (-.2, .22, -.95), (.3, .12, .3), 'plank2', 'bellowsPipe')
    return b

def boat(big=False, flip=False):
    b = []; root = Node(None, (0, 0, 0), ry(math.pi if flip else 0)); hull = 'hull2' if big else 'hull'
    box(b, root, (0, .26, -.05), (.95, .34, 2.0), hull, 'keel')
    box(b, root, (0, .38, -.6), (1.25, .4, .9), hull, 'stern'); box(b, root, (0, .5, .75), (.7, .46, .8), hull, 'bow'); box(b, root, (0, .72, 1.12), (.38, .3, .4), hull, 'prow')
    box(b, root, (0, .6, -.05), (1.0, .06, 1.7), 'plank', 'deck')
    for sg in (1, -1): box(b, root, (sg * .52, .62, -.05), (.07, .22, 1.8), hull, 'rail%d' % sg)
    if big:
        for z in (-.6, -.2, .2, .6): box(b, root, (0, .76, z), (1.0, .14, .08), 'metal', 'plate')
    box(b, root, (0, 1.65, .0), (.09, 2.1, .09), 'plank2', 'mast')
    box(b, root, (0, 1.65, -.42), (.04, 1.6, .8), 'sail2' if big else 'sail', 'sail'); box(b, root, (0, 1.65, .45), (.04, 1.3, .5), 'sail2' if big else 'sail', 'jib')
    box(b, root, (0, 2.78, .05), (.05, .14, .45), 'blue' if big else 'red', 'flag')
    if big: box(b, root, (0, 1.5, -.95), (.07, 1.5, .07), 'plank2', 'mast2'); box(b, root, (0, 1.55, -1.2), (.04, 1.0, .4), 'sail2', 'sail2')
    return b

def mine_door():
    b = []; root = Node(None, (0, 0, 0), ry(0))
    box(b, root, (0, .9, -.25), (3.4, 1.8, 1.3), 'stone', 'mound'); box(b, root, (0, 1.9, -.3), (2.6, .7, 1.1), 'stone', 'mound2'); box(b, root, (.2, 2.4, -.35), (1.4, .5, .9), 'stone', 'mound3')
    for sg in (1, -1): box(b, root, (sg * 1.05, .65, .45), (.5, 1.3, .5), 'stone', 'pillar%d' % sg)
    box(b, root, (0, 1.4, .45), (2.1, .3, .5), 'stone', 'lintel')
    box(b, root, (0, .6, .5), (1.5, 1.2, .3), 'dark', 'hole')
    for sg in (1, -1): box(b, root, (sg * .78, .65, .62), (.14, 1.3, .14), 'plank', 'beam%d' % sg)
    box(b, root, (0, 1.32, .62), (1.7, .14, .14), 'plank', 'crossbeam'); box(b, root, (0, 1.8, .66), (1.1, .46, .08), 'plank2', 'sign')
    for sg in (1, -1): box(b, root, (sg * 1.45, .55, .65), (.1, .5, .1), 'plank2', 'lamp%d' % sg); box(b, root, (sg * 1.45, .85, .65), (.2, .16, .2), 'glow', 'flame%d' % sg)
    return b

def cave_exit():
    b = []; root = Node(None, (0, 0, 0), ry(0))
    box(b, root, (0, 1.0, -.2), (2.6, 2.0, 1.2), 'stone', 'wall'); box(b, root, (0, 2.1, -.25), (2.0, .5, 1.0), 'stone', 'top')
    box(b, root, (0, .8, .68), (1.3, 1.6, .12), 'exitl', 'light'); box(b, root, (0, 1.72, .68), (.9, .3, .12), 'exitl', 'lightTop')
    for sg in (1, -1): box(b, root, (sg * .74, .85, .74), (.18, 1.7, .22), 'stone', 'jamb%d' % sg)
    box(b, root, (0, 1.84, .74), (1.7, .2, .22), 'stone', 'lintel')
    for z in range(5): box(b, root, (0, .1 + z * .3, .56), (.1, .06, .06), 'plank2', 'rung%d' % z) if False else None
    for sg in (1, -1): box(b, root, (sg * .17, .85, .8), (.07, 1.7, .07), 'plank2', 'ladder%d' % sg)
    for k in range(6): box(b, root, (0, .15 + k * .26, .8), (.34, .05, .06), 'plank2', 'rung%d' % k)
    return b

def wreck(i, size=1.0):
    r = seeded('wreck%d' % i); b = []; root = Node(None, (0, 0, 0), ry(.3))
    box(b, root, (0, .1 * size, 0), (1.0 * size, .2 * size, .5 * size), 'plank2', 'floor')
    for k in range(3): box(b, Node(root, (r.uniform(-.3, .3), .3 * size, r.uniform(-.2, .2)), ry(r.uniform(0, 3)) @ rz(r.uniform(-.4, .4))), (0, 0, 0), (.9 * size, .1, .16), 'plank', 'plank%d' % k)
    box(b, root, (-.2 * size, .38 * size, .1), (.12, .6 * size, .12), 'metal', 'pipe'); box(b, Node(root, (.3 * size, .22 * size, -.15), ry(.5)), (0, 0, 0), (.3 * size, .3 * size, .3 * size), 'plank', 'crate')
    box(b, root, (.4 * size, .12, .2), (.36, .05, .14), 'red', 'rag')
    return b

def wall_post(kind):                                                    # a fence post (the rails come as separate pieces)
    b = []; root = Node(None, (0, 0, 0), ry(0))
    box(b, root, (0, .38, 0), (.18, .76, .18), 'plank', 'post'); box(b, root, (0, .8, 0), (.24, .08, .24), 'plank2', 'cap')
    return b

def rail(axis):                                                         # 'x': from a post to the next one along world +x (model +z); 'y': along world +y (model -x)
    b = []; root = Node(None, (0, 0, 0), ry(0 if axis == 'x' else -math.pi / 2))
    for y in (.3, .56): box(b, root, (0, y, .54), (.07, .09, 1.1), 'plank', 'rail')
    return b

def gate():
    b = []; root = Node(None, (0, 0, 0), ry(0))
    for sg in (1, -1): box(b, root, (sg * .5, .45, 0), (.14, .9, .2), 'plank2', 'gpost%d' % sg)
    box(b, root, (0, .42, 0), (.9, .66, .1), 'plank', 'leaf')
    for y in (.2, .64): box(b, root, (0, y, .06), (.9, .08, .05), 'plank2', 'bar')
    box(b, Node(root, (0, .42, .07), rz(.62)), (0, 0, 0), (1.05, .07, .05), 'plank2', 'brace'); box(b, root, (.3, .42, .09), (.08, .08, .05), 'gold', 'latch')
    return b

def wall_block(kind):
    b = []; root = Node(None, (0, 0, 0), ry(0)); m = 'wallst' if kind == 'stone' else 'walliron'
    box(b, root, (0, .38, 0), (1.04, .76, 1.04), m, 'block'); box(b, root, (0, .79, 0), (1.1, .1, 1.1), m, 'cap')
    box(b, root, (0, .4, .53), (1.0, .04, .02), 'dark', 'seam') if kind == 'stone' else None
    if kind == 'iron':
        for sx, sz in ((.4, .4), (-.4, .4), (.4, -.4), (-.4, -.4)): box(b, root, (sx, .86, sz), (.1, .06, .1), 'metal', 'rivet')
    return b

def tower():
    b = []; root = Node(None, (0, 0, 0), ry(0))
    for sx in (1, -1):
        for sz in (1, -1): box(b, root, (sx * .42, .62, sz * .42), (.14, 1.24, .14), 'plank2', 'leg')
    box(b, root, (0, 1.28, 0), (1.15, .12, 1.15), 'plank', 'deck')
    box(b, root, (0, 1.78, 0), (.92, .9, .92), 'plank', 'cabin'); box(b, root, (0, 1.82, .47), (.5, .26, .04), 'dark', 'slit')
    box(b, Node(root, (0, 2.34, 0), ry(.0)), (0, 0, 0), (1.2, .12, 1.2), 'plank2', 'roof1'); box(b, root, (0, 2.5, 0), (.8, .2, .8), 'plank2', 'roof2'); box(b, root, (0, 2.66, 0), (.4, .16, .4), 'plank2', 'roof3')
    for z in (-1, 1): box(b, root, (0, .5, z * .42), (.1, .1, .84), 'plank', 'brace%d' % z) if False else None
    return b

def building_list():
    out = [('fire', fire_pit()), ('fire_out', fire_pit(True)), ('shelter', shelter()), ('furnace', furnace(), 22), ('furnace_on', furnace(True), 22),
           ('boat_e', boat()), ('boat_w', boat(False, True)), ('boat2_e', boat(True)), ('boat2_w', boat(True, True)),
           ('minedoor', mine_door()), ('caveexit', cave_exit()), ('gate', gate()), ('tower', tower()),
           ('wall_wood', wall_post('wood')), ('rail_x', rail('x')), ('rail_y', rail('y')), ('wall_stone', wall_block('stone')), ('wall_iron', wall_block('iron'))]
    for i in range(3): out.append(('wreck%d' % i, wreck(i, 1.0))); out.append(('wreck%d_m' % i, wreck(i, .72))); out.append(('wreck%d_s' % i, wreck(i, .5)))
    return out
