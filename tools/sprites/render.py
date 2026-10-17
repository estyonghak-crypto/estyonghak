"""Low-poly 3D characters -> isometric pixel-art sprite sheets for the game (numpy + PIL only).

Versions of this file: render.py was the first sample (64x96 frames). This one is sized for the game
(a person is ~46 px tall), has tools (axe, pickaxe, spear, rifle), attack / hurt / die poses, a zombie
and a bulky brute, and paints the survivor's jacket in a marker colour so the game can recolour it per armour tier.
"""
import math, numpy as np
from PIL import Image, ImageDraw

SS = 4                      # supersampling while rasterising, reduced to 1x with a per-pixel vote
BIG = (120, 112)            # working canvas for one frame (cropped to the used area afterwards)
ANCHOR = (60, 94)           # where the feet touch the ground inside the working canvas
PX_PER_UNIT = 26            # 1 model unit (about a metre) in final pixels
PITCH = math.radians(30)    # 30 deg camera pitch = the ground is squashed to 0.5 (2:1 tiles)
LIGHT = np.array([-.488, .78, .39])
OUTLINE_DARK = .42

def hexc(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
def ramp(*c): return [hexc(x) for x in c]
MAT = {
  'skin':   ramp('#8c6a52', '#b58d6e', '#d2aa86', '#e3c3a0'),
  'hair':   ramp('#1d1814', '#2c241d', '#3d3228', '#52443a'),
  'armor':  [(66, 36, 60), (106, 60, 98), (141, 79, 130), (175, 98, 160)],       # marker hue: the game swaps these for the armour colours
  'pants':  ramp('#1c2028', '#2a303b', '#3a4250', '#4b5565'),
  'boots':  ramp('#14100d', '#241b15', '#352a21', '#473a2e'),
  'pack':   ramp('#2a3320', '#3b4a2d', '#516340', '#687d52'),
  'strap':  ramp('#17140f', '#241f17', '#322b20', '#41382a'),
  'eye':    ramp('#0c0a09', '#0c0a09', '#0c0a09', '#0c0a09'),
  'wood':   ramp('#3a2814', '#54391d', '#6f4d29', '#8b6437'),
  'metal':  ramp('#3a4148', '#59636b', '#7f8a92', '#a7b2ba'),
  'dark':   ramp('#15181b', '#22272c', '#30363c', '#40484f'),
  'zskin':  ramp('#434f3c', '#5c6b50', '#788a69', '#93a681'),
  'zshirt': ramp('#23262c', '#32363f', '#454a55', '#585f6c'),
  'zpants': ramp('#1d1f22', '#2b2d31', '#3a3d43', '#4a4e56'),
  'blood':  ramp('#33090a', '#521010', '#741919', '#912222'),
  'zeye':   ramp('#7a1010', '#b01818', '#d62a2a', '#ff5a4a'),
  'bskin':  ramp('#4b4756', '#645f73', '#807a92', '#9a94ad'),
  'bshirt': ramp('#1d1826', '#2b2436', '#3c3548', '#4d4560'),
  'bpants': ramp('#17151c', '#221f29', '#2f2b38', '#3d3948'),
}

def rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

class Node:
    """a joint: world = parent * (translate, rotate)"""
    def __init__(self, parent=None, pos=(0, 0, 0), rot=None):
        self.R = np.eye(3) if rot is None else rot; self.p = np.array(pos, float)
        if parent is not None: self.p = parent.p + parent.R @ self.p; self.R = parent.R @ self.R

FAT = [1.0]
def box(boxes, node, off, size, mat, name):
    f = FAT[0]
    boxes.append((node.p + node.R @ np.array(off, float), np.array(size, float) * np.array([f, 1, f]), node.R, mat, name))

FACES = [((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]), ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
         ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]), ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
         ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]), ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)])]

def project(P):               # world (x right, y up, z toward the camera) -> screen px (y down) + closeness
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    return x, -y * math.cos(PITCH) + z * math.sin(PITCH), y * math.sin(PITCH) + z * math.cos(PITCH)

def render(boxes, facing, ppu):
    FW, FH = BIG
    F = ry(facing); view = np.array([0, math.sin(PITCH), math.cos(PITCH)])
    faces = []
    for bi, (c, s, R, mat, name) in enumerate(boxes):
        Rw = F @ R; cw = F @ c
        for n, vs in FACES:
            nw = Rw @ np.array(n, float)
            if nw @ view <= 1e-6: continue
            pts = np.array([cw + Rw @ (np.array(v, float) * s / 2) for v in vs])
            sx, sy, cl = project(pts)
            diff = max(0.0, float(nw @ LIGHT)); sh = .4 + .62 * diff
            tone = 0 if sh < .5 else 1 if sh < .67 else 2 if sh < .8 else 3
            faces.append((cl.mean(), sx, sy, mat, tone, name))
    faces.sort(key=lambda f: f[0])                                  # far to near
    W, H = FW * SS, FH * SS
    key = Image.new('I', (W, H), 0); d = ImageDraw.Draw(key)
    info = {}
    for i, (cl, sx, sy, mat, tone, name) in enumerate(faces, 1):
        info[i] = (mat, tone, name)
        d.polygon([(ANCHOR[0] * SS + x * ppu * SS, ANCHOR[1] * SS + y * ppu * SS) for x, y in zip(sx, sy)], fill=i)
    a = np.array(key, dtype=np.int32).reshape(FH, SS, FW, SS).transpose(0, 2, 1, 3).reshape(FH, FW, SS * SS)
    eq = (a[..., :, None] == a[..., None, :]).sum(-1)               # vote: the face that covers most of each pixel
    k = np.take_along_axis(a, eq.argmax(-1)[..., None], -1)[..., 0]
    cover = (a > 0).sum(-1); k[cover < SS * SS // 2] = 0            # covered less than half -> transparent
    return k, info

def shade_image(k, info, outline=True, inner=True):
    FH, FW = k.shape
    img = np.zeros((FH, FW, 4), np.uint8)
    names = {key: v[2] for key, v in info.items()}
    ys, xs = np.nonzero(k)
    for y, x in zip(ys, xs):
        mat, tone, name = info[k[y, x]]; img[y, x] = (*MAT[mat][tone], 255)
    if inner:                                                       # a darker line on the farther side where two parts meet
        for y, x in zip(ys, xs):
            a = k[y, x]
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < FH and 0 <= xx < FW:
                    b = k[yy, xx]
                    if b > a and names[b] != names[a] and info[a][1] >= 1:      # neighbour is nearer: darken my edge pixel
                        img[y, x, :3] = (img[y, x, :3] * .72).astype(np.uint8); break
    if outline:
        solid = k > 0; pad = np.pad(solid, 1)
        edge = ~solid & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
        for y, x in zip(*np.nonzero(edge)):
            col = None
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < FH and 0 <= xx < FW and solid[yy, xx]: col = img[yy, xx, :3]; break
            img[y, x] = (*(np.array(col) * OUTLINE_DARK).astype(np.uint8), 255)
    return img

# ------------------------------------------------------------------ the people
def tool_boxes(b, hand, tool):
    """the tool sits in the right hand; +z is along the hand's forward axis"""
    if tool == 'spear':
        box(b, hand, (0, 0, .35), (.05, .05, 1.35), 'wood', 'shaft'); box(b, hand, (0, 0, 1.08), (.08, .06, .22), 'metal', 'tip')
    elif tool == 'axe':
        box(b, hand, (0, 0, .26), (.05, .05, .78), 'wood', 'shaft')
        box(b, hand, (0, .04, .60), (.05, .22, .09), 'dark', 'axeback'); box(b, hand, (0, .06, .66), (.035, .26, .12), 'metal', 'axehead')
    elif tool == 'pickaxe':
        box(b, hand, (0, 0, .30), (.05, .05, .86), 'wood', 'shaft')
        box(b, hand, (0, 0, .72), (.12, .08, .08), 'dark', 'pickmid')
        for sg in (1, -1):
            n = Node(hand, (sg * .12, 0, .72), rz(sg * .5)); box(b, n, (sg * .13, 0, 0), (.30, .045, .06), 'metal', 'pick' + str(sg))
    elif tool == 'gun':
        box(b, hand, (0, -.02, -.12), (.07, .15, .46), 'wood', 'stock')
        box(b, hand, (0, .0, .30), (.075, .10, .46), 'wood', 'fore'); box(b, hand, (0, .035, .78), (.045, .045, .55), 'metal', 'barrel')
        box(b, hand, (0, .09, .26), (.04, .05, .30), 'dark', 'sight'); box(b, hand, (0, -.10, .28), (.05, .12, .06), 'dark', 'mag')

def person(pose, look='survivor', tool=None):
    """pose: dict of angles (radians). +z is the front, +x the person's left. positive arm/leg angle swings forward."""
    z = look in ('zombie', 'brute')
    if look == 'brute': M = dict(skin='bskin', top='bshirt', legs='bpants', boots='strap'); FAT[0] = 1.28
    elif z: M = dict(skin='zskin', top='zshirt', legs='zpants', boots='strap'); FAT[0] = 1.0
    else: M = dict(skin='skin', top='armor', legs='pants', boots='boots'); FAT[0] = 1.0
    b = []
    fall = pose.get('fall', 0)                                     # falling backwards about the feet
    hip_y = .98 + pose.get('bob', 0)
    root = Node(None, (0, hip_y * math.cos(fall) + .13 * math.sin(fall), -hip_y * math.sin(fall) + pose.get('sway', 0)), rx(-fall) @ ry(pose.get('twist', 0)))
    box(b, root, (0, .04, 0), (.50, .20, .28), M['legs'], 'pelvis')
    chest = Node(root, (0, .14, 0), rx(-pose.get('lean', 0)) @ ry(-pose.get('twist', 0) * 1.6))
    box(b, chest, (0, .30, 0), (.50, .60, .28), M['top'], 'torso')
    box(b, chest, (0, .02, 0), (.53, .07, .31), 'strap', 'belt')
    box(b, chest, (0, .61, 0), (.46, .06, .29), M['top'], 'collar')
    if not z:
        box(b, chest, (0, .30, -.23), (.38, .46, .20), 'pack', 'pack'); box(b, chest, (0, .56, -.30), (.34, .08, .06), 'pack', 'rollTop'); box(b, chest, (.15, .38, .12), (.04, .40, .17), 'strap', 'pstrapL'); box(b, chest, (-.15, .38, .12), (.04, .40, .17), 'strap', 'pstrapR')
        box(b, chest, (0, .28, .145), (.035, .52, .02), 'strap', 'zip1')
    else:
        box(b, chest, (-.10, .36, .155), (.16, .20, .02), 'blood', 'stain'); box(b, chest, (.12, .12, .155), (.12, .10, .02), 'blood', 'stain2')
        if look == 'brute': box(b, chest, (.0, .28, -.15), (.4, .5, .02), 'blood', 'stain3'); box(b, chest, (.24, .58, 0), (.12, .10, .34), 'bshirt', 'padL'); box(b, chest, (-.24, .58, 0), (.12, .10, .34), 'bshirt', 'padR')
    neck = Node(chest, (0, .63, 0), rx(-pose.get('head_lean', 0)) @ rz(pose.get('head_roll', 0)))
    box(b, neck, (0, .03, 0), (.14, .09, .14), M['skin'], 'neck')
    box(b, neck, (0, .26, 0), (.40, .40, .38), M['skin'], 'head')
    if not z:
        box(b, neck, (0, .45, -.02), (.42, .10, .38), 'hair', 'hairtop'); box(b, neck, (0, .31, -.20), (.42, .30, .05), 'hair', 'hairback')
        box(b, neck, (.215, .36, -.03), (.03, .20, .30), 'hair', 'hairL'); box(b, neck, (-.215, .36, -.03), (.03, .20, .30), 'hair', 'hairR')
        box(b, neck, (0, .425, .20), (.40, .07, .04), 'hair', 'fringe')
        box(b, neck, (-.10, .28, .195), (.07, .07, .02), 'eye', 'eyeR'); box(b, neck, (.10, .28, .195), (.07, .07, .02), 'eye', 'eyeL')
    else:
        box(b, neck, (0, .43, -.02), (.36, .06, .36), M['skin'], 'scalp'); box(b, neck, (-.09, .27, .185), (.07, .06, .02), 'zeye', 'eyeR'); box(b, neck, (.09, .27, .185), (.07, .06, .02), 'zeye', 'eyeL')
        box(b, neck, (0, .15, .185), (.17, .04, .02), 'blood', 'mouth')
        if look == 'zombie': box(b, neck, (-.14, .38, .0), (.10, .10, .36), 'hair', 'tuft')
    for side, sgn in (('L', 1), ('R', -1)):
        ang = pose.get('arm' + side, 0); elb = pose.get('elbow' + side, 0); out = pose.get('out' + side, 0)
        sh = Node(chest, (sgn * .36 * (1 + (FAT[0] - 1) * .6), .54, 0), rz(-sgn * out) @ rx(-ang))
        box(b, sh, (0, -.17, 0), (.17, .37, .17), M['top'], 'uarm' + side)
        el = Node(sh, (0, -.35, 0), rx(-elb))
        box(b, el, (0, -.15, 0), (.15, .32, .15), M['skin'] if z else M['top'], 'farm' + side)
        box(b, el, (0, -.34, 0), (.14, .13, .14), M['skin'], 'hand' + side)
        if tool and not z and side == 'R':
            hand = Node(el, (0, -.34, 0), rx(-pose.get('grip', 1.2)))
            tool_boxes(b, hand, tool)
    for side, sgn in (('L', 1), ('R', -1)):
        ang = pose.get('leg' + side, 0); kn = pose.get('knee' + side, 0)
        hp = Node(root, (sgn * .13, -.02, 0), rx(-ang))
        box(b, hp, (0, -.23, 0), (.20, .48, .22), M['legs'], 'thigh' + side)
        kt = Node(hp, (0, -.47, 0), rx(kn))
        box(b, kt, (0, -.22, 0), (.17, .45, .19), M['legs'], 'shin' + side)
        ft = Node(kt, (0, -.45, 0), rx(0.0))
        box(b, ft, (0, -.04, .06), (.19, .13, .32), M['boots'], 'foot' + side)
    return b

# ------------------------------------------------------------------ poses
def lerp(a, b, t): return a + (b - a) * t
def mix(p, q, t): return {k: lerp(p.get(k, 0), q.get(k, 0), t) for k in set(p) | set(q)}

# how the right arm holds each tool: (grip, carry (arm, elbow), windup, impact, follow-through)
TOOLPOSE = {
    'hand':    dict(grip=0,   carry=(.0, .25),  wind=(-.6, .6),  hit=(1.25, .1),  follow=(.9, .2)),
    'spear':   dict(grip=1.2, carry=(.75, 1.2), wind=(-.15, 1.1), hit=(1.15, .12), follow=(.9, .3)),
    'axe':     dict(grip=.8,  carry=(.45, .7),  wind=(2.5, .6),   hit=(.9, .15),   follow=(.5, .1)),
    'pickaxe': dict(grip=.8,  carry=(.45, .7),  wind=(2.5, .6),   hit=(.9, .15),   follow=(.5, .1)),
    'gun':     dict(grip=1.2, carry=(.45, .5),  wind=(.7, .7),    hit=(.62, .7),   follow=(.7, .7)),
}

def survivor_pose(anim, phase, tool=None):
    tp = TOOLPOSE[tool or 'hand']
    s = math.sin(phase * 2 * math.pi)
    if anim == 'idle':
        p = dict(bob=.012 * s, armL=.05 + .03 * s, armR=-.03 - .03 * s, elbowL=.2, elbowR=.2, lean=.02, head_lean=.02 * s, outL=.04, outR=.04)
    elif anim == 'walk':
        ph = phase * 2 * math.pi; c = math.cos(ph); A = .62
        p = dict(legL=A * s, legR=-A * s, armL=-.55 * s, armR=.55 * s, elbowL=.25 + .2 * max(0, s), elbowR=.25 + .2 * max(0, -s),
                 bob=-.035 * abs(s) + .01, twist=.10 * s, lean=.05)
        p['kneeL'] = max(0, math.sin(ph + math.pi / 2)) * .85 if s < 0 else .08
        p['kneeR'] = max(0, math.sin(ph + math.pi / 2 + math.pi)) * .85 if s > 0 else .08
    elif anim == 'attack':                                          # phase 0..1: windup, swing, impact, follow-through
        steps = [tp['wind'], mix_t(tp['wind'], tp['hit'], .5), tp['hit'], tp['follow']]
        i = min(3, int(phase * 4)); a, e = steps[i]
        p = dict(armR=a, elbowR=e, armL=.3, elbowL=.4, lean=[.12, -.05, -.22, -.12][i], twist=[-.35, -.1, .3, .2][i], bob=[.01, -.01, -.04, -.02][i], head_lean=-.05,
                 legL=[-.2, .1, .4, .3][i], legR=[.3, .0, -.3, -.2][i], kneeL=.15, kneeR=.15)
        if tool == 'gun':                                           # shouldering the rifle: left hand forward on the grip, small recoil
            p.update(armL=1.0, elbowL=.55, lean=[0, -.06, -.02, -.04][i], twist=0, bob=0, legL=.2, legR=-.2, armR=[.6, .62, .5, .56][i], elbowR=[.75, .7, .85, .8][i])
    elif anim == 'hurt':
        p = dict(lean=.28, head_lean=.22, armL=.5, armR=.4, elbowL=.6, elbowR=.5, outL=.3, outR=.3, bob=-.04, legL=-.18, legR=.2, kneeL=.2, kneeR=.15)
    else: raise ValueError(anim)
    if anim in ('idle', 'walk') or (anim == 'hurt'):
        if tool:                                                   # carry the tool at the ready
            a, e = tp['carry']; p.update(armR=a + (.06 * s if anim == 'idle' else .05 * s), elbowR=e)
            if tool == 'gun':
                p.update(armL=.8, elbowL=.5)
    p['grip'] = -tp['grip'] if tool else 0
    return p

def mix_t(p, q, t): return (lerp(p[0], q[0], t), lerp(p[1], q[1], t))

def zombie_pose(anim, phase, look='zombie'):
    big = look == 'brute'
    ph = phase * 2 * math.pi; s = math.sin(ph); A = .34 if not big else .3
    base = dict(armL=1.45, armR=1.35, elbowL=.25, elbowR=.35, lean=-.2, head_lean=-.12, outL=.10, outR=.14)
    if anim == 'idle':
        p = dict(base, bob=.014 * s, armL=1.35 + .06 * s, armR=1.25 - .06 * s, head_roll=.10 * s)
    elif anim == 'walk':
        p = dict(base, legL=A * s, legR=-A * s, head_roll=.12 * s, twist=.16 * s, bob=-.05 * abs(s),
                 armL=1.45 + .1 * s, armR=1.3 - .1 * s)
        p['kneeL'] = max(0, math.sin(ph + math.pi / 2)) * .6 if s < 0 else .1
        p['kneeR'] = max(0, math.sin(ph + math.pi / 2 + math.pi)) * .6 if s > 0 else .1
    elif anim == 'attack':                                          # lunge: arms rear back, then swipe forward and down
        i = min(3, int(phase * 4))
        p = dict(base, armL=[2.4, 2.0, .95, 1.2][i], armR=[2.2, 1.8, .9, 1.1][i], elbowL=[.5, .3, .1, .2][i], elbowR=[.5, .3, .1, .2][i],
                 lean=[.08, -.2, -.5, -.35][i], head_lean=[.1, -.1, -.3, -.2][i], bob=[.0, -.03, -.1, -.07][i], legL=[-.2, .1, .4, .3][i], legR=[.2, -.1, -.3, -.2][i], kneeL=.3, kneeR=.3)
    elif anim == 'hurt':
        p = dict(base, lean=.3, head_lean=.35, armL=.9, armR=.7, elbowL=.7, elbowR=.8, outL=.5, outR=.5, bob=-.04, legL=-.2, legR=.2, kneeL=.3, kneeR=.2, twist=.2)
    elif anim == 'die':                                             # topples backwards
        t = min(1, phase); e = 1 - (1 - t) ** 2
        p = dict(base, fall=lerp(0, 1.5, e), lean=lerp(-.2, .1, e), armL=lerp(1.45, 2.4, e), armR=lerp(1.35, 2.0, e), elbowL=lerp(.25, .8, e), elbowR=lerp(.35, .3, e),
                 legL=lerp(0, .3, e), legR=lerp(0, -.1, e), kneeL=lerp(0, .4, e), kneeR=lerp(0, .2, e), head_lean=lerp(-.12, .25, e), outL=lerp(.1, .6, e), outR=lerp(.14, .6, e))
    else: raise ValueError(anim)
    return p

DIRS = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
FACING = {'S': 0, 'SW': -45, 'W': -90, 'NW': -135, 'N': 180, 'NE': 135, 'E': 90, 'SE': 45}     # E faces screen-right

def frame_img(look, tool, anim, phase, d):
    pose = survivor_pose(anim, phase, tool) if look == 'survivor' else zombie_pose(anim, phase, look)
    ppu = PX_PER_UNIT * (1.42 if look == 'brute' else 1.0)
    k, info = render(person(pose, look, tool if look == 'survivor' else None), math.radians(FACING[d]), ppu)
    return shade_image(k, info)

ANIMS = {   # name -> frames
    'survivor': [('idle', 4), ('walk', 8), ('attack', 4), ('hurt', 1)],
    'zombie':   [('idle', 4), ('walk', 8), ('attack', 4), ('hurt', 1), ('die', 5)],
    'brute':    [('idle', 4), ('walk', 8), ('attack', 4), ('hurt', 1), ('die', 5)],
}

def build_sheet(look, tool=None):
    """returns (RGBA image cropped to the used area, meta)"""
    seq = ANIMS[look]
    frames = {}                                                    # (dir index, col) -> RGBA array
    col = 0; cols = {}
    for name, n in seq:
        cols[name] = [col, n]
        for i in range(n):
            ph = i / n if name != 'die' else (i + 1) / n
            for r, d in enumerate(DIRS): frames[(r, col)] = frame_img(look, tool, name, ph, d)
            col += 1
    ncol = col
    ys = np.concatenate([np.nonzero(f[..., 3])[0] for f in frames.values()]); xs = np.concatenate([np.nonzero(f[..., 3])[1] for f in frames.values()])
    x0, x1, y0, y1 = xs.min() - 1, xs.max() + 2, ys.min() - 1, ys.max() + 2
    # keep the foot anchor in the middle column so every frame is anchored the same way
    half = max(ANCHOR[0] - x0, x1 - ANCHOR[0]); x0, x1 = ANCHOR[0] - half, ANCHOR[0] + half
    cw, chh = x1 - x0, y1 - y0
    out = Image.new('RGBA', (ncol * cw, 8 * chh), (0, 0, 0, 0))
    for (r, c), f in frames.items():
        out.alpha_composite(Image.fromarray(f).crop((x0, y0, x1, y1)), (c * cw, r * chh))
    meta = dict(w=int(cw), h=int(chh), ax=int(ANCHOR[0] - x0), ay=int(ANCHOR[1] - y0), cols=cols, ncol=ncol)
    return out, meta
