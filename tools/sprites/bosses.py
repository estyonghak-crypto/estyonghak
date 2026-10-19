"""The zone bosses, the new night bosses and the creatures of the new islands, for the sprite pipeline (see render.py, animals.py).

Most are variants of the existing models (a four-legged body with other proportions, the golem or the brute painted with other
materials); the spider and the snake are new models.
"""
import math
import numpy as np
import render as R
import animals as A
from render import Node, box, rx, ry, rz, lerp, LOOKS, swapped

# ------------------------------------------------------------------ four-legged variants
A.SPECIES.update({
    'boarking': dict(body=(.50, .46, .90), leg=.30, legw=.15, head=(.32, .30, .30), snout=(.20, .18, .20), ear=(.09, .10, .05), tail=(.05, .05, .16),
                     fur='bkfur', light='bkdark', nose='bsnout', A=.48, K=.7, ppu=2.25, eye='zeye', tusks=True, ridge=True, crown=True, scars=True, fam='boar'),
    'mole':     dict(body=(.62, .44, .80), leg=.20, legw=.20, head=(.36, .30, .30), snout=(.16, .12, .22), ear=(.04, .04, .03), tail=(.06, .06, .12),
                     fur='molef', light='molef', nose='pink', A=.5, K=.6, ppu=2.0, eye='eye', claws=True, fam='bear'),
    'crocking': dict(body=(.52, .30, 1.30), leg=.22, legw=.14, head=(.34, .22, .30), snout=(.26, .13, .42), ear=(.03, .03, .03), tail=(.24, .18, .90),
                     fur='croc', light='cbelly', nose='croc', A=.4, K=.5, ppu=1.6, eye='gold', tusks=True, ridge=True, ridgemat='cdark', crown=True, fam='wolf', tail_drop=.32),
    'ninetail': dict(body=(.28, .30, .70), leg=.44, legw=.09, head=(.24, .22, .24), snout=(.10, .09, .20), ear=(.08, .17, .04), tail=(.11, .11, .55),
                     fur='nfur', light='nlight', nose='dark', A=.62, K=.9, ppu=1.85, eye='spirit', tails=9, tip='spirit', fam='wolf'),
    'turtle':   dict(body=(.80, .34, .90), leg=.28, legw=.20, head=(.26, .24, .28), snout=(.16, .12, .12), ear=(.03, .03, .03), tail=(.08, .08, .16),
                     fur='tskin', light='tskin', nose='tskin', A=.36, K=.4, ppu=2.0, eye='eye', shell=True, fam='boar'),
})

def turtle_pose(anim, phase, tool=None):
    if anim == 'shell':                                               # pulled into the shell, resting on the ground (two frames: a slow spin)
        return dict(tuck=1, bob=-A.SPECIES['turtle']['leg'] + .03, roll=0, dip=0)
    return A.quad_pose('turtle', anim, phase, tool)

_Q = [('idle', 2), ('walk', 6), ('attack', 4), ('hurt', 1), ('die', 4)]
for _n in ('boarking', 'mole', 'crocking', 'ninetail'):
    _sp = A.SPECIES[_n]
    LOOKS[_n] = dict(frames=_Q, pose=(lambda n: lambda an, ph, tool: A.quad_pose(n, an, ph, tool))(_n), model=(lambda sp: lambda pose, tool: A.quad(sp, pose))(_sp), ppu=_sp['ppu'])
LOOKS['turtle'] = dict(frames=_Q + [('shell', 1)], pose=turtle_pose, model=lambda pose, tool: A.quad(A.SPECIES['turtle'], pose), ppu=A.SPECIES['turtle']['ppu'])

# ------------------------------------------------------------------ the spider (8 legs, two segments each; a tetrapod gait)
def spider(pose, queen=False):
    b = []
    body = 'sqfur' if queen else 'spf'; legm = 'sqleg' if queen else 'spf'
    L1, L2, up, down = .36, .50, .7, 1.25
    y0 = L2 * math.sin(down) - L1 * math.sin(up) + .02 + pose.get('bob', 0)
    root = Node(None, (0, y0, 0), rx(pose.get('dip', 0)) @ rz(pose.get('roll', 0)))
    box(b, root, (0, .02, .06), (.30, .18, .30), body, 'thorax')
    box(b, root, (0, .14 + pose.get('abd', 0), -.36), (.46, .38, .52), body, 'abdomen')
    if queen:
        box(b, root, (0, .335 + pose.get('abd', 0), -.36), (.16, .02, .26), 'bred', 'mark'); box(b, root, (0, .335 + pose.get('abd', 0), -.2), (.24, .02, .06), 'bred', 'mark2')
        for i, dx in enumerate((-.08, 0, .08)): box(b, root, (dx, .16 + (.03 if i == 1 else 0), .12), (.04, .08, .04), 'gold', 'crown' + str(i))
    box(b, root, (0, .05, .23), (.2, .14, .08), body, 'head')
    for i, (ex, ey) in enumerate(((-.06, .1), (.06, .1), (-.03, .13), (.03, .13))): box(b, root, (ex, ey, .27), (.04, .04, .02), 'bred', 'eye' + str(i))
    fang = pose.get('fang', 0)
    for sx in (1, -1): box(b, Node(root, (sx * .05, -.02, .27), rx(.5 + fang) @ rz(sx * .2)), (0, -.06, 0), (.04, .12, .04), 'tusk' if queen else 'dark', 'fang' + str(sx))
    ph = pose.get('ph', 0); amp = pose.get('A', 0); curl = pose.get('curl', 0)
    for sx in (1, -1):
        for k, (zp, yaw) in enumerate(((.16, -.95), (.08, -.35), (0, .3), (-.08, .9))):
            grp = (k + (0 if sx > 0 else 1)) % 2                      # alternating tetrapod: L1 R2 L3 R4 / R1 L2 R3 L4
            sw = amp * math.sin(ph + grp * math.pi); lift = amp * .6 * max(0, math.cos(ph + grp * math.pi))
            raise_ = pose.get('front', 0) if k == 0 else 0
            hip = Node(root, (sx * .14, .02, zp), ry(sx * (yaw + sw * .45)) @ rz(sx * (up + lift + raise_ + curl)))
            box(b, hip, (sx * L1 / 2, 0, 0), (L1, .05, .05), legm, 'up%d%d' % (sx, k))
            knee = Node(hip, (sx * L1, 0, 0), rz(-sx * (up + down - curl * 1.6)))
            box(b, knee, (sx * L2 / 2, 0, 0), (L2, .04, .04), legm, 'low%d%d' % (sx, k))
    return b

def spider_pose(anim, phase, tool=None):
    ph = phase * 2 * math.pi; s = math.sin(ph)
    p = dict(ph=ph, A=0)
    if anim == 'idle': p.update(bob=.01 * s, abd=.01 * s)
    elif anim == 'walk': p.update(A=.5, bob=.012 * abs(s))
    elif anim == 'attack':
        i = min(3, int(phase * 4)); p.update(dip=[-.35, -.45, .15, -.1][i], front=[.6, .8, -.2, .2][i], fang=[.6, .8, -.3, .1][i], bob=[.05, .07, -.02, .01][i])
    elif anim == 'hurt': p.update(dip=-.2, bob=.03, curl=.25, fang=.4)
    elif anim == 'die':
        t = min(1, phase); e = 1 - (1 - t) ** 2; p.update(curl=lerp(0, .9, e), bob=lerp(0, -.18, e), dip=lerp(0, .15, e))
    else: raise ValueError(anim)
    return p

LOOKS['spider'] = dict(frames=_Q, pose=spider_pose, model=lambda pose, tool: spider(pose), ppu=1.05)
LOOKS['spiderqueen'] = dict(frames=_Q, pose=spider_pose, model=lambda pose, tool: spider(pose, True), ppu=2.45)

# ------------------------------------------------------------------ the snake (a chain of segments that S-curves along the ground)
def snake(pose):
    b = []; N = 11; ph = pose.get('ph', 0); amp = pose.get('A', .07); rise = pose.get('rise', 0); lunge = pose.get('lunge', 0); roll = pose.get('roll', 0)
    pts = []
    for i in range(N + 1):
        z = .46 - i * .1 + lunge * max(0, 1 - i / 4)
        y = .07 + rise * max(0, 1 - i / 4) ** 1.5
        x = amp * math.sin(ph - i * .8) * min(1, i / 3 + .25)
        pts.append(np.array([x, y, z]))
    for i in range(N):
        a, c = pts[i], pts[i + 1]; mid = (a + c) / 2; d = a - c
        yaw = math.atan2(d[0], d[2]); pitch = -math.atan2(d[1], math.hypot(d[0], d[2]))
        w = .13 * (1 - i / (N + 4)); n = Node(None, mid, rz(roll) @ ry(yaw) @ rx(pitch))
        box(b, n, (0, 0, 0), (w, w * .8, .13), 'snkband' if i % 3 == 1 else 'snk', 'seg%d' % i)
        box(b, n, (0, -w * .32, 0), (w * .8, w * .2, .12), 'snkbelly', 'belly%d' % i)
    hd = pts[0]; d = pts[0] - pts[1]; yaw = math.atan2(d[0], d[2]); pitch = -math.atan2(d[1], math.hypot(d[0], d[2])) * .5
    head = Node(None, hd + np.array([0, .01, 0]), rz(roll) @ ry(yaw) @ rx(pitch))
    box(b, head, (0, .02, .07), (.17, .1, .18), 'snk', 'head'); box(b, head, (0, .06, .06), (.15, .03, .12), 'snkband', 'cap')
    box(b, head, (-.06, .055, .12), (.035, .035, .04), 'gold', 'eyeR'); box(b, head, (.06, .055, .12), (.035, .035, .04), 'gold', 'eyeL')
    if pose.get('tongue', 0): box(b, head, (0, 0, .2), (.02, .015, .1), 'bred', 'tongue')
    return b

def snake_pose(anim, phase, tool=None):
    ph = phase * 2 * math.pi
    p = dict(ph=ph, A=.05)
    if anim == 'idle': p.update(A=.04, rise=.12 + .02 * math.sin(ph), tongue=1 if phase < .5 else 0)
    elif anim == 'walk': p.update(A=.11)
    elif anim == 'attack':
        i = min(3, int(phase * 4)); p.update(rise=[.3, .36, .16, .2][i], lunge=[-.08, -.1, .26, .12][i], tongue=1 if i == 2 else 0, A=.03)
    elif anim == 'hurt': p.update(rise=.22, A=.12, ph=1.2)
    elif anim == 'die':
        t = min(1, phase); p.update(roll=lerp(0, 2.6, t), A=lerp(.06, .14, t), rise=0)
    else: raise ValueError(anim)
    return p

LOOKS['snake'] = dict(frames=[('idle', 2), ('walk', 6), ('attack', 4), ('hurt', 1), ('die', 3)], pose=snake_pose, model=lambda pose, tool: snake(pose), ppu=1.35)

# ------------------------------------------------------------------ recoloured people and golems, and the fire bat
LOOKS['yeti'] = dict(frames=[('idle', 2), ('walk', 8), ('attack', 4), ('hurt', 1), ('die', 4)], pose=lambda an, ph, tool: R.golem_pose(an, ph),
                     model=swapped(dict(strap='yfur2'), lambda pose, tool: R.person(pose, 'yeti', None)), ppu=1.8)
LOOKS['lavagiant'] = dict(frames=[('idle', 2), ('walk', 8), ('attack', 4), ('hurt', 1), ('die', 4)], pose=lambda an, ph, tool: R.golem_pose(an, ph),
                          model=lambda pose, tool: R.person(pose, 'lava', None), ppu=1.95)
LOOKS['magma'] = dict(frames=[('idle', 2), ('walk', 8), ('attack', 4), ('hurt', 1), ('die', 4)], pose=lambda an, ph, tool: R.golem_pose(an, ph),
                      model=lambda pose, tool: R.person(pose, 'lava', None), ppu=1.05)
LOOKS['firebat'] = dict(frames=[('fly', 6), ('hurt', 1), ('die', 4)], pose=lambda an, ph, tool: A.bat_pose(an, ph, tool),
                        model=swapped(dict(batf='fbat', batw='fwing', bred='magma'), lambda pose, tool: A.bat(pose)), ppu=1.9)
LOOKS['ghostcaptain'] = dict(frames=[('idle', 4), ('walk', 8), ('attack', 4), ('hurt', 1)], pose=lambda an, ph, tool: R.survivor_pose(an, ph, 'sword'),
                             model=lambda pose, tool: R.person(pose, 'captain', 'sword'), ppu=1.6)

# ------------------------------------------------------------------ still things for the new islands (added to the object atlas)
import objects as O
from render import MAT, ramp
MAT.update({
    'palmt':  ramp('#3a2a18', '#574024', '#755834', '#957545'), 'palmt2': ramp('#2c1f12', '#43301c', '#5c4428', '#765936'),
    'frond':  ramp('#173818', '#265a24', '#3c7f32', '#5ea446'), 'frond2': ramp('#1d4520', '#30692c', '#4a9040', '#72b65a'), 'coco': ramp('#2a1a0e', '#3d2715', '#55381f', '#6e4a2a'),
    'char':   ramp('#141010', '#1f1918', '#2c2422', '#3b302d'), 'ember': ramp('#a6300a', '#e0561a', '#ff8a2a', '#ffc860'),
    'basalt': ramp('#1e1b1d', '#2c282b', '#3d383b', '#514b4e'),
    'bone':   ramp('#7d7462', '#a69c86', '#cbc2aa', '#ece4cc'), 'totem': ramp('#2a1a10', '#3f2818', '#573a22', '#71502f'), 'rag': ramp('#4a0e10', '#6e1418', '#961e22', '#c02e2e'),
})

def palm(i, size=1.0):
    r = O.seeded('palm%d' % i); b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    lean = (.16, -.12, .22)[i]; n = Node(root, (0, 0, 0), rx(lean * .4)); seg = .42 * size
    for k in range(6):                                                  # a gently curving, ringed trunk
        box(b, n, (0, seg / 2, 0), (.24 * size * (1 - k * .05), seg * 1.02, .24 * size * (1 - k * .05)), 'palmt' if k % 2 == 0 else 'palmt2', 'seg%d' % k)
        n = Node(n, (0, seg, 0), rx(lean * .25))
    for k in range(7):                                                  # fronds droop from the crown
        a = k / 7 * 6.283 + r.uniform(-.2, .2)
        f = Node(n, (0, .04, 0), ry(a) @ rx(.22 + r.uniform(-.1, .15)))
        box(b, f, (0, 0, .42 * size), (.32 * size, .05, .85 * size), 'frond' if k % 2 else 'frond2', 'frond%d' % k)
        f2 = Node(f, (0, 0, .82 * size), rx(.7)); box(b, f2, (0, 0, .25 * size), (.24 * size, .04, .5 * size), 'frond', 'tip%d' % k)
    for k in range(3): box(b, n, ((k - 1) * .12 * size, -.08, (.08 if k == 1 else -.04) * size), (.13 * size, .13 * size, .13 * size), 'coco', 'coco%d' % k)
    return b

def burnt(i, size=1.0):
    r = O.seeded('burnt%d' % i); b = []; root = Node(None, (0, 0, 0), ry(r.uniform(0, 6.28)))
    th = (1.7, 1.35, 2.0)[i] * size
    box(b, root, (0, th / 2, 0), (.26 * size, th, .26 * size), 'char', 'trunk'); box(b, root, (0, .08, 0), (.44 * size, .16, .44 * size), 'char', 'root')
    for k in range(3):
        y = th * (.45 + k * .2); a = r.uniform(0, 6.28)
        br = Node(root, (0, y, 0), ry(a) @ rz(.9 + r.uniform(-.2, .3))); box(b, br, (0, .3 * size, 0), (.08 * size, .6 * size, .08 * size), 'char', 'br%d' % k)
        if k == 1: box(b, br, (0, .62 * size, 0), (.05, .05, .05), 'ember', 'emb%d' % k)
    box(b, root, (.13 * size, th * .3, 0), (.03, .1 * size, .06), 'ember', 'crack')
    return b

def basalt(i, size=1.0):
    R.SWAP.clear(); R.SWAP.update(dict(stone='basalt', moss='ember'))
    try: return O.rock(i, size)
    finally: R.SWAP.clear()

def lair_totem():
    b = []; root = Node(None, (0, 0, 0), ry(.3))
    box(b, root, (0, 1.0, 0), (.16, 2.0, .16), 'totem', 'pole')
    box(b, root, (0, 2.12, .02), (.34, .3, .32), 'bone', 'skull'); box(b, root, (-.08, 2.14, .19), (.08, .08, .02), 'dark', 'eyeR'); box(b, root, (.08, 2.14, .19), (.08, .08, .02), 'dark', 'eyeL')
    for sg in (1, -1): box(b, Node(root, (sg * .17, 2.22, 0), rz(-sg * .9)), (0, .16, 0), (.08, .34, .08), 'bone', 'horn%d' % sg)
    box(b, root, (0, 1.75, .09), (.36, .18, .02), 'rag', 'rag'); box(b, root, (.1, 1.55, .09), (.12, .26, .02), 'rag', 'rag2')
    for k in range(6):
        a = k / 6 * 6.283; box(b, Node(root, (math.cos(a) * .42, .06, math.sin(a) * .42), ry(-a + .4)), (0, 0, 0), (.42, .08, .09), 'bone', 'bone%d' % k)
    box(b, root, (.3, .12, -.25), (.2, .18, .2), 'bone', 'skull2')
    return b

def extra_objects():
    out = []
    for i in range(3): out += [('palm%d' % i, palm(i), 24), ('palm%d_s' % i, palm(i, .78), 24), ('burnt%d' % i, burnt(i), 24), ('burnt%d_s' % i, burnt(i, .78), 24)]
    for i in range(3): out += [('basalt%d' % i, basalt(i), 26), ('basalt%d_m' % i, basalt(i, .72), 26), ('basalt%d_s' % i, basalt(i, .5), 26)]
    out.append(('lairtotem', lair_totem(), 26))
    return out
