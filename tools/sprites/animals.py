"""Four-legged animals, the rabbit and the bat for the sprite pipeline (see render.py)."""
import math
import numpy as np
import render as R
from render import Node, box, rx, ry, rz, lerp, LOOKS

# ------------------------------------------------------------------ species
# body = (width, height, length) of the torso, leg = leg length, head/snout/ear = box sizes, tail = (w, h, length)
SPECIES = {
    'wolf':  dict(body=(.34, .34, .78), leg=.50, legw=.11, head=(.26, .26, .26), snout=(.13, .12, .22), ear=(.07, .15, .05), tail=(.10, .10, .46),
                  fur='wfur', light='wlight', nose='dark', A=.62, K=.9, ppu=1.15, eye='eye'),
    'wolfking': dict(body=(.34, .34, .78), leg=.50, legw=.12, head=(.27, .27, .27), snout=(.14, .12, .22), ear=(.07, .15, .05), tail=(.10, .10, .46),
                  fur='kfur', light='klight', nose='dark', A=.58, K=.9, ppu=1.95, eye='zeye', mane=True, crown=True),
    'boar':  dict(body=(.50, .46, .90), leg=.30, legw=.15, head=(.32, .30, .30), snout=(.20, .18, .20), ear=(.09, .10, .05), tail=(.05, .05, .16),
                  fur='bfur', light='bdark', nose='bsnout', A=.50, K=.7, ppu=1.2, eye='eye', tusks=True, ridge=True),
    'bear':  dict(body=(.64, .62, .96), leg=.46, legw=.20, head=(.40, .38, .38), snout=(.22, .17, .18), ear=(.12, .12, .08), tail=(.10, .10, .10),
                  fur='brown', light='brown', nose='tan', A=.42, K=.6, ppu=1.25, eye='eye'),
    'mountainking': dict(body=(.64, .62, .96), leg=.46, legw=.22, head=(.42, .40, .40), snout=(.24, .18, .20), ear=(.12, .12, .08), tail=(.10, .10, .10),
                  fur='armor', light='armor', nose='tan', A=.36, K=.6, ppu=2.1, eye='zeye', tusks=True, ridge=True, scars=True),
    'rabbit': dict(body=(.20, .20, .34), leg=.12, legw=.07, head=(.16, .15, .15), snout=(.08, .07, .07), ear=(.045, .24, .04), tail=(.09, .09, .09),
                  fur='rfur', light='rfur', nose='pink', A=.5, K=.5, ppu=1.6, eye='eye', hop=True),
}

def quad(sp, pose):
    b = []
    w, h, l = sp['body']; leg = sp['leg']; lw = sp['legw']; hw, hh, hl = sp['head']; sw, sh, sl = sp['snout']
    fur, light = sp['fur'], sp['light']
    rear = pose.get('rear', 0); dip = pose.get('dip', 0); roll = pose.get('roll', 0)
    y0 = leg + h / 2 + pose.get('bob', 0)
    pivot = np.array([0, 0, -l * .38]) if rear else np.zeros(3)
    pos = pivot + rx(-rear) @ (np.array([0, y0, 0]) - pivot) + np.array([0, 0, pose.get('lunge', 0)])
    Rroot = rz(roll) @ rx(dip - rear)
    pos = rz(roll) @ pos + np.array([0, (w / 2) * abs(math.sin(roll)), 0])
    root = Node(None, pos, Rroot)
    box(b, root, (0, 0, 0), (w, h, l), fur, 'body')
    if light != fur: box(b, root, (0, -h * .32, .02), (w * .9, h * .36, l * .85), light, 'belly')
    if sp.get('ridge'):
        for i in range(5): box(b, root, (0, h / 2 + .03, l * (.36 - i * .17)), (.07, .09, .1), 'bdark' if sp['fur'] != 'armor' else 'armor', 'ridge' + str(i))
    if sp.get('mane'):
        for i in range(5): box(b, root, (0, h / 2 + .06, l * (.34 - i * .16)), (.06, .14 + (.05 if i % 2 == 0 else 0), .1), 'klight', 'mane' + str(i))
    if sp.get('scars'):
        box(b, root, (w / 2 + .005, .05, .12), (.01, .05, .3), 'blood', 'scarL'); box(b, root, (-w / 2 - .005, .0, -.1), (.01, .05, .28), 'blood', 'scarR')
    ph = pose.get('ph', 0); A = pose.get('A', 0)
    hop = pose.get('hop')
    for sx in (1, -1):
        for sz in (1, -1):
            front = sz > 0
            if hop is not None: sw_ = A * math.sin(ph) * (1 if front else -1) * .9; kn = sp['K'] * max(0, math.cos(ph)) * (1 if front else .6)
            else:
                sgn = 1 if (sx > 0) == front else -1                            # diagonal pairs move together
                sw_ = sgn * A * math.sin(ph); kn = sp['K'] * max(0, math.cos(ph + (0 if sgn > 0 else math.pi)))
            if front: sw_ += pose.get('fswing', 0)
            sw_ += pose.get('spread', 0) * sx * (1 if front else -1)
            hip = Node(root, (sx * (w / 2 - lw * .5), -h / 2 * .9, sz * (l / 2 - lw * .75)), (rx(rear) if not front else np.eye(3)) @ rx(-sw_))
            box(b, hip, (0, -leg * .26, 0), (lw, leg * .56, lw), fur, 'up')
            knee = Node(hip, (0, -leg * .5, 0), rx(kn))
            box(b, knee, (0, -leg * .26, 0), (lw * .85, leg * .56, lw * .85), fur, 'low'); box(b, knee, (0, -leg * .5, lw * .45), (lw * 1.1, .05, lw * 1.7), 'dark' if fur != 'rfur' else 'rfur', 'paw')
    neck = Node(root, (0, h * .22, l / 2 - .06), rx(pose.get('head_down', 0)))
    box(b, neck, (0, .01, .02), (w * .62, h * .62, .2), fur, 'neck')
    zc = .10 + hl / 2
    box(b, neck, (0, .03, zc), (hw, hh, hl), fur, 'head')
    box(b, neck, (0, 0, zc + hl / 2 + sl / 2 - .03), (sw, sh, sl), sp['nose'] if sp['nose'] != 'dark' else light, 'snout')
    box(b, neck, (0, .0, zc + hl / 2 + sl - .02), (sw * .6, sh * .55, .04), 'dark' if sp['nose'] != 'pink' else 'pink', 'nose')
    jaw = Node(neck, (0, -hh * .35, .10 + hl * .5), rx(pose.get('jaw', 0)))
    box(b, jaw, (0, -.015, sl * .55 + hl * .2), (sw * .9, .05, sl + hl * .45), light if light != fur else fur, 'jaw')
    eyes = sp['eye']
    box(b, neck, (-hw * .5, hh * .18 + .03, zc + hl * .35), (.04, .045, .06), eyes, 'eyeR'); box(b, neck, (hw * .5, hh * .18 + .03, zc + hl * .35), (.04, .045, .06), eyes, 'eyeL')
    ew, eh, ed = sp['ear']
    for sx in (1, -1):
        box(b, neck, (sx * hw * .32, .03 + hh / 2 + eh / 2 - .02, zc - hl * .2 + pose.get('ear', 0)), (ew, eh, ed), fur, 'ear' + str(sx))
        if sp.get('hop'): box(b, neck, (sx * hw * .32, .03 + hh / 2 + eh / 2 - .02, zc - hl * .2 + .026), (ew * .45, eh * .8, .01), 'pink', 'earin' + str(sx))
    if sp.get('tusks'):
        for sx in (1, -1): box(b, neck, (sx * sw * .62, .02 + sh * .3, zc + hl / 2 + sl * .55), (.04, .13, .05), 'tusk', 'tusk' + str(sx))
    if sp.get('crown'):
        for i, dx in enumerate((-.09, -.045, 0, .045, .09)): box(b, neck, (dx, .03 + hh / 2 + .02 + (.03 if i % 2 == 0 else 0), zc - .02), (.04, .08 + (.04 if i % 2 == 0 else 0), .05), 'gold', 'crown' + str(i))
        box(b, neck, (0, .03 + hh / 2 + .01, zc - .02), (hw + .02, .04, .14), 'gold', 'crownBand')
    tw, th, tl = sp['tail']
    tail = Node(root, (0, h * .28, -l / 2 + .03), ry(pose.get('wag', 0)) @ rx(pose.get('tail_up', 0)))
    box(b, tail, (0, 0, -tl / 2), (tw, th, tl), light if sp.get('hop') else fur, 'tail')
    return b

# ------------------------------------------------------------------ poses
def quad_pose(name, anim, phase, tool=None):
    sp = SPECIES[name]; ph = phase * 2 * math.pi; s = math.sin(ph); A = sp['A']
    hop = sp.get('hop')
    p = dict(ph=ph, A=0)
    if anim == 'idle':
        p.update(bob=.008 * s, head_down=.04 * s, wag=.25 * s, tail_up=.1)
        if hop: p.update(head_down=.12, ear=.0, tail_up=0)
    elif anim == 'walk':
        if hop:                                                       # a hop: legs gather, push, and the body arcs through the air
            air = max(0, math.sin(ph))
            p.update(A=.7, hop=1, bob=.2 * air, dip=-.25 * math.cos(ph) * .6, head_down=-.1 + .2 * math.cos(ph), tail_up=.2, ear=-.02 * air)
        else:
            p.update(A=A, bob=-.02 * abs(s), head_down=.12 + .05 * s, wag=.35 * math.sin(ph * 2), tail_up=.15, dip=.02 * s)
    elif anim == 'attack':
        i = min(3, int(phase * 4)); p.update(A=0)
        if name in ('wolf', 'wolfking'):
            p.update(dip=[-.12, .1, .25, .08][i], head_down=[-.25, .1, .3, .1][i], jaw=[.55, .6, .02, .25][i], lunge=[-.05, .12, .24, .1][i], tail_up=.3, bob=[.0, -.02, -.05, -.02][i], fswing=[-.2, .3, .5, .2][i])
        elif name == 'boar':
            p.update(dip=[.05, .15, .25, .05][i], head_down=[.5, .6, .0, .15][i], jaw=[.2, .3, .1, .1][i], lunge=[-.05, .14, .22, .08][i], fswing=[-.2, .3, .4, .1][i], bob=[.0, -.03, -.04, -.01][i])
        else:                                                         # bears rise up and swipe
            p.update(rear=[.55, .75, .35, .12][i], dip=0, head_down=[.0, -.2, .25, .1][i], jaw=[.5, .6, .3, .2][i], fswing=[1.0, 1.6, .5, .3][i], lunge=[0, .04, .12, .04][i])
    elif anim == 'hurt':
        p.update(A=0, head_down=-.35, jaw=.5, rear=.12, bob=-.02, dip=-.05, fswing=-.2)
    elif anim == 'die':
        t = min(1, phase); e = 1 - (1 - t) ** 2
        p.update(A=0, roll=lerp(0, 1.5, e), head_down=lerp(0, -.25, e), jaw=lerp(0, .5, e), spread=lerp(0, .5, e), tail_up=lerp(0, .1, e), bob=lerp(0, -.04, e))
    else: raise ValueError(anim)
    return p

# ------------------------------------------------------------------ the bat
def bat(pose):
    b = []
    ph = pose.get('ph', 0); flap = math.sin(ph) * pose.get('amp', 1.0)
    root = Node(None, (0, .85 + pose.get('bob', 0) + .06 * math.cos(ph) * pose.get('amp', 1), 0), ry(pose.get('yaw', 0)) @ rx(pose.get('dip', 0)) @ rz(pose.get('roll', 0)))
    box(b, root, (0, 0, 0), (.20, .20, .28), 'batf', 'body'); box(b, root, (0, -.04, .02), (.16, .1, .22), 'batw', 'belly')
    box(b, root, (0, .05, .2), (.17, .16, .15), 'batf', 'head'); box(b, root, (0, -.01, .29), (.09, .07, .06), 'batf', 'snout')
    box(b, root, (-.045, .09, .275), (.04, .04, .02), 'bred', 'eyeR'); box(b, root, (.045, .09, .275), (.04, .04, .02), 'bred', 'eyeL')
    for sx in (1, -1):
        box(b, root, (sx * .06, .17, .18), (.04, .1, .04), 'batf', 'ear' + str(sx))
        inner = Node(root, (sx * .09, .05, -.0), rz(sx * (flap * .8 + .25)))
        box(b, inner, (sx * .2, 0, -.02), (.4, .03, .34), 'batw', 'wingin' + str(sx)); box(b, inner, (sx * .02, 0, 0), (.05, .04, .3), 'batf', 'arm' + str(sx))
        outer = Node(inner, (sx * .4, 0, 0), rz(sx * (flap * .9 - .2)))
        box(b, outer, (sx * .2, 0, -.06), (.4, .025, .28), 'batw', 'wingout' + str(sx)); box(b, outer, (sx * .2, .014, .07), (.4, .025, .035), 'batf', 'finger' + str(sx))
        box(b, root, (sx * .05, -.13, -.1), (.03, .08, .03), 'batf', 'foot' + str(sx))
    return b

def bat_pose(anim, phase, tool=None):
    ph = phase * 2 * math.pi
    p = dict(ph=ph, amp=1.0)
    if anim == 'fly': p.update(bob=.02 * math.sin(ph), dip=.12)
    elif anim == 'hurt': p.update(amp=.3, roll=.5, dip=-.4, bob=-.05)
    elif anim == 'die':
        t = min(1, phase); p.update(amp=.2, roll=lerp(.3, 3.0, t), dip=lerp(0, .9, t), bob=-.8 * t * t)
    else: raise ValueError(anim)
    return p

# ------------------------------------------------------------------ registry
_Q = [('idle', 2), ('walk', 6), ('attack', 4), ('hurt', 1), ('die', 4)]
for _n, _sp in SPECIES.items():
    frames = [('idle', 2), ('walk', 6), ('hurt', 1), ('die', 3)] if _n == 'rabbit' else _Q
    LOOKS[_n] = dict(frames=frames, pose=(lambda n: lambda an, ph, tool: quad_pose(n, an, ph, tool))(_n), model=(lambda sp: lambda pose, tool: quad(sp, pose))(_sp), ppu=_sp['ppu'])
LOOKS['bat'] = dict(frames=[('fly', 6), ('hurt', 1), ('die', 4)], pose=lambda an, ph, tool: bat_pose(an, ph, tool), model=lambda pose, tool: bat(pose), ppu=1.9)
LOOKS['batqueen'] = dict(frames=[('fly', 6), ('hurt', 1), ('die', 4)], pose=lambda an, ph, tool: bat_pose(an, ph, tool), model=R.swapped(dict(batf='qfur', batw='qwing', bred='gold'), lambda pose, tool: bat(pose)), ppu=4.3)      # boss: the vampire bat queen
