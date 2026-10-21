"""The plum swordsman (매화검수): his own look and a three-strike sword combo.

The survivor's body with the pack taken off: a long split robe (the front panels ride on the thighs, so the legs still
show when he walks), a topknot tied with a plum ribbon, a white inner collar, a plum sash, a blossom on the back and a
tassel on the sword. The robe is painted in the marker colour, so the game dresses it in the job's (or the armour's) colours.
Attacks: 'attack' an overhead cut, 'atk2' a flat sweep across the body, 'atk3' a lunging thrust.
"""
import math
import render as R
from render import Node, box, rx, ry, rz, MAT, ramp, LOOKS, TOOLPOSE, mix, mix_t, survivor_pose, person

MAT['plumpink'] = ramp('#7a2449', '#b23c6c', '#df6e9b', '#ffaccb')
MAT['plumwhite'] = ramp('#77727f', '#a8a2b1', '#d3cedb', '#f2eef6')

PACK = {'pack', 'rollTop', 'pstrapL', 'pstrapR', 'zip1'}

def plum_model(pose, tool):
    b = [x for x in person(pose, 'survivor', tool) if x[4] not in PACK]
    # the same joints person() builds (it does not hand them back)
    fall = pose.get('fall', 0); hip_y = .98 + pose.get('bob', 0)
    root = Node(None, (0, hip_y * math.cos(fall) + .13 * math.sin(fall), -hip_y * math.sin(fall) + pose.get('sway', 0)), rx(-fall) @ ry(pose.get('twist', 0)))
    chest = Node(root, (0, .14, 0), rx(-pose.get('lean', 0)) @ ry(-pose.get('twist', 0) * 1.6))
    neck = Node(chest, (0, .63, 0), rx(-pose.get('head_lean', 0)) @ rz(pose.get('head_roll', 0)))
    # the robe: a back panel on the hips, a front panel on each thigh
    box(b, root, (0, -.27, -.12), (.52, .54, .06), 'armor', 'robeBack')
    for side, sgn in (('L', 1), ('R', -1)):
        hp = Node(root, (sgn * .13, -.02, 0), rx(-pose.get('leg' + side, 0)))
        box(b, hp, (sgn * .02, -.25, .12), (.25, .5, .04), 'armor', 'robe' + side)
        box(b, hp, (sgn * .135, -.25, 0), (.03, .5, .24), 'armor', 'robeS' + side)
        box(b, hp, (sgn * .02, -.49, .125), (.25, .04, .045), 'plumpink', 'hem' + side)
    box(b, chest, (0, .02, 0), (.56, .1, .33), 'plumpink', 'sash')
    box(b, chest, (.17, -.12, .17), (.07, .22, .02), 'plumpink', 'sashTail')
    box(b, chest, (0, .56, .14), (.22, .1, .03), 'plumwhite', 'innerCollar')
    box(b, chest, (0, .36, -.15), (.13, .13, .02), 'plumpink', 'blossom')
    box(b, chest, (0, .36, -.155), (.04, .04, .02), 'plumwhite', 'blossomEye')
    # the topknot, its ribbon and two tails down the back
    box(b, neck, (0, .56, -.05), (.16, .16, .16), 'hair', 'topknot')
    box(b, neck, (0, .5, -.05), (.18, .045, .18), 'plumpink', 'ribbon')
    for sg in (1, -1):
        t = Node(neck, (sg * .04, .49, -.15), rx(.35) @ rz(sg * .15)); box(b, t, (0, -.12, 0), (.045, .26, .02), 'plumpink', 'tail' + str(sg))
    return b

def plum_pose(anim, phase, tool):
    if anim not in ('atk2', 'atk3'):
        p = survivor_pose(anim, phase, tool)
        if anim == 'idle': p.update(armL=-.1, elbowL=.5, outL=.18)        # the free hand behind the back, a swordsman's stance
        return p
    i = min(3, int(phase * 4))
    if anim == 'atk2':                 # a flat sweep: from out on the right, across the front, round to the left
        armR = [1.25, 1.4, 1.45, 1.3][i]; outR = [1.15, .45, -.35, -.8][i]
        p = dict(armR=armR, outR=outR, elbowR=[.5, .2, .05, .15][i], armL=.2, elbowL=.5, outL=.3,
                 twist=[-.55, -.15, .4, .6][i], lean=[.05, -.05, -.12, -.08][i], bob=[.0, -.02, -.035, -.02][i],
                 legL=[-.15, .1, .35, .3][i], legR=[.25, .05, -.25, -.2][i], kneeL=.2, kneeR=.15, head_lean=-.04)
        p['grip'] = -[1.5, 1.45, 1.4, 1.5][i]
    else:                              # a lunge: the sword drawn back, then driven straight out
        p = dict(armR=[.85, 1.2, 1.55, 1.5][i], elbowR=[1.7, .9, .0, .05][i], outR=[.1, .05, .0, .0][i], armL=[.3, -.2, -.6, -.5][i], elbowL=.3, outL=[.2, .3, .45, .4][i],
                 twist=[-.45, -.1, .35, .3][i], lean=[.12, -.1, -.32, -.26][i], bob=[.0, -.03, -.08, -.07][i],
                 legL=[.0, .35, .8, .75][i], legR=[.1, -.25, -.55, -.5][i], kneeL=[.1, .15, .3, .3][i], kneeR=[.1, .2, .35, .3][i], head_lean=-.08)
        p['grip'] = -[.4, .9, 1.55, 1.5][i]
    return p

def plum_tool_boxes(b, hand, tool):
    R_tool(b, hand, tool)
    if tool == 'sword': box(b, hand, (0, -.07, -.16), (.04, .1, .04), 'plumpink', 'tassel'); box(b, hand, (0, -.13, -.18), (.03, .06, .03), 'plumpink', 'tassel2')

R_tool = R.tool_boxes
def _model(pose, tool):
    R.tool_boxes = plum_tool_boxes
    try: return plum_model(pose, tool)
    finally: R.tool_boxes = R_tool

LOOKS['plum'] = dict(frames=[('idle', 4), ('walk', 8), ('attack', 4), ('atk2', 4), ('atk3', 4), ('hurt', 1)], pose=plum_pose, model=_model)
