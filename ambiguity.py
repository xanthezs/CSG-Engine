
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from vecmath import add, sub, mul, norm, TOL
from surfaces import Surface
from csg import R, Lit, And, Or, Not, Halfspace, point_in_region, to_dnf


#CSGNodes

#Iterate through each surface, find boolean derivative

def iter_halfspaces(region: R) -> Iterable[Halfspace]:
    """Yield every Halfspace literal appearing in the region AST."""
    halfspaces = [region]
    while halfspaces:
        n = halfspaces.pop()
        if isinstance(n, Lit):
            yield n.hs
        elif isinstance(n, And) or isinstance(n, Or):
            halfspaces.append(n.a)
            halfspaces.append(n.b)
        elif isinstance(n, Not):
            halfspaces.append(n.r)

def iter_surfaces(region: R) -> Iterable[Surface]:
    seen: Set[int] = set()
    for hs in iter_halfspaces(region):
        sid = id(hs.surface)
        if sid not in seen:
            seen.add(sid)
            yield hs.surface

NodeOrBool = Union[R, bool]

def _not(x: NodeOrBool) -> NodeOrBool:
    if x is True: return False
    if x is False: return True
    return Not(x)

def _and(a: NodeOrBool, b: NodeOrBool) -> NodeOrBool:
    if a is False or b is False: return False
    if a is True: return b
    if b is True: return a
    if a == b: return a
    return And(a, b)

def _or(a: NodeOrBool, b: NodeOrBool) -> NodeOrBool:
    if a is True or b is True: return True
    if a is False: return b
    if b is False: return a
    if a == b: return a
    return Or(a, b)


def substitute_surface(region: R, surface: Surface, x_value: bool) -> NodeOrBool:
    """
    Replace the boolean variable for `surface` with x_value and constant-fold.
    """
    if isinstance(region, Lit):
        hs = region.hs
        if hs.surface is surface:
            # literal is x if sense<0 else ¬x
            lit_val = x_value if hs.sense < 0 else (not x_value)
            return lit_val
        return region

    if isinstance(region, Not):
        return _not(substitute_surface(region.r, surface, x_value))

    if isinstance(region, And):
        return _and(
            substitute_surface(region.a, surface, x_value),
            substitute_surface(region.b, surface, x_value),
        )

    if isinstance(region, Or):
        return _or(
            substitute_surface(region.a, surface, x_value),
            substitute_surface(region.b, surface, x_value),
        )

    raise TypeError(f"Unknown node type: {type(region)}")

def _xor(a: NodeOrBool, b: NodeOrBool) -> NodeOrBool:
    # XOR = (a & ~b) | (~a & b)
    return _or(_and(a, _not(b)), _and(_not(a), b))


def boolean_derivative(region: R, surface: Surface) -> NodeOrBool:
    f0 = substitute_surface(region, surface, False)
    f1 = substitute_surface(region, surface, True)
    return _xor(f0, f1)

#If the surface is false everywhere (never represents switch) -> can immediately classify as ambiguity surface
def is_derivative_unsat(deriv: NodeOrBool) -> bool:
    if deriv is False:
        return True
    if deriv is True:
        return False
    dnf = to_dnf(deriv)  # List[List[Halfspace]]
    return (len(dnf) == 0)

#Else, run the octree algorithm to split into search regions to identify ambiguity surfaces

def _corners(lo, hi):
    # generate all 8 corners of a box
    xs = (lo[0], hi[0]); ys = (lo[1], hi[1]); zs = (lo[2], hi[2])
    return [(x,y,z) for x in xs for y in ys for z in zs]


def intersects_box(surface: Surface, lo, hi) -> bool:
    # check if surface intersects box by evaluating signed distance at corners
    vals = [surface.signed_distance(c) for c in _corners(lo, hi)]
    mn, mx = min(vals), max(vals)
    if mn <= 0.0 <= mx:
        return True
    return False

import random

def _rand_in_box(lo, hi):
    # generate random point in box
    return (
        random.uniform(lo[0], hi[0]),
        random.uniform(lo[1], hi[1]),
        random.uniform(lo[2], hi[2]),
    )

def find_point_on_surface_in_box(surface: Surface, lo, hi, *, tries=40, iters=100) -> Optional[Tuple[float,float,float]]:
    # Try corners first
    cs = _corners(lo, hi)
    vs = [surface.signed_distance(c) for c in cs]
    for c, v in zip(cs, vs):
        if abs(v) <= 1e-8:
            return c
    # Try edges via random pairs
    for _ in range(tries):
        a = _rand_in_box(lo, hi)
        b = _rand_in_box(lo, hi)
        fa = surface.signed_distance(a)
        fb = surface.signed_distance(b)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            continue  # same sign, no root on segment

        # bisection on segment a->b
        pa, pb = a, b
        fpa, fpb = fa, fb
        for i in range(iters):
            mid = ((pa[0]+pb[0])*0.5, (pa[1]+pb[1])*0.5, (pa[2]+pb[2])*0.5)
            fm = surface.signed_distance(mid)
            if abs(fm) <= 1e-8:
                return mid
            if fpa * fm <= 0.0:
                pb, fpb = mid, fm
            else:
                pa, fpa = mid, fm
        return ((pa[0]+pb[0])*0.5, (pa[1]+pb[1])*0.5, (pa[2]+pb[2])*0.5)

    return None


#if there are any regions of the plane where it doesn't affect the cell the point is in
# then it is an ambiguity surface

def is_ambiguous_at_point(region: R, surface: Surface, p, *, eps: float) -> bool:
    n = surface.normal_at(p)
    p_plus  = add(p, mul(n, eps))
    p_minus = sub(p, mul(n, eps))

    in_plus  = point_in_region(p_plus, region)
    in_minus = point_in_region(p_minus, region)

    # DtB-relevant ambiguity:
    return (in_plus and in_minus) or (not in_plus and not in_minus)

@dataclass
class OctNode:
    lo: Tuple[float,float,float]
    hi: Tuple[float,float,float]
    depth: int


def _subdivide(node: OctNode) -> List[OctNode]:
    lo, hi = node.lo, node.hi
    mx = (lo[0]+hi[0])*0.5
    my = (lo[1]+hi[1])*0.5
    mz = (lo[2]+hi[2])*0.5

    xs = [(lo[0], mx), (mx, hi[0])]
    ys = [(lo[1], my), (my, hi[1])]
    zs = [(lo[2], mz), (mz, hi[2])]

    out = []
    for (x0,x1) in xs:
        for (y0,y1) in ys:
            for (z0,z1) in zs:
                out.append(OctNode((x0,y0,z0), (x1,y1,z1), node.depth+1))
    return out


def surface_is_geometrically_ambiguous(region: R, surface: Surface, bbox, *, max_depth=8) -> bool:
    root = OctNode(bbox[0], bbox[1], 0)

    # choose eps relative to bbox size
    diag = norm(sub(bbox[1], bbox[0]))
    eps = 1e-6

    stack = [root]
    while stack:
        node = stack.pop()

        if not intersects_box(surface, node.lo, node.hi):
            continue

        p = find_point_on_surface_in_box(surface, node.lo, node.hi)
        if p is not None and is_ambiguous_at_point(region, surface, p, eps=eps):
            return True

        if node.depth < max_depth:
            stack.extend(_subdivide(node))

    return False


#Return ambiguity surfaces

def find_ambiguity_surfaces(region: R, bbox, *, max_depth=8) -> Set[Surface]:
    amb: Set[Surface] = set()

    for s in iter_surfaces(region):
        # Step 1: boolean derivative early exit
        if is_derivative_unsat(region, s):
            amb.add(s)
            continue

        # Step 2: geometric octree search
        if surface_is_geometrically_ambiguous(region, s, bbox, max_depth=max_depth):
            amb.add(s)

    amb_ids = {id(s) for s in amb}

    return amb_ids


#Modified DtB algorithm that skips checks to ambiguity surfaces

def distance_to_boundary(self, o, d, skip_surface_ids):

    best_t, best_i = None, -1
    for i, h in enumerate(self.halfspaces):
        sid = getattr(h.surface, "id", None)  # if you store an OpenMC id here
        if sid is None:
            sid = id(h.surface)

        if sid in skip_surface_ids:
            continue

        t = h.distance_along(o, d)
        if t is None:
            continue

        if best_t is None or t < best_t:
            best_t, best_i = t, i

    return (best_t, best_i) if best_t is not None else None
