from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any, Set
from vecmath import add, mul, TOL, norm, sub
from surfaces import Surface, OpenMCWrapper
from collections import defaultdict

Vec3 = Tuple[float, float, float]
NormalFn = Callable[[float], Vec3]
Hit = Tuple[float, bool, NormalFn]

TOL = 1e-9
_START_NUDGE = TOL*10

def hit_point(o, d, t):
    ''' find point along the ray'''
    return add(o, mul(d, t))

def start_probe(o, d):
    '''samples just ahead of ray origin
    in case the ray starts on a surface'''
    return hit_point(o, d, _START_NUDGE)

def first_positive_t(hits):
    ''' finds first boundary intersection beyond the origin '''
    for t, n in hits:
        if t > TOL:
            return t, n
    return None

def entering(surface, o, d, t):
    tb = max(t - TOL*10, 0.0)
    ta = t + TOL*10
    fb = surface.signed_distance(hit_point(o, d, tb))
    fa = surface.signed_distance(hit_point(o, d, ta))
    return (fb > 0.0 and fa <= 0.0)


@dataclass(frozen=True)
class Halfspace:
    surface: Surface
    sense: int

    def contains(self, p: Vec3) -> bool:
        f = self.surface.signed_distance(p)
        if self.sense < 0:
            return (f<= 0.0)
        else:
            return (-f <= 0.0)

    def distance_along(self, o: Vec3, d: Vec3) -> Optional[float]:
        ts = [t for t in self.surface.intersect_ray(o, d)]
        if ts:
            return ts[0]
        else:
            return None

# build syntax tree
class R:
    def __and__(self, other):
        return And(self, other)
    def __or__(self, other):
        return Or(self, other)
    def __invert__(self):
        return Not(self)
    def __sub__(self, other):
        return And(self, Not(other))

@dataclass(frozen=True)
class Lit(R):
    hs: Halfspace

@dataclass(frozen=True)
class And(R):
    a: R; b: R

@dataclass(frozen=True)
class Or(R):
    a: R; b: R

@dataclass(frozen=True)
class Not(R):
    r: R

def AND(nodes):
    it = iter(nodes); out = next(it)
    for n in it: out = And(out, n)
    return out

def OR(nodes):
    it = iter(nodes)
    out = next(it)
    for n in it: out = Or(out, n)
    return out

def DIFF(a, b):
    return And(a, Not(b))


Clause = List[Halfspace]
DNF = List[Clause]

def push_not(r):
    if isinstance(r, Not):
        x = r.r
        if isinstance(x, Lit):
            hs = x.hs
            return Lit(Halfspace(hs.surface, -hs.sense))
        if isinstance(x, And):
            # ¬(a ∧ b) = (¬a) ∨ (¬b)
            return Or(push_not(Not(x.a)), push_not(Not(x.b)))
        if isinstance(x, Or):
            # ¬(a ∨ b) = (¬a) ∧ (¬b)
            return And(push_not(Not(x.a)), push_not(Not(x.b)))
        if isinstance(x, Not):
            # ¬(¬r) = r
            return push_not(x.r)

    if isinstance(r, And):
        return And(push_not(r.a), push_not(r.b))
    if isinstance(r, Or):
        return Or(push_not(r.a), push_not(r.b))
    return r

def to_dnf(r: R) -> DNF:
    r = push_not(r)

    def go(node: R) -> DNF:
        if isinstance(node, Lit):
            return [[node.hs]]
        if isinstance(node, And):
            L = go(node.a); R = go(node.b)
            out: DNF = []
            for c1 in L:
                for c2 in R:
                    out.append(c1 + c2)
            return out
        if isinstance(node, Or):
            return go(node.a) + go(node.b)

    d = go(r)

    simplify = []
    for clause in d:
        unique: Dict[Tuple[int,int], Halfspace] = {}
        keep = True
        for hs in clause:
            # detect contradictions where both pos + neg sense are included
            key = (id(hs.surface), hs.sense)
            opp = (id(hs.surface), -hs.sense)
            if opp in unique:
                keep = False
                break
            unique[key] = hs
        if keep:
            simplify.append(list(unique.values()))
    return simplify

@dataclass
class Cell:
    halfspaces: List[Halfspace]
    neighbors: Dict[int, Optional["Cell"]]  # index -> neighbor cell (None = void)

    def contains(self, p):
        return all(h.contains(p) for h in self.halfspaces)

    def distance_to_boundary(self, o, d):
        best_t, best_i = None, -1
        for i, h in enumerate(self.halfspaces):
            t = h.distance_along(o, d)
            if t is None:
                continue
            if best_t is None or t < best_t:
                best_t, best_i = t, i
        return (best_t, best_i) if best_t is not None else None

    def compile_region_to_cells(r: R) -> List[Cell]:
        dnf = to_dnf(r)
        cells: List[Cell] = []
        for clause in dnf:
            sortclause = sorted(clause, key=lambda hs: id(hs.surface, hs.sense))
            cells.append(Cell(halfspaces=sortclause, neighbors={}))
        return cells

    #neighbor map
    def build_neighbor_map(cells: List[Cell]) -> None:
        # create signature for cells to index with
        for cell in cells:
            cell.halfspaces = sorted(cell.halfspaces, key=lambda hs: (id(hs.surface), hs.sense))

        sigs = [tuple((id(hs.surface), hs.sense) for hs in c.halfspaces) for c in cells]

        buckets = defaultdict(list)
        for ci, cell in enumerate(cells):
            for i, hs in enumerate(cell.halfspaces):
                # create reduced lists with ith index removed
                # so neighbors across the ith face share this signature
                reduced_pairs = sigs[ci][:i] + sigs[ci][i+1:]
                reduced_sig = tuple(sorted(reduced_pairs))
                key = (reduced_sig, id(hs.surface))
                buckets[key].append((ci, i, hs.sense))

        # sort senses by positive/negative
        for _key, entries in buckets.items():
            pos, neg = [], []
            for ci, i, sense in entries:
                if sense > 0:
                    pos.append(ci, i)
                else:
                    neg.append(ci, i)
            # neighbors have the same face with opposite sense
            for (c_pos, i_pos) in pos:
                for (c_neg, i_neg) in neg:
                    cells[c_pos].neighbors[i_pos] = cells[c_neg]
                    cells[c_neg].neighbors[i_neg] = cells[c_pos]

        # any face not assigned neighbors is void
        for cell in cells:
            for i in range(len(cell.halfspaces)):
                cell.neighbors.setdefault(i, None)

# Point containment

@dataclass
class EvalResult:
    inside: bool
    slack: float
    faces: list

class _SlackCache:
    """Cache f_s(p) per surface-id to avoid repeated computation"""
    def __init__(self, p):
        self.p = p
        self._c: Dict[int, float] = {}

    def f(self, surface) -> float:
        sid = id(surface)
        if sid in self._c:
            return self._c[sid]
        val = surface.signed_distance(self.p)
        self._c[sid] = val
        return val

def _push_not_to_leaves(r: "R") -> "R":
    """Safe NOT-pushing so NOT only wraps literals and leaves the boolean structure"""
    if isinstance(r, Not):
        x = r.r
        if isinstance(x, Lit):
            hs = x.hs
            return Lit(Halfspace(hs.surface, -hs.sense))
        if isinstance(x, And):
            return Or(_push_not_to_leaves(Not(x.a)), _push_not_to_leaves(Not(x.b)))
        if isinstance(x, Or):
            return And(_push_not_to_leaves(Not(x.a)), _push_not_to_leaves(Not(x.b)))
        if isinstance(x, Not):
            return _push_not_to_leaves(x.r)
    if isinstance(r, And):
        return And(_push_not_to_leaves(r.a), _push_not_to_leaves(r.b))
    if isinstance(r, Or):
        return Or(_push_not_to_leaves(r.a), _push_not_to_leaves(r.b))
    return r  # Lit

def _eval_lit(hs: "Halfspace", sc: _SlackCache) -> EvalResult:
    base = sc.f(hs.surface)
    g = base if hs.sense < 0 else -base  # inside iff g <= 0
    inside = (g <= TOL)
    faces = []
    if inside and abs(g) <= TOL:
        faces.append((id(hs.surface), hs.sense, abs(g)))
    return EvalResult(inside=inside, slack=g, faces=faces)

def _eval_ast(node: "R", sc: _SlackCache) -> EvalResult:
    if isinstance(node, Lit):
        return _eval_lit(node.hs, sc)

    if isinstance(node, And):
        L = _eval_ast(node.a, sc)
        Rr = _eval_ast(node.b, sc)
        inside = L.inside and Rr.inside
        slack = max(L.slack, Rr.slack)
        faces = (L.faces + Rr.faces) if inside else []
        return EvalResult(inside, slack, faces)

    if isinstance(node, Or):
        L = _eval_ast(node.a, sc)
        if L.inside:
            return L  # short-circuit
        Rr = _eval_ast(node.b, sc)
        inside = Rr.inside
        slack = min(L.slack, Rr.slack)
        faces = Rr.faces if inside else []
        return EvalResult(inside, slack, faces)

    if isinstance(node, Not):
        # normalize, then eval (Not should only wrap Lit after push)
        return _eval_ast(_push_not_to_leaves(node), sc)

    raise TypeError(f"Unknown region node: {type(node)}")

def classify_point_in_region(p: Vec3, region: "R") -> Tuple[str, EvalResult]:
    """
    Returns ("INSIDE" | "ON_BOUNDARY" | "OUTSIDE", EvalResult).
    """
    region_n = _push_not_to_leaves(region)
    sc = _SlackCache(p)
    res = _eval_ast(region_n, sc)
    if not res.inside:
        return "OUTSIDE", res
    if res.faces:
        return "ON_BOUNDARY", res
    return "INSIDE", res

def point_in_region(p: Vec3, region: "R") -> bool:
    s, _ = classify_point_in_region(p, region)
    return s != "OUTSIDE"


# Boundary detection

def first_boundary(
    o: Vec3,
    d: Vec3,
    start_cell: Optional[Cell],
    *,
    max_steps: int = 512,
    start_nudge: float = 10.0 * TOL,
) -> Optional[Tuple[float, Vec3, Optional[Cell], Optional[Cell], int]]:
    """
    "next boundary" search along +d:
      - Only the current cell's faces are considered
      - Step to the nearest face, nudge past, hop to neighbor
      - Return the first boundary encountered
    Returns:
      (t_from_origin, normal_at_hit, from_cell, to_cell, face_index)
    or None if no boundary is hit.
    """
    d = norm(d)
    current = start_cell
    pos = o

    for i in range(max_steps):
        hit = current.distance_to_boundary(pos, d)
        if hit is None:
            # No intersection with any face of this cell along +d
            return None

        t, i_face = hit
        if t <= TOL:
            # We're too close to a face; push forward slightly and try again
            pos = add(pos, mul(d, max(10.0 * TOL, start_nudge)))
            continue

        # Compute the t from original origin, hit point, and normal
        p_hit = hit_point(pos, d, t)
        # Convert to param from the *original* origin
        distance = norm(sub(p_hit, o))
        n = current.halfspaces[i_face].surface.normal_at(p_hit)

        # Determine neighbor across face
        nbr = current.neighbors.get(i_face, None)

        # Report this boundary
        return (distance, n, current, nbr, i_face)
