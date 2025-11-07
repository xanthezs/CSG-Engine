from vecmath import dot, sub, add, mul, norm, length, TOL, Ray
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Callable
import openmc

Vec3 = Tuple[float, float, float]
Intsec = Tuple[float, bool, Callable[[float], Vec3]]
class Surface:
    def contains(self, p) -> bool:
        return self.signed_distance(p) <= 0.0 # negative if inside
    def signed_distance(self, p: Vec3) -> float:
        raise NotImplementedError
    def normal_at(self, p: Vec3) -> Vec3:
        raise NotImplementedError
    def intersect_ray(self, o: Vec3, d: Vec3):
        raise NotImplementedError


class OpenMCWrapper(Surface):
    def __init__(self, s):
        self.s = s
    def signed_distance(self, p: Vec3) -> float:
        return float(self.s.evaluate(p))
    def contains(self, p: Vec3) -> bool:
        return self.signed_distance(p) <= 0.0
    def normal_at(self, p: Vec3) -> Vec3:
        h = 1e-6
        x, y, z = p
        f = self.s.evaluate
        fx1, fx0 = f(x+h, y, z), f(x-h, y, z)
        fy1, fy0 = f(x, y+h, z), f(x, y-h, z)
        fz1, fz0 = f(x, y, z+h), f(x, y, z-h)
        nx = (fx1 - fx0)/(2*h)
        ny = (fy1 - fy0)/(2*h)
        nz = (fz1 - fz0)/(2*h)
        L = math.sqrt(nx**2 + ny**2 + nz**2)
        norm = (nx/L, ny/L, nz/L)
        return norm

    def intersect_ray(self, o: Vec3, d: Vec3):
        ''' analytical analysis time taken for ray to reach a given surface for basic primitives
            Primitives: Sphere, XPlane, YPlane, ZPlane, Plane, XCylinder, YCylinder, ZCylinder, XCone, YCone, ZCone, Quadric'''
        if isinstance(self.s, openmc.Sphere):
            return self.hit_sphere(o,d)
        if isinstance(self.s, openmc.XPlane) or isinstance(self.s, openmc.YPlane) or isinstance(self.s, openmc.ZPlane) or isinstance(self.s, openmc.Plane):
            return self.hit_plane(o,d, self.s.a, self.s.b, self.s.c, self.s.d)
        if isinstance(self.s, openmc.XCylinder):
            return self.hit_cylinder(o,d, a1=self.s.y0, a2=self.s.z0, r=self.s.r, plane = 'yz')
        if isinstance(self.s, openmc.YCylinder):
            return self.hit_cylinder(o,d, a1=self.s.x0, a2=self.s.z0, r=self.s.r, plane = 'xz')
        if isinstance(self.s, openmc.ZCylinder):
            return self.hit_cylinder(o,d, a1=self.s.x0, a2=self.s.y0, r=self.s.r, plane = 'xy')
        if isinstance(self.s, openmc.XCone):
            return self.hit_cone(o, d, a1=self.s.y0, a2=self.s.z0, k=math.sqrt(self.s.r2), plane='yz', a3=self.s.x0)
        if isinstance(self.s, openmc.YCone):
            return self.hit_cone(o, d, a1=self.s.x0, a2=self.s.z0, k=math.sqrt(self.s.r2), plane='xz', a3=self.s.y0)
        if isinstance(self.s, openmc.ZCone):
            return self.hit_cone(o, d, a1=self.s.x0, a2=self.s.y0, k=math.sqrt(self.s.r2), plane='xy', a3=self.s.z0)
        if isinstance(self.s, openmc.Quadric):
            return self.hit_quadric(o,d, self.s)

    def hit_sphere(self, o, d):
        cx, cy, cz, r = float(self.s.x0), float(self.s.y0), float(self.s.z0), float(self.s.r)
        ox, oy, oz = o
        oc = ox - cx, oy - cy, oz - cz
        a = dot(d, d)
        b = 2.0 * dot(oc, d)
        c = dot(oc, oc) - r**2

        disc = b**2 - 4.0*a*c

        hits = []
        if disc < 0:
            return hits
        else:
            t0 = (-b - math.sqrt(disc))/(2.0*a)
            t1 = (-b + math.sqrt(disc))/(2.0*a)
        if t0 > 0:
            hits.append(t0)
        if t1 > 0:
            hits.append(t1)
        return hits

    def hit_plane(self, o, d, a, b, c, e):
        np = a, b, c
        denom = dot(np, d)
        numer = e - dot(np, o)
        t = numer / denom
        if t > 0:
            return [t]
        else:
            return []

    def hit_cylinder(self, o, d, a1, a2, r, plane):
        ox, oy, oz = o
        dx, dy, dz = d
        if plane == 'xy':
            u1, u2 = ox, oy
            v1, v2 = dx, dy
        elif plane == 'yz':
            u1, u2 = oy, oz
            v1, v2 = dy, dz
        else:
            u1, u2 = ox, oz
            v1, v2 = dx, dz
        oc1, oc2 = u1 - a1, u2 - a2
        a = v1*v1 + v2*v2
        b = 2.0 * (oc1*v1 + oc2*v2)
        c = oc1*oc1 + oc2*oc2 - r**2
        disc = b*b - 4.0*a*c
        hits = []
        if a == 0.0:
            return hits
        if disc < 0.0:
            return hits
        else:
            t0 = (-b - math.sqrt(disc)) / (2.0*a)
            t1 = (-b + math.sqrt(disc)) / (2.0*a)
        if t0 > 0.0:
            hits.append(t0)
        if t1 > 0.0:
            hits.append(t1)
        return hits


    def hit_cone(self, o, d, a1, a2, k, plane, a3):
        ox, oy, oz = o
        dx, dy, dz = d
        kk = k*k
        if plane == 'xy':
            ux, uy, uz = ox - a1, oy - a2, oz - a3
            a = dx*dx + dy*dy - kk*dz*dz
            b = 2.0*(ux*dx + uy*dy - kk*uz*dz)
            c = ux*ux + uy*uy - kk*uz*uz
        elif plane == 'yz':
            ux, uy, uz = oy - a1, oz - a2, ox - a3
            a = dy*dy + dz*dz - kk*dx*dx
            b = 2.0*(ux*dy + uy*dz - kk*uz*dx)
            c = ux*ux + uy*uy - kk*uz*uz
        else:
            ux, uy, uz = ox - a1, oz - a2, oy - a3
            b = 2.0*(ux*dx + uy*dz - kk*uz*dy)
            c = ux*ux + uy*uy - kk*uz*uz
        disc = b*b - 4.0*a*c
        hits = []
        if a == 0.0:
            return hits
        if disc < 0.0:
            return hits
        t0 = (-b - math.sqrt(disc)) / (2.0*a)
        t1 = (-b + math.sqrt(disc)) / (2.0*a)
        if t0 > 0.0:
            hits.append(t0)
        if t1 > 0.0:
            hits.append(t1)
        return hits

    def hit_quadric(self, o, d, s):
        ox, oy, oz = o
        dx, dy, dz = d
        A, B, C, D, E, F, G, H, I, J = float(s.a), float(s.b), float(s.c), float(s.d), float(s.e), float(s.f), float(s.g), float(s.h), float(s.i), float(s.j)
        a = A*dx*dx + B*dy*dy + C*dz*dz + D*dx*dy + E*dx*dz + F*dy*dz
        b = 2.0*(A*ox*dx + B*oy*dy + C*oz*dz) + D*(ox*dy + oy*dx) + E*(ox*dz + oz*dx) + F*(oy*dz + oz*dy) + G*dx + H*dy + I*dz
        c = A*ox*ox + B*oy*oy + C*oz*oz + D*ox*oy + E*ox*oz + F*oy*oz + G*ox + H*oy + I*oz + J
        disc = b*b - 4.0*a*c
        hits = []
        if a == 0.0:
            return hits
        if disc < 0.0:
            return hits
        t0 = (-b - math.sqrt(disc)) / (2.0*a)
        t1 = (-b + math.sqrt(disc)) / (2.0*a)
        if t0 > 0.0:
            hits.append(t0)
        if t1 > 0.0:
            hits.append(t1)
        return hits


S = OpenMCWrapper(openmc.Plane(1,0,0,1))
print(S.intersect_ray((-1, -1, -1), (0.5, 1, 1.5)))
