#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
Exact volume of a union of spheres (the van der Waals cavity) by a
power-diagram decomposition. Support module for xppcm.py.
'''

import warnings
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError
from scipy.optimize import linprog

def union_of_spheres_volume(coords, radii):
    """
    Exact analytical volume of a union of spheres (the van der Waals cavity)
    using a decomposition via power diagrams. Based on (but not exactly
    following) the approach of Cazals et al., Computing the volume of a union
    of balls: A certified algorithm. ACM Trans. Math. Softw. 38, 3:1-3:20 (2011).

    Parameters
    ----------
    coords : array, shape (n, 3)
        Sphere centres (Bohr).
    radii : array, shape (n,)
        Sphere radii (Bohr), all positive.

    Returns
    -------
    float
        Volume in Bohr^3.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if coords.shape != (radii.size, 3):
        raise ValueError(
            f"coords must have shape ({radii.size}, 3) to match radii, got {coords.shape}.")
    if np.any(radii <= 0.0):
        raise ValueError("All radii must be positive.")

    total_volume = 0.0

    for i in range(len(coords)): # For each atom i:
        # Get radical planes between atom i and all other atoms
        rad_planes = _get_radical_planes(coords, radii, i)

        # Add bounding box around atom i to ensure a closed region for HalfspaceIntersection.
        c_i = coords[i]
        r_i = radii[i]
        bbox_planes = np.array([
            [-1.0,  0.0,  0.0, c_i[0] - r_i],
            [ 1.0,  0.0,  0.0, -(c_i[0] + r_i)],
            [ 0.0, -1.0,  0.0, c_i[1] - r_i],
            [ 0.0,  1.0,  0.0, -(c_i[1] + r_i)],
            [ 0.0,  0.0, -1.0, c_i[2] - r_i],
            [ 0.0,  0.0,  1.0, -(c_i[2] + r_i)],
        ])
        if len(rad_planes) > 0:
            all_halfspaces = np.vstack([rad_planes, bbox_planes])
        else:
            all_halfspaces = bbox_planes

        # Find an interior point of C_i using scipy.linprog
        norm_A = np.linalg.norm(all_halfspaces[:, :3], axis=1)
        A_lp = np.hstack([all_halfspaces[:, :3], norm_A[:, None]])
        b_lp = -all_halfspaces[:, 3]
        c_lp = np.array([0, 0, 0, -1])  # Maximise alpha => Minimise -alpha
        bounds_lp = [(None, None), (None, None), (None, None), (None, None)]
        res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=bounds_lp)
        if not res.success or res.x[3] <= 0:
            # Atom i is completely swallowed by others, C_i is empty.
            continue
        interior_point = res.x[:3]

        # Get C_i (the convex polyhedron for atom i)
        try: # Catch Qhull errors if C_i degenerate
            convex_polyhedron = HalfspaceIntersection(all_halfspaces, interior_point)

            # Extract the planar faces of C_i
            planar_faces = _extract_planar_faces(convex_polyhedron)
            c_i = coords[i]
            r_i = radii[i]

            # Compute planar face volume contribution (and solid angle)
            C_i_volume = 0.0
            omega_P_total = 0.0
            for face in planar_faces:
                n = face['normal']
                d = face['d']
                verts = face['vertices']
                h_dist = np.abs(np.dot(n, c_i) + d)
                if h_dist >= r_i:
                    continue # This face is outside the sphere, no contribution

                R_circle = np.sqrt(r_i**2 - h_dist**2) # Radius of the boundary circle for the planar face
                signed_h = np.dot(n, c_i) + d # Positive if center is on the same side as normal, negative otherwise
                p0 = c_i - signed_h * n # Origin for 2d projection

                # Project the vertices of the face to 2D coordinates in the plane of the face
                u, v = _get_orthogonal_basis_for_plane(n)
                verts_2d = np.vstack([np.dot(verts - p0, u), np.dot(verts - p0, v)]).T

                # Compute the area and solid angle of the intersection of the planar face with its boundary circle
                area_P, omega_P = _polygon_circle_intersection_area_and_solid_angle(verts_2d, R_circle, -signed_h, r_i)
                omega_P_total += omega_P

                # Volume contribution for the planar faces (generalised pyramid formula)
                C_i_volume += (-signed_h * area_P) / 3.0

            # Spherical cap contribution
            # Motivation: the total solid angle of C_i is 4pi, so 4pi - omega_P_total
            # is the solid angle of the (total) spherical cap.
            # "Everything not a planar face is a spherical cap"
            # Normalised by norm_A so the tolerance is a real distance.
            signed_dists = (all_halfspaces[:, :3] @ c_i + all_halfspaces[:, 3]) / norm_A
            is_center_inside = np.all(signed_dists <= 1e-8)
            omega_S = (4.0 * np.pi if is_center_inside else 0.0) - omega_P_total
            area_S_exact = (r_i**2) * omega_S
            # Volume contribution from the spherical cap (generalised pyramid formula)
            C_i_volume += (r_i * area_S_exact) / 3.0
            # Total contribution from atom i:
            total_volume += C_i_volume

        except QhullError as e:
            # Almost empty or degenerate cell, skip this atom
            warnings.warn(
                f"HalfspaceIntersection failed for atom {i}; its cell is skipped: {e}",
                RuntimeWarning, stacklevel=2)
            continue

    return total_volume

def _solid_angle_triangle(R1, R2, R3):
    """
    Computes the solid angle subtended by the triangle formed by R1, R2, R3 at the origin.
    From Van Oosterom and Strackee, "The Solid Angle of a Plane Triangle",
    IEEE Transactions on Biomedical Engineering, 1983.
    """
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    r3 = np.linalg.norm(R3)
    num = np.dot(R1, np.cross(R2, R3))
    den = r1*r2*r3 + np.dot(R1, R2)*r3 + np.dot(R2, R3)*r1 + np.dot(R3, R1)*r2
    return 2 * np.arctan2(num, den)

def _segment_circle_intersection_area_and_solid_angle(A, B, R, h, r_i):
    """
    Computes both the area and the exact solid angle of the intersection of
    a triangle (Origin, A, B) and a circle of radius R centered at the origin.
    """
    # Check if A and B are collinear with the origin (degenerate triangle)
    if np.isclose(A[0]*B[1] - A[1]*B[0], 0):
        return 0.0, 0.0

    # Set up the quadratic for line-circle intersection
    D = B - A
    a_coef = np.dot(D, D)
    b_coef = 2 * np.dot(A, D)
    c_coef = np.dot(A, A) - R**2

    # discriminant for intersection
    disc = b_coef**2 - 4*a_coef*c_coef

    # Solve for t values of intersection points, only consider those strictly between A and B (0 < t < 1)
    t_roots = []
    if disc > 0 and not np.isclose(a_coef, 0):
        root = np.sqrt(disc)
        t1 = (-b_coef - root) / (2*a_coef)
        t2 = (-b_coef + root) / (2*a_coef)
        t_roots.extend([t for t in (t1, t2) if 0 < t < 1])
    t_roots = sorted(t_roots)
    pts = [A] + [A + t*D for t in t_roots] + [B] # Points along the segment in order

    area = 0.0
    omega = 0.0
    for i in range(len(pts) - 1): # For each segment between consecutive points
        p1 = pts[i]
        p2 = pts[i+1]

        mid = (p1 + p2) / 2

        if np.linalg.norm(mid) <= R:
            # Segment is inside the circle: triangle area via shoelace formula
            area += 0.5 * (p1[0]*p2[1] - p1[1]*p2[0])
            R1 = np.array([0, 0, h])
            R2 = np.array([p1[0], p1[1], h])
            R3 = np.array([p2[0], p2[1], h])
            omega += _solid_angle_triangle(R1, R2, R3)
        else:
            # Segment is outside the circle: circular sector area
            theta1 = np.arctan2(p1[1], p1[0])
            theta2 = np.arctan2(p2[1], p2[0])
            d_theta = (theta2 - theta1 + np.pi) % (2*np.pi) - np.pi
            area += 0.5 * R**2 * d_theta
            omega += d_theta * (np.sign(h) - h / r_i)

    return area, omega

def _polygon_circle_intersection_area_and_solid_angle(verts_2d, R, h, r_i):
    """
    Finds the exact intersection area and solid angle between an arbitrary
    2D polygon and a circle of radius R centered at the origin.
    """
    area_total = 0.0
    omega_total = 0.0
    num_verts = len(verts_2d)
    for k in range(num_verts): # For each face
        A = verts_2d[k]
        B = verts_2d[(k+1) % num_verts]
        a, o = _segment_circle_intersection_area_and_solid_angle(A, B, R, h, r_i)
        area_total += a
        omega_total += o

    # The vertices run counter-clockwise in the (u, v) frame, so area_total comes
    # out positive. The solid angle carries that same orientation, so should the
    # sign ever flip, both must flip together.
    if area_total < 0.0:
        return -area_total, -omega_total
    return area_total, omega_total

def _extract_planar_faces(convex_polyhedrons, tol=1e-8):
    """
    Extracts the planar faces of the convex polyhedron defined by a HalfspaceIntersection.
    """
    vertices = convex_polyhedrons.intersections # vertices of the polyhedron
    halfspaces = convex_polyhedrons.halfspaces # radical planes + bounding box planes

    faces = []
    seen = [] # Planes already made into a face, so none is counted twice
    for h in halfspaces:
        h_norm = np.linalg.norm(h[:3])
        if h_norm <= 0.0:
            continue # Degenerate halfspace, defines no plane

        normal = h[:3] / h_norm
        d = h[3] / h_norm

        # Check if this plane has already been processed (within tolerance)
        if any(np.abs(normal - n_prev).max() <= tol and abs(d - d_prev) <= tol
               for n_prev, d_prev in seen):
            continue
        seen.append((normal, d))

        # Check distance of all vertices to radical plane
        dists = np.dot(vertices, normal) + d

        # Vertices lying on the plane
        on_plane_idx = np.where(np.abs(dists) <= tol)[0]

        if len(on_plane_idx) >= 3: # At least 3 vertices needed to define a face
            face_verts = vertices[on_plane_idx]

            # Order the vertices in the plane of the face by projecting to 2D and using Convex Hull
            u, v = _get_orthogonal_basis_for_plane(normal)

            # Project vertices
            proj_2d = np.vstack([np.dot(face_verts, u), np.dot(face_verts, v)]).T

            # Sort via 2D Convex Hull
            try:
                hull_2d = ConvexHull(proj_2d)
            except QhullError:
                continue # Degenerate (e.g. collinear) face, contributes no area
            ordered_verts = face_verts[hull_2d.vertices]

            faces.append({
                'normal': normal,
                'd': d,
                'vertices': ordered_verts
            })

    return faces

def _get_radical_planes(coords, radii, i):
    """
    Construct the radical planes for the i-th sphere
    against all other spheres j in a power diagram.

    For two spheres i and j, the power distances to a point x are equal when:
    ||x - c_i||^2 - r_i^2 = ||x - c_j||^2 - r_j^2
    This simplifies to a plane equation: 2(c_j - c_i) * x + (||c_i||^2 - ||c_j||^2 - r_i^2 + r_j^2) = 0
    """
    N = len(coords)
    c_i = coords[i]
    r_i = radii[i]

    rad_planes = []

    for j in range(N):
        if i == j:
            continue

        c_j = coords[j]
        r_j = radii[j]

        # Power Diagram Plane: 2*(c_j - c_i)*x + (||c_i||^2 - ||c_j||^2 - r_i^2 + r_j^2) = 0
        normal = 2.0 * (c_j - c_i)

        c_i_2 = np.dot(c_i, c_i)
        c_j_2 = np.dot(c_j, c_j)

        d = (c_i_2 - c_j_2) - (r_i**2 - r_j**2)

        # Check if spheres are coincident or if the radical plane is degenerate
        if np.linalg.norm(normal) <= 1e-12:
            if (r_j > r_i) or (r_j == r_i and j < i):
                return np.array([[0.0, 0.0, 0.0, 1.0]]) # Infeasible => C_i empty
            continue # Sphere j constrains nothing

        # Format Ax + By + Cz + D <= 0 for HalfspaceIntersection
        rad_planes.append([normal[0], normal[1], normal[2], d])

    return np.array(rad_planes)

def _get_orthogonal_basis_for_plane(normal):
    """
    Given a (orthonormal) normal vector, find two orthogonal vectors that form a basis for the plane.
    """
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    return u, v
