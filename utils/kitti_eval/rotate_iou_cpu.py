"""CPU rotated-box IoU for the KITTI evaluation.

The original evaluation gets its rotated 2D IoU from a Numba `@cuda.jit`
kernel in `nms_gpu.py`. That kernel builds a CUDA context the moment it is
imported, and on a host whose Numba/CUDA driver does not match it segfaults at
compile time. On a small test run that only shows up at the AP step, where the
BEV / 3D overlap needs the rotated IoU, so the score never gets produced.

This module is a plain NumPy replacement with the same call signature and the
same area convention as `nms_gpu.devRotateIoUEval`:

    boxes  : (N, 5) array of [cx, cy, w, l, angle]
    qboxes : (K, 5) array of [cx, cy, w, l, angle]
    return : (N, K) array, entry [i, k] = IoU(qboxes[k], boxes[i])

    criterion = -1 -> intersection / union            (standard IoU)
    criterion =  0 -> intersection / area(qboxes[k])
    criterion =  1 -> intersection / area(boxes[i])
    else           -> intersection area only

The polygon intersection is the standard Sutherland-Hodgman clip of one
rotated rectangle against the other, then the shoelace area. It matches the
GPU kernel's result to floating-point tolerance and needs no GPU at all.
"""
import numpy as np


def _corners(box):
    cx, cy, w, l, angle = box
    c, s = np.cos(angle), np.sin(angle)
    # half extents along the box axes; w is the x-extent, l the y-extent, to
    # match the area = w * l convention used by the GPU kernel.
    dx, dy = w / 2.0, l / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject, clip_poly):
    # Sutherland-Hodgman polygon clipping. clip_poly must be convex and given
    # counter-clockwise; a rotated rectangle's corners are already in order.
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        d1 = (b[0] - a[0]) * (p1[1] - a[1]) - (b[1] - a[1]) * (p1[0] - a[0])
        d2 = (b[0] - a[0]) * (p2[1] - a[1]) - (b[1] - a[1]) * (p2[0] - a[0])
        denom = d1 - d2
        if abs(denom) < 1e-12:
            return p1
        t = d1 / denom
        return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]

    output = [list(p) for p in subject]
    cp = [list(p) for p in clip_poly]
    n = len(cp)
    for i in range(n):
        a, b = cp[i], cp[(i + 1) % n]
        inp = output
        output = []
        if not inp:
            break
        s = inp[-1]
        for p in inp:
            if inside(p, a, b):
                if not inside(s, a, b):
                    output.append(intersect(s, p, a, b))
                output.append(p)
            elif inside(s, a, b):
                output.append(intersect(s, p, a, b))
            s = p
    return np.array(output) if output else np.zeros((0, 2))


def _ensure_ccw(poly):
    # Sutherland-Hodgman expects the clip polygon counter-clockwise.
    x = poly[:, 0]
    y = poly[:, 1]
    signed = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if signed >= 0 else poly[::-1]


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    out = np.zeros((n, k), dtype=np.float64)
    if n == 0 or k == 0:
        return out
    box_corners = [_ensure_ccw(_corners(b)) for b in boxes]
    q_corners = [_ensure_ccw(_corners(q)) for q in query_boxes]
    box_area = boxes[:, 2] * boxes[:, 3]
    q_area = query_boxes[:, 2] * query_boxes[:, 3]
    for i in range(n):
        bc = box_corners[i]
        for j in range(k):
            inter_poly = _clip(q_corners[j], bc)
            inter_area = _poly_area(inter_poly)
            if inter_area <= 0:
                continue
            if criterion == -1:
                denom = q_area[j] + box_area[i] - inter_area
                out[i, j] = inter_area / denom if denom > 0 else 0.0
            elif criterion == 0:
                out[i, j] = inter_area / q_area[j] if q_area[j] > 0 else 0.0
            elif criterion == 1:
                out[i, j] = inter_area / box_area[i] if box_area[i] > 0 else 0.0
            else:
                out[i, j] = inter_area
    return out
