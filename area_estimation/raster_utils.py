"""Raster helpers: census cross-tabulation of class rasters and map class at sample points."""
from __future__ import annotations

import numpy as np
import rasterio


def read_class_raster(path):
    """Class raster as a uint8 array, with its transform, CRS and pixel area (m2)."""
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs, abs(src.transform.a * src.transform.e)


def crosstab(tif_a, tif_b=None):
    """Census pixel counts C[a, b] of two uint8 rasters on one grid (C[a, 0] if tif_b is None)."""
    c = np.zeros((256, 256), np.int64)
    with rasterio.open(tif_a) as a:
        b = rasterio.open(tif_b) if tif_b else None
        try:
            if b and (b.transform != a.transform or b.shape != a.shape):
                raise ValueError("the two rasters are not on the same grid")
            for _, win in a.block_windows(1):
                x = a.read(1, window=win).astype(np.int64)
                y = b.read(1, window=win).astype(np.int64) if b else np.zeros_like(x)
                c += np.bincount((x * 256 + y).ravel(), minlength=65536).reshape(256, 256)
        finally:
            if b:
                b.close()
    return c


def class_at_points(class_tif, points, nodata=255):
    """Map class of the pixel containing each point (GeoDataFrame, reprojected to the raster CRS).

    A pixel contains its left and top edges but not its right and bottom edges.  Points outside
    the raster get `nodata`.
    """
    with rasterio.open(class_tif) as src:
        cls = src.read(1)
        pts = points.to_crs(src.crs)
        T = src.transform
    if T.b != 0 or T.d != 0:
        raise ValueError("rotated rasters are not supported")
    cols = np.floor((pts.geometry.x.to_numpy() - T.c) / T.a).astype(int)
    rows = np.floor((pts.geometry.y.to_numpy() - T.f) / T.e).astype(int)
    inside = (rows >= 0) & (rows < cls.shape[0]) & (cols >= 0) & (cols < cls.shape[1])
    out = np.full(len(pts), nodata, dtype=np.int32)
    out[inside] = cls[rows[inside], cols[inside]]
    return out
