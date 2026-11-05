"""Procedural material textures: generation, caching and texture-array packing."""
from __future__ import annotations

import logging
import time

import numpy as np

from emberdeep.core.types import Material

log = logging.getLogger(__name__)
N_VARIANTS = 4
N_MATERIALS = len(Material)
TEX_VERSION = 11


def build_texture_arrays(res: int = 128, cache: bool = True, parallel: bool = True):
    """Return (albedo [L,res,res,4] u8, normal [L,res,res,4] u8) with L = N_MATERIALS * N_VARIANTS.

    Layer index = material * N_VARIANTS + variant. Cached as .npz in ``assets.cache_dir()``.
    """
    from emberdeep.render.textures.materials import generate_one
    path = None
    if cache:
        try:
            from emberdeep.render.assets import cache_dir
            path = cache_dir() / f"textures_v{TEX_VERSION}_{res}.npz"
            if path.exists():
                z = np.load(path)
                return z["alb"], z["nrm"]
        except Exception as e:  # noqa: BLE001
            log.warning("texture cache unreadable: %s", e)
    t0 = time.time()
    jobs = [(m, v, res) for m in range(N_MATERIALS) for v in range(N_VARIANTS)]
    results = None
    if parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor
            import os
            with ProcessPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as ex:
                results = list(ex.map(generate_one, jobs, chunksize=4))
        except Exception as e:  # noqa: BLE001
            log.warning("parallel texture generation failed (%s); falling back to serial", e)
    if results is None:
        results = [generate_one(j) for j in jobs]
    alb = np.stack([r[0] for r in results])
    nrm = np.stack([r[1] for r in results])
    log.info("generated %d material layers in %.1fs", len(jobs), time.time() - t0)
    if path is not None:
        try:
            np.savez(path, alb=alb, nrm=nrm)
            from emberdeep.render.assets import prune_cache
            prune_cache("textures_v", path.name) if False else None
            import re as _re
            for old in path.parent.glob("textures_v*_%d.npz" % res):
                if old.name != path.name:
                    old.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("could not write texture cache: %s", e)
    return alb, nrm
