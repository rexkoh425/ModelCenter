import numpy as np


def pointcloud_to_bev(points: np.ndarray, extent, bev_size, max_height=3.0, min_height=-3.0):
  """
  Very simple BEV rasterizer: accumulates density, mean height, max height, min height, and intensity.
  points: Nx4 [x,y,z,intensity] in ego frame (meters).
  extent: [x_min, x_max, y_min, y_max] in meters.
  bev_size: (H, W)
  """
  x_min, x_max, y_min, y_max = extent
  H, W = bev_size
  mask = (
    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
    (points[:, 2] >= min_height) & (points[:, 2] <= max_height)
  )
  pts = points[mask]
  if pts.shape[0] == 0:
    return np.zeros((5, H, W), dtype=np.float32)

  # Map to pixel indices
  xs = (pts[:, 0] - x_min) / (x_max - x_min) * (W - 1)
  ys = (pts[:, 1] - y_min) / (y_max - y_min) * (H - 1)
  xs = xs.astype(np.int32)
  ys = ys.astype(np.int32)

  bev = np.zeros((5, H, W), dtype=np.float32)
  for x, y, z, intensity in zip(xs, ys, pts[:, 2], pts[:, 3]):
    bev[0, y, x] += 1.0              # density
    bev[1, y, x] += z                # sum height
    bev[2, y, x] = max(bev[2, y, x], z)  # max height
    bev[3, y, x] = min(bev[3, y, x], z) if bev[0, y, x] > 1 else z  # min height
    bev[4, y, x] += intensity        # sum intensity

  # Normalize sum height and intensity by density where density>0
  nonzero = bev[0] > 0
  bev[1][nonzero] /= bev[0][nonzero]
  bev[4][nonzero] /= bev[0][nonzero]
  return bev
