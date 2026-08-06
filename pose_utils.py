"""
Helpers for loading SLEAP prediction files and pulling out per-frame
keypoint coordinates/confidences for a single tracked rat.
"""

from pathlib import Path
import numpy as np


def _fps_from_video_file(video_path):
    """Read fps directly off the video file via cv2. Returns None if cv2
    isn't available, the file doesn't exist, or fps isn't usable."""
    try:
        import cv2
    except ImportError:
        return None
    video_path = Path(video_path)
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return fps if fps and fps > 0 else None


class Track:
    """Per-frame keypoint data for one tracked animal in one video.

    Attributes
    ----------
    fps : float
    n_frames : int
    node_names : list[str]
    points : np.ndarray, shape (n_frames, n_nodes, 2)   (x, y), NaN if missing
    scores : np.ndarray, shape (n_frames, n_nodes)       confidence, NaN if missing
    """

    def __init__(self, fps, node_names, points, scores):
        self.fps = fps
        self.node_names = list(node_names)
        self.points = points
        self.scores = scores
        self.n_frames = points.shape[0]

    def node_index(self, name):
        return self.node_names.index(name)

    def xy(self, name):
        return self.points[:, self.node_index(name), :]

    def conf(self, name):
        return self.scores[:, self.node_index(name)]


def load_track_from_slp(slp_path, track_index=0, video_path=None):
    """Load a single animal's track from a SLEAP predictions .slp file.

    Requires the `sleap-io` package (lightweight reader, no GPU needed):
        pip install sleap-io

    fps resolution order: (1) the .slp's own embedded video.fps; (2)
    reading it off `video_path` (or the .slp's recorded filename) via cv2;
    (3) a hardcoded 30.0 fallback, with a warning -- a wrong fps here
    silently miscalculates every MERGE_GAP_S/MIN_BOUT_DURATION_S threshold
    in scoring.py, so pass video_path explicitly whenever available.
    """
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path))
    video = labels.video
    fps = getattr(video, "fps", None)
    if not fps:
        fallback_video_path = video_path if video_path is not None else getattr(video, "filename", None)
        fps = _fps_from_video_file(fallback_video_path) if fallback_video_path else None
    if not fps:
        print(
            f"WARNING: could not determine real fps for {slp_path} -- falling back to 30.0. "
            f"Pass video_path= explicitly if you have it."
        )
        fps = 30.0

    node_names = [n.name for n in labels.skeletons[0].nodes]
    n_nodes = len(node_names)
    n_frames = video.shape[0] if video.shape is not None else len(labels)

    points = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float64)
    scores = np.full((n_frames, n_nodes), np.nan, dtype=np.float64)

    frames_by_idx = {lf.frame_idx: lf for lf in labels}
    for frame_idx, lf in frames_by_idx.items():
        if frame_idx >= n_frames or not lf.instances:
            continue
        instances = sorted(
            lf.instances,
            key=lambda inst: getattr(inst, "track", None).name
            if getattr(inst, "track", None) is not None
            else "",
        )
        if track_index >= len(instances):
            continue
        inst = instances[track_index]
        for node_i, node in enumerate(node_names):
            # inst[node] returns a structured numpy record (not an object
            # with .x/.y attributes) -- fields are pt["xy"], pt["visible"],
            # pt["score"] (score only present on PredictedInstance rows).
            pt = inst[node]
            if pt is None:
                continue
            xy = pt["xy"]
            x, y = float(xy[0]), float(xy[1])
            if np.isnan(x) or np.isnan(y):
                continue
            points[frame_idx, node_i, 0] = x
            points[frame_idx, node_i, 1] = y
            if pt.dtype.names is not None and "score" in pt.dtype.names:
                score = pt["score"]
                scores[frame_idx, node_i] = float(score) if score is not None and not np.isnan(score) else 0.0
            else:
                scores[frame_idx, node_i] = 1.0

    return Track(fps=fps, node_names=node_names, points=points, scores=scores)
