import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace

default_cfg = {
    'track_thresh': 0.5,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'frame_rate': 30,
    'mot20': False
}

class SinglePersonTracker:
    def __init__(self, tracker_cfg=None, max_lost_frames=300, allow_auto_reacquire=True, debug=False):
        cfg = tracker_cfg or default_cfg
        cfg = SimpleNamespace(**cfg)
        self.byte_tracker = BYTETracker(cfg)
        self.main_track_id = None
        self.lost_counter = 0
        self.max_lost_frames = max_lost_frames
        self.allow_auto_reacquire = allow_auto_reacquire
        self.debug = debug
        self._tracking_initialized = False
        self._manual_reacquire_requested = False
        self.main_person_lost = False

    def set_main_track_id(self, track_id, reason='reid'):
        """Externally force the current main track ID."""
        if track_id is None:
            self.clear_main_track(reason=reason)
            return

        self.main_track_id = int(track_id)
        self.lost_counter = 0
        self._tracking_initialized = True
        self._manual_reacquire_requested = False
        self.main_person_lost = False
        if self.debug:
            print(f"[Tracker] Main ID set externally to {self.main_track_id} ({reason})")

    def clear_main_track(self, reason='lost'):
        """Externally clear the main track selection."""
        self.main_track_id = None
        self.lost_counter = 0
        self._manual_reacquire_requested = False
        self.main_person_lost = reason == 'lost'
        if self.debug:
            print(f"[Tracker] Main ID cleared ({reason})")

    def _select_main_person(self, tracked_dets, image_shape):
        h, w = image_shape[:2]
        cx = w / 2
        best_score = float('inf')
        best_id = None

        for det in tracked_dets:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            center_x = (x1 + x2) / 2
            center_dist = abs(center_x - cx)
            score = -area + center_dist  # prefer large center-aligned box

            if score < best_score:
                best_score = score
                best_id = det['track_id']

        if best_id is not None:
            if self.main_track_id is None or self.lost_counter > self.max_lost_frames:
                self.main_track_id = best_id
                self.lost_counter = 0
                self._tracking_initialized = True
                self._manual_reacquire_requested = False
                self.main_person_lost = False
                if self.debug:
                    print(f"[Tracker] Selected main person: ID {best_id}")

    def reset(self):
        """Clear tracking state and allow re-selection of the next person."""
        self.main_track_id = None
        self.lost_counter = 0
        self._tracking_initialized = False
        self._manual_reacquire_requested = False
        self.main_person_lost = False

    def request_manual_reacquire(self):
        """Allow a manual re-acquisition even when auto-reacquire is disabled."""
        self._manual_reacquire_requested = True

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _map_tracks_to_detections(self, active_stracks, detections):
        """Greedy IoU matching from active track boxes to current detections."""
        mapping = {}
        if not active_stracks or not detections:
            return mapping

        pairs = []
        for ti, track in enumerate(active_stracks):
            tbox = track.tlbr.tolist()
            for di, det in enumerate(detections):
                iou = self._bbox_iou(tbox, det['bbox'])
                if iou > 0.0:
                    pairs.append((iou, ti, di))

        used_tracks = set()
        used_dets = set()
        for _, ti, di in sorted(pairs, key=lambda x: x[0], reverse=True):
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            mapping[ti] = di

        return mapping

    def update(self, detections, image_shape):
        """
        Args:
            detections: [{'bbox': [...], 'keypoints': [...], 'score': ..., 'visibility': [...]}]
            image_shape: (height, width)
        Returns:
            tracked_dets: list of active detections with tracking info
            main_person: main person dict when the main track is active, otherwise None
        """
        if not detections:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                if self.debug and not self.main_person_lost:
                    print("[Tracker] Main person lost. Clearing ID.")
                self.main_person_lost = True
                self.main_track_id = None
            return [], None

        dets_np = np.array([
            det['bbox'].tolist() + [det['score']] for det in detections
        ], dtype=np.float32)

        img_h, img_w = image_shape[:2]
        stracks = self.byte_tracker.update(dets_np, (img_h, img_w), (img_h, img_w))

        active_stracks = [s for s in stracks if s.is_activated]
        matched = self._map_tracks_to_detections(active_stracks, detections)

        tracked_dets = []
        for ti, s in enumerate(active_stracks):
            det = detections[matched[ti]] if ti in matched else None
            det_score = float(det['score']) if det is not None else float(getattr(s, 'score', 0.0))

            is_main = s.track_id == self.main_track_id
            if is_main and det is not None:
                keypoints = det.get('keypoints')
                visibility = det.get('visibility')
                if visibility is None and keypoints is not None:
                    visibility = np.ones(len(keypoints), dtype=np.float32)
            else:
                keypoints = None
                visibility = None

            tracked_dets.append({
                'track_id': int(s.track_id),
                'bbox': s.tlbr.tolist(),
                'score': det_score,
                'keypoints': keypoints,
                'visibility': visibility,
                'matched_detection': det is not None,
            })

        should_select = (
            self.main_track_id is None and
            (
                not self._tracking_initialized or
                self.allow_auto_reacquire or
                self._manual_reacquire_requested
            )
        )
        if should_select:
            self._select_main_person(tracked_dets, image_shape)

        main_person = None
        for det in tracked_dets:
            if det['track_id'] == self.main_track_id:
                self.lost_counter = 0
                main_person = det
                break
        if main_person is None:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                if self.debug and not self.main_person_lost:
                    print("[Tracker] Main person lost. Clearing ID.")
                self.main_person_lost = True
                self.main_track_id = None

        return tracked_dets, main_person
