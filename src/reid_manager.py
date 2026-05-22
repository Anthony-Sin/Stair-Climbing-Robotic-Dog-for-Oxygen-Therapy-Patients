import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nfc_gpu import nfc
from reid_augment import lgpr_augment


@dataclass
class ReIDConfig:
    gallery_size: int = 50
    update_interval_sec: float = 2.0
    dedupe_cos: float = 0.995
    seed_stable_sec: float = 2.0
    seed_count: int = 5
    lgpr_per_image: int = 2
    match_thresh: float = 0.85
    match_margin: float = 0.20
    nfc_k1: int = 2
    nfc_k2: int = 2
    reacquire_timeout_sec: float = 5.0
    search_pid_sec: float = 2.0
    id_switch_iou_thresh: float = 0.55
    id_switch_expand_ratio: float = 0.15
    reacquire_consistency_frames: int = 3
    reacquire_consistency_required: int = 2
    reacquire_mean_gap_scale: float = 0.8


class ReIDManager:
    IDLE = "IDLE"
    SEED_WAIT_STABLE = "SEED_WAIT_STABLE"
    SEED_BURST = "SEED_BURST"
    TRACKING_READY = "TRACKING_READY"
    REACQUIRE = "REACQUIRE"

    def __init__(self, inference, config: ReIDConfig):
        self.inference = inference
        self.cfg = config

        self.state = self.IDLE
        self.gallery: List[np.ndarray] = []
        self.last_accepted_feature: Optional[np.ndarray] = None

        self.expected_track_id: Optional[int] = None
        self.stable_track_id: Optional[int] = None
        self.stable_since: Optional[float] = None
        self.last_gallery_update_ts: Optional[float] = None
        self.last_expected_bbox: Optional[Tuple[float, float, float, float]] = None

        self.seed_original_crops: List[np.ndarray] = []

        self.reacquire_start_ts: Optional[float] = None
        self.reacquire_reason: Optional[str] = None
        self.reacquire_history: List[Tuple[int, float, float]] = []

    def _normalize_rows(self, feats: np.ndarray) -> np.ndarray:
        feats = np.asarray(feats, dtype=np.float32)
        if feats.size == 0:
            return feats
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return feats / norms

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _extract_crop(self, frame_bgr: np.ndarray, bbox: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if bbox is None or len(bbox) != 4:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        xi1 = int(np.clip(np.floor(x1), 0, w - 1))
        yi1 = int(np.clip(np.floor(y1), 0, h - 1))
        xi2 = int(np.clip(np.ceil(x2), 0, w))
        yi2 = int(np.clip(np.ceil(y2), 0, h))
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        crop = frame_bgr[yi1:yi2, xi1:xi2]
        if crop.size == 0:
            return None
        return crop

    def _add_feature(self, feat: np.ndarray) -> bool:
        if feat.ndim != 1:
            raise ValueError(f"Expected 1D feature, got shape={feat.shape}")

        norm = np.linalg.norm(feat)
        if norm <= 1e-12:
            return False
        feat_norm = (feat / norm).astype(np.float32, copy=False)

        if self.last_accepted_feature is not None:
            cos = self._cosine(feat_norm, self.last_accepted_feature)
            if cos >= self.cfg.dedupe_cos:
                return False

        self.gallery.append(feat_norm)
        self.last_accepted_feature = feat_norm

        if len(self.gallery) > self.cfg.gallery_size:
            self.gallery.pop(0)

        return True

    def _add_features(self, feats: np.ndarray) -> int:
        accepted = 0
        for i in range(feats.shape[0]):
            if self._add_feature(feats[i]):
                accepted += 1
        return accepted

    def _collect_candidate_crops(
        self,
        frame_bgr: np.ndarray,
        tracked_dets: Sequence[Dict],
    ) -> Tuple[List[int], List[np.ndarray]]:
        track_ids: List[int] = []
        crops: List[np.ndarray] = []

        for det in tracked_dets:
            track_id = det.get("track_id")
            bbox = det.get("bbox")
            if track_id is None or bbox is None:
                continue

            crop = self._extract_crop(frame_bgr, bbox)
            if crop is None:
                continue

            track_ids.append(int(track_id))
            crops.append(crop)

        return track_ids, crops

    def _enter_seed_wait(self, main_track_id: Optional[int], now: float):
        self.state = self.SEED_WAIT_STABLE
        self.stable_track_id = int(main_track_id) if main_track_id is not None else None
        self.stable_since = now if main_track_id is not None else None
        self.seed_original_crops = []
        self.last_expected_bbox = None

    def _enter_reacquire(self, reason: str, now: float):
        self.state = self.REACQUIRE
        self.reacquire_reason = reason
        self.reacquire_start_ts = now
        self.reacquire_history = []

    def _update_reacquire_history(self, track_id: int, best_score: float, score_gap: float):
        self.reacquire_history.append((int(track_id), float(best_score), float(score_gap)))
        max_frames = max(1, int(self.cfg.reacquire_consistency_frames))
        if len(self.reacquire_history) > max_frames:
            self.reacquire_history = self.reacquire_history[-max_frames:]

    def _select_consensus_candidate(self) -> Optional[int]:
        if not self.reacquire_history:
            return None

        required = max(1, int(self.cfg.reacquire_consistency_required))
        wins: Dict[int, int] = {}
        for tid, _, _ in self.reacquire_history:
            wins[tid] = wins.get(tid, 0) + 1
        best_track_id, vote_count = max(wins.items(), key=lambda kv: kv[1])
        if vote_count < required:
            return None

        selected_scores = [s for tid, s, _ in self.reacquire_history if tid == best_track_id]
        selected_gaps = [g for tid, _, g in self.reacquire_history if tid == best_track_id]
        if not selected_scores or not selected_gaps:
            return None

        mean_score = float(np.mean(selected_scores))
        mean_gap = float(np.mean(selected_gaps))
        gap_thresh = float(self.cfg.match_margin) * max(0.0, float(self.cfg.reacquire_mean_gap_scale))

        if mean_score >= float(self.cfg.match_thresh) and mean_gap >= gap_thresh:
            return int(best_track_id)
        return None

    @staticmethod
    def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(float(ax1), float(bx1))
        iy1 = max(float(ay1), float(by1))
        ix2 = min(float(ax2), float(bx2))
        iy2 = min(float(ay2), float(by2))
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, float(ax2) - float(ax1)) * max(0.0, float(ay2) - float(ay1))
        area_b = max(0.0, float(bx2) - float(bx1)) * max(0.0, float(by2) - float(by1))
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _is_tolerable_id_switch(self, current_bbox: Sequence[float]) -> bool:
        if self.last_expected_bbox is None:
            return False

        prev = self.last_expected_bbox
        cur = tuple(float(v) for v in current_bbox)
        if len(cur) != 4:
            return False

        iou = self._bbox_iou(prev, cur)
        if iou >= self.cfg.id_switch_iou_thresh:
            return True

        px1, py1, px2, py2 = prev
        bw = max(1.0, px2 - px1)
        bh = max(1.0, py2 - py1)
        mx = bw * max(0.0, self.cfg.id_switch_expand_ratio)
        my = bh * max(0.0, self.cfg.id_switch_expand_ratio)
        ex1, ey1, ex2, ey2 = px1 - mx, py1 - my, px2 + mx, py2 + my

        cx = 0.5 * (cur[0] + cur[2])
        cy = 0.5 * (cur[1] + cur[3])
        return ex1 <= cx <= ex2 and ey1 <= cy <= ey2

    def _default_result(self) -> Dict:
        return {
            "state": self.state,
            "reacquire_active": self.state == self.REACQUIRE,
            "use_frozen_pid": False,
            "recovered_track_id": None,
            "timeout_exit": False,
            "reacquire_reason": self.reacquire_reason,
        }

    def update(
        self,
        frame_bgr: np.ndarray,
        tracked_dets: Sequence[Dict],
        main_person: Optional[Dict],
        main_track_id: Optional[int],
    ) -> Dict:
        now = time.perf_counter()
        result = self._default_result()

        if self.state == self.IDLE:
            self._enter_seed_wait(main_track_id, now)

        if self.state == self.SEED_WAIT_STABLE:
            if main_person is None or main_track_id is None:
                self.stable_track_id = None
                self.stable_since = None
                self.seed_original_crops = []
            else:
                if self.stable_track_id != int(main_track_id):
                    self.stable_track_id = int(main_track_id)
                    self.stable_since = now
                    self.seed_original_crops = []
                elif self.stable_since is not None and (now - self.stable_since) >= self.cfg.seed_stable_sec:
                    self.state = self.SEED_BURST
                    self.seed_original_crops = []

        if self.state == self.SEED_BURST:
            if main_person is None or main_track_id is None:
                self._enter_seed_wait(main_track_id, now)
            elif self.stable_track_id != int(main_track_id):
                self._enter_seed_wait(main_track_id, now)
            else:
                bbox = main_person.get("bbox") if isinstance(main_person, dict) else None
                crop = self._extract_crop(frame_bgr, bbox) if bbox is not None else None
                if crop is not None and len(self.seed_original_crops) < self.cfg.seed_count:
                    self.seed_original_crops.append(crop)

                if len(self.seed_original_crops) >= self.cfg.seed_count:
                    seed_batch: List[np.ndarray] = []
                    for base in self.seed_original_crops:
                        seed_batch.append(base)
                        seed_batch.extend(
                            lgpr_augment(base, num_variants=self.cfg.lgpr_per_image, patches_per_variant=1)
                        )

                    embeddings = self.inference.infer_crops(seed_batch)
                    embeddings = self._normalize_rows(embeddings)
                    accepted = self._add_features(embeddings)

                    if accepted > 0:
                        self.expected_track_id = int(main_track_id)
                        self.last_expected_bbox = tuple(float(v) for v in bbox)
                        self.last_gallery_update_ts = now
                        self.state = self.TRACKING_READY
                    else:
                        self._enter_seed_wait(main_track_id, now)

        if self.state == self.TRACKING_READY:
            if main_person is None or main_track_id is None:
                self._enter_reacquire("lost", now)
            elif self.expected_track_id is not None and int(main_track_id) != self.expected_track_id:
                bbox = main_person.get("bbox") if isinstance(main_person, dict) else None
                if bbox is not None and self._is_tolerable_id_switch(bbox):
                    # Accept tracker ID churn when the box remains in the same local region.
                    self.expected_track_id = int(main_track_id)
                    self.last_expected_bbox = tuple(float(v) for v in bbox)
                else:
                    self._enter_reacquire("id_switch", now)
            else:
                self.expected_track_id = int(main_track_id)
                bbox = main_person.get("bbox") if isinstance(main_person, dict) else None
                if bbox is not None:
                    self.last_expected_bbox = tuple(float(v) for v in bbox)
                should_update = (
                    self.last_gallery_update_ts is None
                    or (now - self.last_gallery_update_ts) >= self.cfg.update_interval_sec
                )
                if should_update and isinstance(main_person, dict) and main_person.get("bbox") is not None:
                    crop = self._extract_crop(frame_bgr, main_person["bbox"])
                    if crop is not None:
                        feat = self.inference.infer_crops([crop])
                        if feat.shape[0] > 0:
                            feat = self._normalize_rows(feat)
                            self._add_feature(feat[0])
                            self.last_gallery_update_ts = now

        if self.state == self.REACQUIRE:
            result["reacquire_active"] = True
            elapsed = (now - self.reacquire_start_ts) if self.reacquire_start_ts is not None else 0.0
            result["use_frozen_pid"] = elapsed <= self.cfg.search_pid_sec

            if elapsed >= self.cfg.reacquire_timeout_sec:
                result["timeout_exit"] = True
                result["state"] = self.state
                return result

            if not self.gallery:
                result["state"] = self.state
                return result

            candidate_track_ids, candidate_crops = self._collect_candidate_crops(frame_bgr, tracked_dets)
            if not candidate_crops:
                result["state"] = self.state
                return result

            cand_embeddings = self.inference.infer_crops(candidate_crops)
            if cand_embeddings.shape[0] == 0:
                result["state"] = self.state
                return result

            cand_embeddings = self._normalize_rows(cand_embeddings)
            cand_embeddings = nfc(
                cand_embeddings,
                k1=self.cfg.nfc_k1,
                k2=self.cfg.nfc_k2,
                device="cuda",
            )
            cand_embeddings = self._normalize_rows(np.asarray(cand_embeddings, dtype=np.float32))

            gallery = np.stack(self.gallery, axis=0).astype(np.float32)
            sims = cand_embeddings @ gallery.T
            candidate_scores = sims.max(axis=1)

            order = np.argsort(candidate_scores)[::-1]
            best_idx = int(order[0])
            best_score = float(candidate_scores[best_idx])
            second_best = float(candidate_scores[int(order[1])]) if len(order) > 1 else -1.0
            best_track_id = int(candidate_track_ids[best_idx])
            score_gap = best_score - second_best

            self._update_reacquire_history(best_track_id, best_score, score_gap)
            recovered_track_id = self._select_consensus_candidate()
            if recovered_track_id is not None:
                result["recovered_track_id"] = recovered_track_id

                self.expected_track_id = recovered_track_id
                recovered_indices = [i for i, tid in enumerate(candidate_track_ids) if int(tid) == recovered_track_id]
                if recovered_indices:
                    recovered_best_idx = max(recovered_indices, key=lambda i: float(candidate_scores[i]))
                    self._add_feature(cand_embeddings[recovered_best_idx])
                self.last_gallery_update_ts = now

                self.state = self.TRACKING_READY
                self.reacquire_start_ts = None
                self.reacquire_reason = None
                self.reacquire_history = []
                result["reacquire_active"] = False
                result["use_frozen_pid"] = False

        result["state"] = self.state
        result["reacquire_reason"] = self.reacquire_reason
        return result
