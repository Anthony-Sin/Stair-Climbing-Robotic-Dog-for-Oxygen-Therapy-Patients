import logging
import cv2
import numpy as np
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.getLogger(__name__).debug(
        "[YoloPoseInference] torch or torchvision not available, using CPU NMS"
    )

KEYPOINT_PAIRS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

YOLO_MAIN = (220, 238, 96)
YOLO_MAIN_DIM = (86, 112, 64)
YOLO_SECONDARY = (70, 79, 84)
YOLO_BADGE_BG = (16, 28, 45)
YOLO_TEXT = (205, 218, 218)

class YoloPoseInference:
    def __init__(self, use_gpu_preprocessing=True, verbose=True):
        """YoloPoseInference handles preprocessing, decoding, and visualization for YOLO pose detection.
        Inference is handled by TRTInference separately.
        
        Args:
            use_gpu_preprocessing: If True, use cv2.cuda for preprocessing (requires OpenCV with CUDA)
        """
        self.verbose = bool(verbose)
        self.use_gpu_preprocessing = use_gpu_preprocessing and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.use_gpu_preprocessing:
            self._log("[YoloPoseInference] GPU preprocessing enabled")
            # Pre-allocate GPU buffers for common operations
            self.gpu_img = None
            self.gpu_resized = None
            self.gpu_padded = None
            self.gpu_rgb = None
        else:
            self._log("[YoloPoseInference] Using CPU preprocessing")
        
        # Check GPU NMS availability
        self.gpu_nms_available = False
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.gpu_nms_available = self._probe_gpu_nms()
                    if self.gpu_nms_available:
                        self._log("[YoloPoseInference] GPU NMS enabled (torchvision)")
                    else:
                        self._log("[YoloPoseInference] Using CPU NMS (torchvision CUDA NMS unavailable)")
                else:
                    self._log("[YoloPoseInference] Using CPU NMS (CUDA not available)")
            except Exception as e:
                self._log(f"[YoloPoseInference] Using CPU NMS (error checking CUDA: {e})")
        else:
            self._log("[YoloPoseInference] Using CPU NMS (torch/torchvision not available)")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _first_line(error):
        return str(error).splitlines()[0] if str(error) else error.__class__.__name__

    def _probe_gpu_nms(self):
        """Probe torchvision CUDA NMS once at startup to avoid runtime stalls."""
        try:
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device='cuda')
            scores = torch.tensor([0.5], device='cuda')
            torchvision.ops.nms(boxes, scores, 0.5)
            return True
        except Exception as e:
            self._log(f"[YoloPoseInference] CUDA NMS probe failed: {self._first_line(e)}")
            return False

    def preprocess(self, image, input_size=(640, 640)):
        """Preprocess image with optional GPU acceleration."""
        if self.use_gpu_preprocessing:
            return self._preprocess_gpu(image, input_size)
        else:
            return self._preprocess_cpu(image, input_size)
    
    def _preprocess_cpu(self, image, input_size=(640, 640)):
        """CPU-based preprocessing (original implementation)."""
        h0, w0 = image.shape[:2]
        r = min(input_size[0]/h0, input_size[1]/w0)
        nh, nw = int(h0 * r), int(w0 * r)
        image_resized = cv2.resize(image, (nw, nh))
        top = (input_size[0] - nh) // 2
        bottom = input_size[0] - nh - top
        left = (input_size[1] - nw) // 2
        right = input_size[1] - nw - left
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
        image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_chw = np.transpose(image_norm, (2, 0, 1))
        return np.expand_dims(image_chw, axis=0).astype(np.float32), r, top, left
    
    def _preprocess_gpu(self, image, input_size=(640, 640)):
        """GPU-accelerated preprocessing using cv2.cuda."""
        h0, w0 = image.shape[:2]
        r = min(input_size[0]/h0, input_size[1]/w0)
        nh, nw = int(h0 * r), int(w0 * r)
        
        top = (input_size[0] - nh) // 2
        bottom = input_size[0] - nh - top
        left = (input_size[1] - nw) // 2
        right = input_size[1] - nw - left
        
        # Upload to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_img, (nw, nh))
        
        # cv2.cuda.copyMakeBorder supports 1- and 4-channel inputs.
        # Convert BGR -> BGRA before padding, then BGRA -> RGB.
        gpu_rgba = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2BGRA)

        # Padding on GPU (RGBA)
        gpu_padded = cv2.cuda.copyMakeBorder(
            gpu_rgba, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114, 0)
        )
        
        # Color conversion on GPU
        gpu_rgb = cv2.cuda.cvtColor(gpu_padded, cv2.COLOR_BGRA2RGB)
        
        # Download to CPU for normalization and transpose
        # (these operations are fast on CPU and TRT expects CPU input)
        image_rgb = gpu_rgb.download()
        
        # Normalize and transpose on CPU (minimal overhead)
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_chw = np.transpose(image_norm, (2, 0, 1))
        
        return np.expand_dims(image_chw, axis=0).astype(np.float32), r, top, left



    def decode_output(self, output, conf_threshold=0.25, iou_threshold=0.5):
        output = output.transpose(0, 2, 1)
        pred = output[0]

        bboxes = pred[:, 0:4]
        scores = pred[:, 4]

        # Extract 51 values for keypoints: x,y,v for 17 keypoints
        keypoints_all = pred[:, 5:56]
        keypoints_all = keypoints_all.reshape(-1, 17, 3)  # (N, 17, 3)

        mask = scores > conf_threshold
        bboxes = bboxes[mask]
        scores = scores[mask]
        keypoints_all = keypoints_all[mask]

        detections = []
        for i in range(len(scores)):
            xc, yc, w, h = bboxes[i]
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            bbox = np.array([x1, y1, x2, y2])

            kpts = keypoints_all[i]
            coords = kpts[:, 0:2]  # x,y
            visibility = kpts[:, 2]  # visibility/confidence

            # Optional: zero out coords where visibility is low
            coords[visibility < 0.5] = 0

            detections.append({'bbox': bbox, 'score': scores[i], 'keypoints': coords, 'visibility': visibility})

        # Apply NMS on bbox + score only
        detections_nms = self.nms_boxes(detections, iou_threshold)
        return detections_nms

    def nms_boxes(self, detections, iou_threshold=0.5):
        """NMS with GPU acceleration if torch is available."""
        if len(detections) == 0:
            return []
        
        if self.gpu_nms_available:
            return self._nms_gpu(detections, iou_threshold)
        else:
            return self._nms_cpu(detections, iou_threshold)
    
    def _nms_gpu(self, detections, iou_threshold=0.5):
        """GPU-accelerated NMS using torchvision."""
        try:
            boxes = np.array([det['bbox'] for det in detections], dtype=np.float32)
            scores = np.array([det['score'] for det in detections], dtype=np.float32)

            # Convert to torch tensors on GPU
            boxes_tensor = torch.as_tensor(boxes, device='cuda', dtype=torch.float32)
            scores_tensor = torch.as_tensor(scores, device='cuda', dtype=torch.float32)

            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
            
            # Convert back to CPU
            keep = keep_indices.cpu().numpy().tolist()
            
            return [detections[idx] for idx in keep]
        except Exception as e:
            print(
                "[YoloPoseInference] GPU NMS failed "
                f"({self._first_line(e)}), falling back to CPU"
            )
            self.gpu_nms_available = False  # Disable for future calls
            return self._nms_cpu(detections, iou_threshold)
    
    def _nms_cpu(self, detections, iou_threshold=0.5):
        """CPU-based NMS (original implementation)."""
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['score'] for det in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[idx] for idx in keep]

    def scale_coords_pad(self, xy, r, pad_left, pad_top, orig_shape):
        xy[:, 0] -= pad_left
        xy[:, 1] -= pad_top
        xy /= r
        xy[:, 0] = np.clip(xy[:, 0], 0, orig_shape[1]-1)
        xy[:, 1] = np.clip(xy[:, 1], 0, orig_shape[0]-1)
        return xy

    def draw_detections(
        self,
        image,
        detections,
        r,
        pad_left,
        pad_top,
        orig_shape,
        tracked_dets=None,
        main_person=None,
        main_annotation=None,
    ):
        img = image.copy()
        height, width = orig_shape
        draw_list = tracked_dets if tracked_dets is not None else detections

        MAIN_CLR = YOLO_MAIN
        MAIN_DIM = YOLO_MAIN_DIM
        SEC_CLR = YOLO_SECONDARY

        for det in draw_list:
            bbox = np.array(det['bbox'], dtype=np.float32)
            bbox[:2] = self.scale_coords_pad(bbox[:2].reshape(1, 2), r, pad_left, pad_top, orig_shape)[0]
            bbox[2:] = self.scale_coords_pad(bbox[2:].reshape(1, 2), r, pad_left, pad_top, orig_shape)[0]
            x1, y1, x2, y2 = bbox.astype(int)
            bw_box = max(1, x2 - x1)
            bh_box = max(1, y2 - y1)

            track_id = det.get("track_id")
            is_main = main_person is not None and track_id == main_person.get("track_id")

            if is_main:
                # --- Tactical target-lock designator ---
                blen = max(14, min(bw_box, bh_box) // 5)

                # Very faint full-box hint
                cv2.rectangle(img, (x1, y1), (x2, y2),
                               (MAIN_CLR[0] // 6, MAIN_CLR[1] // 6, MAIN_CLR[2] // 6), 1)

                # Corner L-brackets
                for bx, by, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(img, (bx, by), (bx + sx * blen, by), MAIN_CLR, 2, cv2.LINE_AA)
                    cv2.line(img, (bx, by), (bx, by + sy * blen), MAIN_CLR, 2, cv2.LINE_AA)

                rail_len = max(18, min(44, bh_box // 4))
                cv2.line(img, (x1 - 4, y1 + bh_box // 2 - rail_len // 2),
                         (x1 - 4, y1 + bh_box // 2 + rail_len // 2), MAIN_CLR, 1, cv2.LINE_AA)
                cv2.line(img, (x2 + 4, y1 + bh_box // 2 - rail_len // 2),
                         (x2 + 4, y1 + bh_box // 2 + rail_len // 2), MAIN_CLR, 1, cv2.LINE_AA)

                # Center crosshair on person bounding box
                pcx, pcy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(img, (pcx - 10, pcy), (pcx + 10, pcy), MAIN_CLR, 1, cv2.LINE_AA)
                cv2.line(img, (pcx, pcy - 10), (pcx, pcy + 10), MAIN_CLR, 1, cv2.LINE_AA)
                cv2.circle(img, (pcx, pcy), 2, MAIN_CLR, -1, cv2.LINE_AA)

                # Top-center lock ring
                cv2.line(img, (pcx - 14, y1 - 6), (pcx + 14, y1 - 6), MAIN_CLR, 1, cv2.LINE_AA)
                cv2.circle(img, (pcx, y1), 4, MAIN_CLR, 1, cv2.LINE_AA)

                # Data badge (below box, or above if near bottom edge)
                score = det.get("score", 0)
                avg_distance = (main_annotation.get("depth_distance_m")
                                if isinstance(main_annotation, dict) else None)
                parts = []
                if track_id is not None:
                    parts.append(f"ID:{track_id}")
                if avg_distance is not None:
                    parts.append(f"{float(avg_distance):.2f}m")
                if score:
                    parts.append(f"{score * 100:.0f}%")
                badge_text = "  ".join(parts) if parts else "TARGET"
                badge_h = 16
                badge_tw = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
                badge_w = badge_tw + 10
                badge_y = y2 + 2
                if badge_y + badge_h > height - 5:
                    badge_y = y1 - badge_h - 2
                badge_x = max(0, min(x1, width - badge_w - 1))
                cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                               MAIN_CLR, -1)
                cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                               YOLO_BADGE_BG, -1)
                cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                               MAIN_CLR, 1)
                cv2.putText(img, badge_text, (badge_x + 5, badge_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, YOLO_TEXT, 1, cv2.LINE_AA)

                # Skeleton (main person only)
                if 'keypoints' in det:
                    kpts_raw = det['keypoints']
                    if kpts_raw is not None:
                        kpts = np.array(kpts_raw, dtype=np.float32)
                        visibility = det.get("visibility", np.ones(len(kpts)))
                        kpts = self.scale_coords_pad(kpts, r, pad_left, pad_top, orig_shape)
                        kpts_int = kpts.astype(int)

                        for i, j in KEYPOINT_PAIRS:
                            if i >= len(kpts_int) or j >= len(kpts_int):
                                continue
                            if visibility[i] < 0.5 or visibility[j] < 0.5:
                                continue
                            pt1, pt2 = tuple(kpts_int[i]), tuple(kpts_int[j])
                            if min(pt1[0], pt2[0]) < 3 or min(pt1[1], pt2[1]) < 3:
                                continue
                            cv2.line(img, pt1, pt2, MAIN_DIM, 1, cv2.LINE_AA)

                        for idx, (kx, ky) in enumerate(kpts_int):
                            if visibility[idx] < 0.5:
                                continue
                            if kx < 3 or ky < 3 or kx >= width - 3 or ky >= height - 3:
                                continue
                            cv2.circle(img, (kx, ky), 3, MAIN_CLR, -1, cv2.LINE_AA)
                            cv2.circle(img, (kx, ky), 3, YOLO_TEXT, 1, cv2.LINE_AA)

            else:
                # --- Secondary detection: small dim corner brackets ---
                blen = max(8, min(bw_box, bh_box) // 7)
                for bx, by, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(img, (bx, by), (bx + sx * blen, by), SEC_CLR, 1, cv2.LINE_AA)
                    cv2.line(img, (bx, by), (bx, by + sy * blen), SEC_CLR, 1, cv2.LINE_AA)
                if track_id is not None:
                    cv2.putText(img, str(track_id), (x1 + 2, y1 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, SEC_CLR, 1, cv2.LINE_AA)

        return img


    def average_person_distance(self, depth_image, keypoints, visibility=None,
                                depth_min=100, depth_max=10000, radius=10):
        """
        Calculate average depth (in meters) around visible keypoints over a circular neighborhood.

        Args:
            depth_image (np.ndarray): 2D array with depth values in millimeters.
            keypoints (np.ndarray): Array of (N,2) keypoint coordinates (x, y).
            visibility (np.ndarray or None): Visibility scores, or None to assume visible.
            depth_min (float): Min valid depth in millimeters (default 100 mm).
            depth_max (float): Max valid depth in millimeters (default 10000 mm).
            radius (int): Radius in pixels to average around keypoints.

        Returns:
            float or None: Average distance in meters, or None if no valid data.
        """
        if visibility is None:
            visibility = np.ones(len(keypoints))

        height, width = depth_image.shape
        all_depth_values = []

        for idx, (x, y) in enumerate(keypoints.astype(int)):
            if visibility[idx] < 0.5:
                continue
            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            x_min = max(x - radius, 0)
            x_max = min(x + radius + 1, width)
            y_min = max(y - radius, 0)
            y_max = min(y + radius + 1, height)

            patch = depth_image[y_min:y_max, x_min:x_max]

            yy, xx = np.ogrid[y_min - y:y_max - y, x_min - x:x_max - x]
            mask = xx**2 + yy**2 <= radius**2

            patch_masked = patch[mask]
            valid_depths = patch_masked[(patch_masked >= depth_min) & (patch_masked <= depth_max)]

            if valid_depths.size > 0:
                all_depth_values.extend(valid_depths.tolist())

        if len(all_depth_values) == 0:
            return None

        avg_depth_mm = np.mean(all_depth_values)
        avg_depth_m = avg_depth_mm / 1000.0  # Convert mm to meters

        return float(avg_depth_m)
