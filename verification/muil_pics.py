import cv2
from pathlib import Path
import numpy as np

latest_run_file = Path('log/latest_run.txt')
if latest_run_file.exists():
    run = Path(latest_run_file.read_text(encoding='utf-8-sig').strip())
else:
    runs = sorted(Path('log').glob('run_sim_*'))
    run = runs[-1] if runs else Path(r'log\run_sim_20260618_142522_480')
video_dir = run / 'videos'
out_dir = run / 'reports'
out_dir.mkdir(parents=True, exist_ok=True)
summary = []
thumbs = []
for name in ('scene_view.mp4', 'topdown.mp4'):
    path = video_dir / name
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    indices = sorted(set([0, max(0, frame_count // 2), max(0, frame_count - 1)]))
    means = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        means.append(float(frame.mean()))
        thumb = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
        label = f'{name} frame {idx}/{max(0, frame_count - 1)}'
        cv2.rectangle(thumb, (0, 0), (383, 24), (0, 0, 0), -1)
        cv2.putText(thumb, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    cap.release()
    summary.append((name, frame_count, round(fps, 3), width, height, [round(m, 2) for m in means]))
if thumbs:
    rows = []
    for i in range(0, len(thumbs), 3):
        row = thumbs[i:i+3]
        while len(row) < 3:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_dir / 'cinematic_camera_contact_sheet.jpg'), sheet)
for item in summary:
    print(item)
