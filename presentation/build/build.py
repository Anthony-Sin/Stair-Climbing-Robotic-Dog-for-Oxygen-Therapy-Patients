#!/usr/bin/env python3
"""Build a PowerPoint (.pptx) mirroring the TASH pitch website.

Reproduces the website's slides, two-column sheet design, palette and type,
the SVG diagrams / charts (as crisp high-DPI images), the render photos, the
demo VIDEOS (embedded, playable), and the robot as a native PowerPoint 3D
model (injected post-build via build_3d.py).
"""
import json, os, re, html
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(BASE)
ASSETS = os.path.join(BASE, "assets")
POSTERS = os.path.join(BASE, "posters")
CLIPS = "/home/user/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/src/tools/blueprint_viewer/assets/clips"
os.makedirs(POSTERS, exist_ok=True)
content = json.load(open(os.path.join(SP, "content.json")))
CONTENT = content["CONTENT"]; POLICIES = content["POLICIES"]

# ---- palette (from js/palette.js light theme) ----
BACKDROP = RGBColor(0x1b,0x1b,0x18)
SHEET    = RGBColor(0xd6,0xd2,0xca)
SHEET_HI = RGBColor(0xe3,0xe0,0xd9)
BORDER   = RGBColor(0xb1,0xac,0xa1)
INK      = RGBColor(0x2f,0x2c,0x28)
HEAD     = RGBColor(0x26,0x23,0x20)
GREY     = RGBColor(0x6f,0x6a,0x62)
OAK      = RGBColor(0xc0,0x7d,0x3c)
TEAL     = RGBColor(0x2e,0x8f,0x9e)
RED      = RGBColor(0xb8,0x50,0x50)
DARKCARD = RGBColor(0x22,0x20,0x1d)
PAPER    = RGBColor(0xef,0xec,0xe6)
DISP = "Segoe UI"      # display face (present on the target Windows machine)
MONO = "Consolas"      # mono face

EMU_IN = 914400
SW, SH = 13.333, 7.5

def RG(x): return RGBColor(x>>16 & 255, x>>8 & 255, x & 255)

# ---------------- helpers ----------------
def strip_tags(s): return re.sub(r"<[^>]+>", "", s or "")

def runs_from_html(s):
    """Return [(text, bold)] parsing only <b>..</b>."""
    s = s or ""
    parts = re.split(r"(</?b>)", s)
    out, bold = [], False
    for p in parts:
        if p == "<b>": bold = True
        elif p == "</b>": bold = False
        elif p:
            out.append((html.unescape(p), bold))
    return out or [("", False)]

def solid(shape, color):
    shape.fill.solid(); shape.fill.fore_color.rgb = color
def noline(shape):
    shape.line.fill.background()
def line(shape, color, w=0.75):
    shape.line.color.rgb = color; shape.line.width = Pt(w)
def noshadow(shape):
    try: shape.shadow.inherit = False
    except Exception: pass

def rrect(slide, l,t,w,h, fill=None, brd=None, bw=0.75, radius=0.08):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l),Inches(t),Inches(w),Inches(h))
    noshadow(sp)
    try: sp.adjustments[0] = radius
    except Exception: pass
    if fill is None: sp.fill.background()
    else: solid(sp, fill)
    if brd is None: noline(sp)
    else: line(sp, brd, bw)
    return sp

def rect(slide, l,t,w,h, fill=None, brd=None, bw=0.75):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l),Inches(t),Inches(w),Inches(h))
    noshadow(sp)
    if fill is None: sp.fill.background()
    else: solid(sp, fill)
    if brd is None: noline(sp)
    else: line(sp, brd, bw)
    return sp

def para(tf, runs, size, color, bold=False, font=DISP, align=PP_ALIGN.LEFT,
         space_before=0, space_after=4, leading=1.08, upper=False, tracking=None, new=True):
    p = tf.add_paragraph() if new else tf.paragraphs[0]
    p.alignment = align
    p.space_before = Pt(space_before); p.space_after = Pt(space_after)
    try: p.line_spacing = leading
    except Exception: pass
    if isinstance(runs, str): runs = [(runs, bold)]
    for txt, b in runs:
        r = p.add_run(); r.text = txt.upper() if upper else txt
        f = r.font; f.size = Pt(size); f.name = font; f.bold = b or bold
        f.color.rgb = color
    return p

def fit_box(iw, ih, bw, bh):
    """contain: return (w,h) fitting image (iw,ih) inside box (bw,bh)."""
    s = min(bw/iw, bh/ih)
    return iw*s, ih*s

def place_image(slide, path, l,t,bw,bh, align="center", name=None):
    if not os.path.exists(path):
        rrect(slide, l,t,bw,bh, fill=DARKCARD, brd=BORDER, radius=0.04); return
    iw, ih = Image.open(path).size
    w, h = fit_box(iw, ih, bw, bh)
    ox = l + (bw-w)/2; oy = t + (bh-h)/2
    pic = slide.shapes.add_picture(path, Inches(ox), Inches(oy), Inches(w), Inches(h))
    if name: pic.name = name
    return pic

def make_poster(name, caption, w=1280, h=720, accent=OAK):
    """Branded video poster (dark card + caption + play glyph)."""
    p = os.path.join(POSTERS, name+".png")
    img = Image.new("RGB", (w,h), (0x1e,0x1c,0x19))
    d = ImageDraw.Draw(img)
    # subtle border
    d.rectangle([4,4,w-5,h-5], outline=(0x3a,0x37,0x31), width=3)
    # play triangle in a ring
    cx, cy, r = w//2, h//2-30, 62
    d.ellipse([cx-r,cy-r,cx+r,cy+r], outline=(accent[0],accent[1],accent[2]), width=6)
    tri = [(cx-20,cy-30),(cx-20,cy+30),(cx+34,cy)]
    d.polygon(tri, fill=(accent[0],accent[1],accent[2]))
    try: fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30)
    except Exception: fnt = ImageFont.load_default()
    tw = d.textlength(caption, font=fnt)
    d.text(((w-tw)/2, cy+r+40), caption, font=fnt, fill=(0xcf,0xca,0xc0))
    img.save(p); return p

def add_video(slide, clip, l,t,w,h, caption="", accent=OAK):
    path = os.path.join(CLIPS, clip)
    poster = make_poster(clip.replace(".mp4",""), caption, accent=accent)
    if not os.path.exists(path):
        place_image(slide, poster, l,t,w,h); return
    # frame
    rrect(slide, l-0.03,t-0.03,w+0.06,h+0.06, fill=DARKCARD, brd=BORDER, bw=1, radius=0.04)
    mv = slide.shapes.add_movie(path, Inches(l),Inches(t),Inches(w),Inches(h),
                                poster_frame_image=poster, mime_type="video/mp4")
    return mv

def eyebrow_title(tf, eyebrow, title):
    para(tf, eyebrow, 10.5, GREY, font=MONO, upper=True, tracking=1, space_after=8, new=False)
    para(tf, title, 25, HEAD, bold=True, leading=1.06, space_after=10)

def copy_block(slide, l, t, w, h, group, title, lead=None, body=None):
    tb = slide.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    eyebrow_title(tf, group, title)
    if lead: para(tf, runs_from_html(lead), 12.5, INK, bold=True, leading=1.18, space_after=8)
    if body: para(tf, runs_from_html(body), 10.8, INK, leading=1.28, space_after=4)
    return tb

def stat_cards(slide, l, t, w, stats):
    n = len(stats); gap = 0.16
    cw = (w - gap*(n-1))/n; ch = 0.92
    for i, s in enumerate(stats):
        x = l + i*(cw+gap)
        rrect(slide, x, t, cw, ch, fill=None, brd=BORDER, bw=1, radius=0.06)
        tb = slide.shapes.add_textbox(Inches(x+0.12), Inches(t+0.10), Inches(cw-0.24), Inches(ch-0.18))
        tf = tb.text_frame; tf.word_wrap = True
        para(tf, s.get("v",""), 16, HEAD, bold=True, space_after=2, new=False)
        para(tf, s.get("l",""), 8, GREY, font=MONO, leading=1.05)
    return t+ch

def tag_row(slide, l, t, w, tags):
    gap = 0.12
    widths = [0.075*len(tg) + 0.30 for tg in tags]
    total = sum(widths) + gap*(len(tags)-1)
    scale = min(1.0, w/total) if total > w else 1.0   # shrink to fit if needed
    widths = [x*scale for x in widths]; gg = gap*scale
    fs = 9*scale
    x = l
    for tg, tw in zip(tags, widths):
        rrect(slide, x, t, tw, 0.34, fill=None, brd=BORDER, bw=1, radius=0.5)
        tb = slide.shapes.add_textbox(Inches(x), Inches(t+0.035), Inches(tw), Inches(0.27))
        tf = tb.text_frame; tf.word_wrap=False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        para(tf, tg, fs, INK, font=MONO, align=PP_ALIGN.CENTER, new=False)
        x += tw + gg

# ---------------- deck skeleton ----------------
prs = Presentation()
prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]

M = 0.32                     # outer margin
SL, ST = M, M
SWD, SHT = SW-2*M, SH-2*M     # sheet size
PAD = 0.30
VIS_FRAC = 0.575

def new_slide(sheet=True, split=True):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0,0,SW,SH, fill=BACKDROP)
    if sheet:
        rrect(s, SL, ST, SWD, SHT, fill=SHEET, brd=BORDER, bw=1, radius=0.035)
    if sheet and split:
        divx = SL + SWD*VIS_FRAC
        ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(divx), Inches(ST+0.2), Inches(0.014), Inches(SHT-0.4))
        noshadow(ln); solid(ln, BORDER); noline(ln)
    return s

def vis_box():
    return (SL+PAD, ST+PAD, SWD*VIS_FRAC-1.6*PAD, SHT-2*PAD)
def copy_box():
    x = SL + SWD*VIS_FRAC + 0.7*PAD
    return (x, ST+PAD, SL+SWD-PAD - x, SHT-2*PAD)

def content_slide(cid, visual=None, video=None, video_cap="", extra=None):
    c = CONTENT[cid]; cp = c["copy"]
    s = new_slide()
    vx,vy,vw,vh = vis_box()
    if visual: place_image(s, os.path.join(ASSETS, visual), vx,vy,vw,vh)
    if video: add_video(s, video, vx+vw*0.12, vy+vh*0.18, vw*0.76, vh*0.64, video_cap)
    if extra: extra(s, (vx,vy,vw,vh))
    cx,cy,cw,ch = copy_box()
    copy_block(s, cx, cy, cw, ch, c["group"], c["title"], cp.get("lead"), cp.get("body"))
    # stats + tags anchored near bottom
    by = ST+SHT-PAD-0.34
    if cp.get("tags"): tag_row(s, cx, by, cw, cp["tags"]); by -= 0.5
    if cp.get("stats"): stat_cards(s, cx, by-0.92, cw, cp["stats"])
    return s

# ============ COVER ============
def cover():
    s = prs.slides.add_slide(BLANK)
    rect(s, 0,0,SW,SH, fill=BACKDROP)
    place_image(s, os.path.join(ASSETS,"hero_robot.png"), 7.2,0.5,5.7,6.5)
    tb = s.shapes.add_textbox(Inches(0.8),Inches(1.7),Inches(6.6),Inches(4))
    tf = tb.text_frame; tf.word_wrap=True
    para(tf, "STAIR-CLIMBING ROBOTIC OXYGEN CARRIER", 12, OAK, font=MONO, upper=True, space_after=14, new=False)
    para(tf, "TASH", 60, PAPER, bold=True, space_after=6)
    para(tf, [("A robot dog that ", False),("carries the oxygen and climbs the stairs", True),(" alongside therapy patients — so they don’t have to.", False)], 16, RG(0xcfc9bf), leading=1.3, space_before=6)
    tb2 = s.shapes.add_textbox(Inches(0.8),Inches(6.4),Inches(9),Inches(0.5))
    para(tb2.text_frame, "Unitree Go2 · 2.22 kg O₂ payload · blind-RL climb · Isaac Sim ⇄ Jetson Orin", 11, RG(0x9a958c), font=MONO, new=False)

# ============ build all ============
cover()
# 1 Problem
content_slide("s-problem", visual="problem_photos.png")
# 2 Architecture (native 3D robot injected later; show robot render + diagram now)
def arch_extra(s, box):
    vx,vy,vw,vh = box
    place_image(s, os.path.join(ASSETS,"diagram_arch.png"), vx+0.1, vy+vh*0.66, vw*0.95, vh*0.32)
a = new_slide()
vx,vy,vw,vh = vis_box()
place_image(a, os.path.join(ASSETS,"hero_robot.png"), vx, vy, vw, vh*0.64, name="ARCH_ROBOT_RENDER")
place_image(a, os.path.join(ASSETS,"diagram_arch.png"), vx+vw*0.03, vy+vh*0.66, vw*0.94, vh*0.32)
cx,cy,cw,ch = copy_box()
pol = POLICIES["architecture"]
copy_block(a, cx,cy,cw,ch, "Solution & technology", pol["title"], None, pol["body"])
# 3 Walking (video)
w = new_slide()
vx,vy,vw,vh = vis_box()
add_video(w, "walk.mp4", vx, vy+vh*0.06, vw, vh*0.52, "Isaac Sim — flat-ground follow gait")
place_image(w, os.path.join(ASSETS,"diagram_walk.png"), vx+vw*0.05, vy+vh*0.64, vw*0.9, vh*0.32)
cx,cy,cw,ch = copy_box()
pol = POLICIES["walking"]
copy_block(w, cx,cy,cw,ch, "Solution & technology", pol["title"], None, pol["body"])
# 4 Climbing (video)
cl = new_slide()
vx,vy,vw,vh = vis_box()
add_video(cl, "climb_0p130_trimmed.mp4", vx, vy+vh*0.06, vw, vh*0.52, "Isaac Sim — 0.15 m riser climb")
place_image(cl, os.path.join(ASSETS,"diagram_climb.png"), vx+vw*0.08, vy+vh*0.64, vw*0.84, vh*0.32)
cx,cy,cw,ch = copy_box()
pol = POLICIES["blind-rl"]
copy_block(cl, cx,cy,cw,ch, "Solution & technology", pol["title"], None, pol["body"])
# 5 Live demo (grid of videos)
d = new_slide(split=False)
tb = d.shapes.add_textbox(Inches(SL+PAD),Inches(ST+0.12),Inches(SWD-2*PAD),Inches(0.7))
tf=tb.text_frame
para(tf,"LIVE DEMO · REAL ISAAC ROLLOUT",10.5,GREY,font=MONO,upper=True,space_after=3,new=False)
para(tf,"It sees the patient and follows — and climbs rising stair heights",21,HEAD,bold=True)
# main HUD video left
add_video(d, "opencv_follow.mp4", SL+PAD, ST+1.1, 6.3, 3.6, "Target-acquisition HUD · YOLO + depth + LiDAR", accent=TEAL)
tb2=d.shapes.add_textbox(Inches(SL+PAD),Inches(ST+4.75),Inches(6.3),Inches(1.4)); tf2=tb2.text_frame; tf2.word_wrap=True
para(tf2, runs_from_html("<b>Nothing here is scripted.</b> YOLO-World locks onto the patient in every frame; depth + LiDAR read the scene; the learned gait steers to keep pace — all from the dog’s own camera."),11,INK,leading=1.25,new=False)
# right: 2x3 grid of climb clips
clips=[("climb_0p120.mp4","0.12 m · 4.7″"),("climb_0p130.mp4","0.13 m · 5.1″"),("climb_0p140.mp4","0.14 m · 5.5″"),
       ("climb_0p150.mp4","0.15 m · 6″ ADA"),("climb_0p160.mp4","0.16 m · over ADA"),("climb_0p175.mp4","0.175 m · partial")]
gx0=SL+PAD+6.7; gw=(SWD-2*PAD-6.9)/3; gh=1.7; ggap=0.18
for i,(cvid,lab) in enumerate(clips):
    r,ccol=divmod(i,3)
    x=gx0+ccol*(gw+ggap); y=ST+1.1+r*(gh+0.55)
    add_video(d, cvid, x, y, gw, gh, lab)
# 6 Training
content_slide("s-training", visual="charts_training.png")
# 7 Sweep (charts + fall video)
sw = new_slide()
vx,vy,vw,vh = vis_box()
place_image(sw, os.path.join(ASSETS,"charts_sweep_only.png"), vx, vy+vh*0.02, vw, vh*0.44)
add_video(sw, "fall.mp4", vx+vw*0.11, vy+vh*0.53, vw*0.78, vh*0.42, "Past code-legal risers — rears up, can’t get over", accent=RED)
cx,cy,cw,ch = copy_box()
c=CONTENT["s-sweep"]; cp=c["copy"]
copy_block(sw, cx,cy,cw,ch, c["group"], c["title"], cp.get("lead"), cp.get("body"))
by=ST+SHT-PAD-0.34
tag_row(sw,cx,by,cw,cp["tags"]); by-=0.5
stat_cards(sw,cx,by-0.92,cw,cp["stats"])
# 8 Challenges (topple scene + fault cards)
ch_s = new_slide()
vx,vy,vw,vh = vis_box()
place_image(ch_s, os.path.join(ASSETS,"scene_topple.png"), vx,vy,vw,vh)
cx,cy,cw,chh = copy_box()
c=CONTENT["s-challenges"]; cp=c["copy"]
tb=ch_s.shapes.add_textbox(Inches(cx),Inches(cy),Inches(cw),Inches(1.4)); tf=tb.text_frame; tf.word_wrap=True
eyebrow_title(tf, c["group"], c["title"])
para(tf, runs_from_html(cp["lead"]), 11.5, INK, bold=True, leading=1.2)
fy = cy+1.7
for fa in cp["faults"]:
    fh=0.86
    rrect(ch_s, cx, fy, cw, fh, fill=None, brd=BORDER, bw=1, radius=0.05)
    tbf=ch_s.shapes.add_textbox(Inches(cx+0.14),Inches(fy+0.06),Inches(cw-0.28),Inches(fh-0.10)); tff=tbf.text_frame; tff.word_wrap=True
    para(tff, [("×  ",False)]+runs_from_html(fa["p"]), 9.5, RED, bold=True, leading=1.12, space_after=2, new=False)
    para(tff, [("→  ",False)]+runs_from_html(fa["f"]), 9.5, INK, leading=1.12)
    fy += fh+0.12
tag_row(ch_s, cx, ST+SHT-PAD-0.34, cw, cp["tags"])
# 9 Potential
content_slide("s-potential", visual="scene_potential.png")

out = os.path.join(BASE, "TASH_pitch.pptx")
prs.save(out)
print("saved", out, os.path.getsize(out), "bytes,", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
