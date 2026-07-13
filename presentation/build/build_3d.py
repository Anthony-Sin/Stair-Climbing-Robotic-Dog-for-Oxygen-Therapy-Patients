#!/usr/bin/env python3
"""Inject a native PowerPoint 3D model (robot.glb) onto the Architecture slide.

Uses the Office 2017 model3D extension wrapped in <mc:AlternateContent>:
  - <mc:Choice Requires="am3d"> holds the native 3D model (rotatable in PowerPoint)
  - <mc:Fallback> holds a normal picture (the robot render) so the file ALWAYS
    opens and shows the robot even in readers without 3D support.
Produces TASH_pitch_3D.pptx and verifies python-pptx can reopen it.
"""
import shutil, zipfile, os, re, sys
from pptx import Presentation

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "TASH_pitch.pptx")
DST = os.path.join(BASE, "TASH_pitch_3D.pptx")
GLB = "/home/user/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients/src/tools/blueprint_viewer/models/robot.glb"
ROBOT_IMG = os.path.join(BASE, "assets", "hero_robot.png")

shutil.copy(SRC, DST)
z = zipfile.ZipFile(SRC)
names = z.namelist()

# locate the Architecture slide xml
arch = None
for n in names:
    if re.match(r"ppt/slides/slide\d+\.xml$", n):
        xml = z.read(n).decode("utf-8")
        if "System architecture" in xml:
            arch = n; arch_xml = xml; break
assert arch, "architecture slide not found"
slide_no = re.search(r"slide(\d+)\.xml", arch).group(1)
rels_name = f"ppt/slides/_rels/slide{slide_no}.xml.rels"
rels_xml = z.read(rels_name).decode("utf-8")
ct_xml = z.read("[Content_Types].xml").decode("utf-8")
z.close()

# --- content types: register glb + ensure png default present ---
if "gltf-binary" not in ct_xml:
    ct_xml = ct_xml.replace("</Types>",
        '<Default Extension="glb" ContentType="model/gltf-binary"/></Types>')

# --- relationships: model3d (embed) + a fallback image rel ---
used = re.findall(r'Id="rId(\d+)"', rels_xml)
nid = max((int(x) for x in used), default=0)
rid_model = f"rId{nid+1}"; rid_img = f"rId{nid+2}"
new_rels = (
  f'<Relationship Id="{rid_model}" '
  'Type="http://schemas.microsoft.com/office/2017/06/relationships/model3d" '
  'Target="../media/model3d1.glb"/>'
  f'<Relationship Id="{rid_img}" '
  'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
  'Target="../media/robot3dfallback.png"/>'
)
rels_xml = rels_xml.replace("</Relationships>", new_rels + "</Relationships>")

# --- geometry: place over the left visual (EMU). Mirror build.py arch robot box. ---
EMU = 914400
off_x = int(0.62*EMU); off_y = int(0.62*EMU)
ext_cx = int(6.6*EMU); ext_cy = int(4.0*EMU)

frame = (
 f'<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
 f'<mc:Choice xmlns:am3d="http://schemas.microsoft.com/office/drawing/2017/model3d" Requires="am3d">'
 f'<p:graphicFrame>'
 f'<p:nvGraphicFramePr><p:cNvPr id="900" name="Robot 3D model"/>'
 f'<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
 f'<p:xfrm><a:off x="{off_x}" y="{off_y}"/><a:ext cx="{ext_cx}" cy="{ext_cy}"/></p:xfrm>'
 f'<a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/drawing/2017/model3d">'
 f'<am3d:model3D xmlns:am3d="http://schemas.microsoft.com/office/drawing/2017/model3d" '
 f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
 f'<am3d:spPr><a:xfrm><a:off x="{off_x}" y="{off_y}"/><a:ext cx="{ext_cx}" cy="{ext_cy}"/></a:xfrm>'
 f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></am3d:spPr>'
 f'<am3d:relId r:embed="{rid_model}"/>'
 f'<am3d:camera name="orbittopright"/>'
 f'<am3d:extentUnits>ux</am3d:extentUnits>'
 f'</am3d:model3D>'
 f'</a:graphicData></a:graphic></p:graphicFrame>'
 f'</mc:Choice>'
 f'<mc:Fallback>'
 f'<p:pic><p:nvPicPr><p:cNvPr id="901" name="Robot 3D fallback"/>'
 f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
 f'<p:blipFill><a:blip r:embed="{rid_img}" '
 f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
 f'<a:stretch><a:fillRect/></a:stretch></p:blipFill>'
 f'<p:spPr><a:xfrm><a:off x="{off_x}" y="{off_y}"/><a:ext cx="{ext_cx}" cy="{ext_cy}"/></a:xfrm>'
 f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
 f'</mc:Fallback></mc:AlternateContent>'
)
# remove the flat robot render (tagged in build.py) so only the 3D model shows
arch_xml = re.sub(r"<p:pic>(?:(?!</p:pic>).)*?ARCH_ROBOT_RENDER.*?</p:pic>", "", arch_xml, flags=re.S)
# insert just before the closing spTree
arch_xml2 = arch_xml.replace("</p:spTree>", frame + "</p:spTree>")
assert arch_xml2 != arch_xml, "spTree injection failed"

# --- rewrite the zip with the modified/added parts ---
with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.namelist():
        data = zin.read(item)
        if item == "[Content_Types].xml": data = ct_xml.encode("utf-8")
        elif item == arch: data = arch_xml2.encode("utf-8")
        elif item == rels_name: data = rels_xml.encode("utf-8")
        zout.writestr(item, data)
    zout.write(GLB, "ppt/media/model3d1.glb")
    zout.write(ROBOT_IMG, "ppt/media/robot3dfallback.png")

print("wrote", DST, os.path.getsize(DST), "bytes")
# validate reopen
try:
    p = Presentation(DST); n = len(p.slides._sldIdLst)
    print("VALIDATION: python-pptx reopened OK,", n, "slides")
except Exception as e:
    print("VALIDATION FAILED:", e); sys.exit(1)
