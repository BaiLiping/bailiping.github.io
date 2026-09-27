#!/usr/bin/env python3
"""Place the original PowerPoint plot pixels in cropped, self-contained SVGs.

The SVG viewport excludes the embedded headings. Original plot and annotation
PNG bytes stay unchanged, with the red ink placed using PowerPoint coordinates.
A header-only SVG mask clears slide 8's title edge above its top axis tick.
"""

import argparse
import base64
import hashlib
import io
import json
import posixpath
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile

from PIL import Image


ASSETS = Path(__file__).resolve().parent / "assets"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
FIGURES = [
    (6, [0, 30, 1718, 726], (1718, 752), "BS5 distributed EOT, frame 182/200"),
    (7, [0, 47, 1679, 726], (1679, 808), "BS5 distributed EOT, frame 91/200"),
    (8, [0, 31, 1717, 726], (1717, 757), "BS3 distributed EOT, frame 139/200"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("powerpoint", type=Path)
    args = parser.parse_args()
    report = {"source": args.powerpoint.name, "figures": []}
    with ZipFile(args.powerpoint) as archive:
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        relationships = {r.attrib["Id"]: r.attrib["Target"] for r in
                         ET.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))}
        for page, crop, expected_size, description in FIGURES:
            slide_id = presentation.find("p:sldIdLst", NS)[page - 1].attrib[REL + "id"]
            slide = posixpath.normpath(posixpath.join("ppt", relationships[slide_id]))
            doc = ET.fromstring(archive.read(slide))
            relpath = posixpath.join(posixpath.dirname(slide), "_rels", posixpath.basename(slide) + ".rels")
            media_refs = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(archive.read(relpath))}
            pictures = []
            for picture in doc.findall(".//p:pic", NS):
                blip = picture.find("p:blipFill/a:blip", NS)
                media = posixpath.normpath(posixpath.join(posixpath.dirname(slide), media_refs[blip.attrib[REL + "embed"]]))
                data = archive.read(media)
                transform = picture.find("p:spPr/a:xfrm", NS)
                assert not transform.attrib, "Rotated or flipped pictures need explicit handling"
                assert picture.find("p:blipFill/a:srcRect", NS) is None, "Unexpected source crop"
                offset = transform.find("a:off", NS)
                extent = transform.find("a:ext", NS)
                geometry = [int(offset.attrib["x"]), int(offset.attrib["y"]),
                            int(extent.attrib["cx"]), int(extent.attrib["cy"])]
                with Image.open(io.BytesIO(data)) as image:
                    assert image.format == "PNG"
                    pictures.append({"media": media, "data": data, "size": image.size, "geometry": geometry})
            main_picture = max(pictures, key=lambda picture: picture["size"][0] * picture["size"][1])
            assert main_picture["size"] == expected_size
            origin_x, origin_y, original_width, original_height = main_picture["geometry"]
            scale_x = expected_size[0] / original_width
            scale_y = expected_size[1] / original_height
            layers = []
            layer_report = []
            for picture in pictures:
                px, py, pw, ph = picture["geometry"]
                geometry = [(px - origin_x) * scale_x, (py - origin_y) * scale_y,
                            pw * scale_x, ph * scale_y]
                ix, iy, iw, ih = geometry
                payload = base64.b64encode(picture["data"]).decode("ascii")
                layers.append(f'  <image x="{ix:.6f}" y="{iy:.6f}" width="{iw:.6f}" height="{ih:.6f}" preserveAspectRatio="none" href="data:image/png;base64,{payload}"/>')
                layer_report.append({
                    "powerpoint_media": picture["media"], "source_size": list(picture["size"]),
                    "powerpoint_geometry": picture["geometry"], "svg_geometry": geometry,
                    "role": "plot" if picture is main_picture else "red annotation",
                    "source_sha256": hashlib.sha256(picture["data"]).hexdigest(),
                })
            x, y, width, height = crop
            image_layers = "\n".join(layers)
            # The title's last antialiased row overlaps the top of the 250 tick
            # vertically, so clear only the title area instead of cropping it.
            title_masks = [[250, 0, 400, 33]] if page == 8 else []
            masks = "\n".join(f'  <rect x="{mx}" y="{my}" width="{mw}" height="{mh}" fill="white"/>'
                              for mx, my, mw, mh in title_masks)
            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{x} {y} {width} {height}" overflow="hidden" role="img" aria-labelledby="title description">
  <title id="title">{escape(description)}</title>
  <desc id="description">Tracking and particle-cloud plots from PowerPoint slide {page}. The embedded headings are removed. Axes, trajectories, particle clouds, and the original red highlight remain unchanged.</desc>
{image_layers}
{masks}
</svg>
'''
            output = ASSETS / f"multimodal-slide-{page}.svg"
            output.write_text(svg)
            # Verify the plots and original red highlights byte for byte.
            saved = ET.fromstring(output.read_bytes()).findall("{http://www.w3.org/2000/svg}image")
            assert len(saved) == len(pictures)
            for layer, picture in zip(saved, pictures):
                assert base64.b64decode(layer.attrib["href"].split(",", 1)[1]) == picture["data"]
            report["figures"].append({
                "powerpoint_slide": page, "output": output.name, "viewport": crop,
                "title_masks": title_masks, "panels": 2,
                "description": description, "layers": layer_report,
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "source_pixels_preserved": True,
            })
            print(f"Slide {page}: {output.name}, plot and red annotation preserved")
    (ASSETS / "multimodal-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
