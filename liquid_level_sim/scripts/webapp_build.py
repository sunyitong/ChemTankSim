"""Assemble the web app: template + bundled cases -> artifact fragment and standalone local copy.

  python scripts/webapp_build.py <template.html> <cases_embed.js> <fragment_out.html>
The standalone copy is always written to webapp/level_diff_bench.html.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

from mvp_common import PROJ_ROOT

tpl, cases_js, frag_out = (Path(p) for p in sys.argv[1:4])
s = io.open(tpl, encoding="utf-8").read()
assert "/*__CASES__*/" in s
s = s.replace("/*__CASES__*/", io.open(cases_js, encoding="utf-8").read())
assert not re.findall(r"[一-鿿]", s), "CJK text in the UI"
io.open(frag_out, "w", encoding="utf-8", newline="\n").write(s)
head, tail = s.split("</style>\n", 1)
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
       + head + "</style>\n</head>\n<body>\n" + tail + "</body>\n</html>\n")
local = PROJ_ROOT / "webapp" / "level_diff_bench.html"
io.open(local, "w", encoding="utf-8", newline="\n").write(doc)
print(f"fragment {len(s) / 1e6:.2f} MB -> {frag_out}\nstandalone -> {local}")
