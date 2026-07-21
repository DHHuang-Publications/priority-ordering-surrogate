"""
make_drawio.py
==============
從拓撲資料產生 draw.io XML（可於 https://app.diagrams.net 開啟編輯美化）。

用法:
  python make_drawio.py nsfnet   -> nsfnet_topology.drawio.xml （標 K=5 對）
  python make_drawio.py usnet    -> usnet_topology.drawio.xml  （標 K=10 對）
  python make_drawio.py          -> 兩個都產

節點顏色（s-t 角色）：source=綠、terminal=紅、relay=灰。
若改了 s-t 對，重跑本檔即可更新。
"""

# --- path bootstrap (restructured package) ---
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('', 'core', 'models', 'instances', 'baselines', 'experiments', 'figures_code'):
    _p = _os.path.join(ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
HERE = ROOT   # data/ and figures/ live at the package root after restructuring
CODE = ROOT

import os
import sys


# draw.io 柔和配色
GREEN = "fillColor=#d5e8d4;strokeColor=#82b366"
RED   = "fillColor=#f8cecc;strokeColor=#b85450"
GRAY  = "fillColor=#f5f5f5;strokeColor=#999999"

# 各拓撲畫布參數（W,H 依節點數調整；節點直徑 S）
CANVAS = {
    'nsfnet': dict(W=1040.0, H=540.0, M=50.0, S=58.0, pairs_attr='ST_PAIRS_K5',  klabel='K=5'),
    'usnet':  dict(W=1240.0, H=620.0, M=55.0, S=50.0, pairs_attr='ST_PAIRS_K10', klabel='K=10'),
}


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def build_xml(topology):
    if topology == 'nsfnet':
        import nsfnet_data as data
    elif topology == 'usnet':
        import usnet_data as data
    else:
        raise ValueError(topology)
    cv = CANVAS[topology]
    W, H, M, S = cv['W'], cv['H'], cv['M'], cv['S']
    pairs = getattr(data, cv['pairs_attr'])
    NODES, LINKS = data.NODES, data.LINKS

    lons = [NODES[n][1] for n in NODES]
    lats = [NODES[n][2] for n in NODES]
    lo0, lo1, la0, la1 = min(lons), max(lons), min(lats), max(lats)

    def pos(n):
        _, lon, lat = NODES[n]
        cx = (lon - lo0) / (lo1 - lo0) * W + M
        cy = (la1 - lat) / (la1 - la0) * H + M   # 緯度翻轉：北在上
        return cx - S / 2, cy - S / 2

    sources = {s for (s, t) in pairs}
    terminals = {t for (s, t) in pairs}

    cells = []
    for i, (u, v) in enumerate(LINKS, 1):
        cells.append(
            f'<mxCell id="e{i}" style="endArrow=none;html=1;strokeColor=#9AA5B1;'
            f'strokeWidth=1.6;edgeStyle=none;rounded=0;" edge="1" parent="1" '
            f'source="n{u}" target="n{v}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    for n in NODES:
        city = NODES[n][0].split(',')[0]
        x, y = pos(n)
        style = ("ellipse;whiteSpace=wrap;html=1;fontSize=10;fontStyle=1;verticalAlign=middle;"
                 + (GREEN if n in sources else RED if n in terminals else GRAY))
        val = esc(f"<b>{n}</b><br>{city}")
        cells.append(
            f'<mxCell id="n{n}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.0f}" y="{y:.0f}" width="{S:.0f}" height="{S:.0f}" as="geometry"/></mxCell>')

    # 圖例
    ly = H + M + 60
    legend = [
        ("lg_t", f"Source-terminal ({cv['klabel']}) roles",
         "text;html=1;fontStyle=1;fontSize=12;align=left", M, ly - 30, 320, 20),
        ("lg_s", "Source", "ellipse;html=1;fontSize=10;" + GREEN, M, ly, 90, 30),
        ("lg_r", "Terminal", "ellipse;html=1;fontSize=10;" + RED, M + 110, ly, 90, 30),
        ("lg_g", "Relay", "ellipse;html=1;fontSize=10;" + GRAY, M + 220, ly, 90, 30),
    ]
    for cid, val, style, x, y, w, h in legend:
        cells.append(
            f'<mxCell id="{cid}" value="{esc(val)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    body = "\n        ".join(cells)
    pw, ph = int(W + 2 * M + 60), int(H + 2 * M + 120)
    xml = f'''<mxfile host="app.diagrams.net" type="device">
  <diagram id="{topology}" name="{topology.upper()} topology">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1"
                  connect="1" arrows="1" fold="1" page="1" pageScale="1"
                  pageWidth="{pw}" pageHeight="{ph}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
    out = os.path.join(HERE, 'figures', f'{topology}_topology.drawio.xml')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f"[存檔] {out}  (節點 {len(NODES)}、鏈路 {len(LINKS)}、"
          f"source={sorted(sources)}, terminal={sorted(terminals)})")


if __name__ == "__main__":
    topos = [sys.argv[1]] if len(sys.argv) > 1 else ['nsfnet', 'usnet']
    for t in topos:
        build_xml(t)
