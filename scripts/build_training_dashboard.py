"""Build a self-contained HTML dashboard from the latest local Observer/Proposer reports and Proposer train log."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / 'outputs/observer/report.json'
PROP = ROOT / 'outputs/proposer/report.json'
LOG = ROOT / 'outputs/demo_training_20261014/after_b2/proposer_train.log'
OUT = ROOT / 'outputs/demo_training_20261014/after_b2/metrics_dashboard.html'

def e(value):
    return html.escape(str(value))

def percent(value):
    return f'{100 * value:.1f}%'

def bars(title, rows, maximum=100):
    lines = [f'<section class="card"><h2>{e(title)}</h2>']
    for name, value, color, note in rows:
        width = max(0, min(100, value / maximum * 100))
        lines.append(f'<div class="bar"><span>{e(name)}</span><div class="track"><div style="width:{width:.2f}%;background:{color}"></div></div><strong>{value:.1f}%</strong></div>')
        if note:
            lines.append(f'<small>{e(note)}</small>')
    lines.append('</section>')
    return ''.join(lines)

def linechart(title, points, color, ylabel):
    if not points:
        return ''
    x0, x1, y0, y1 = 64, 600, 25, 240
    values = [v for _, v in points]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.14, 0.02)
    lo, hi = lo - pad, hi + pad
    xmin, xmax = points[0][0], points[-1][0]
    xy = [(x0 + (x - xmin) / max(xmax - xmin, 1) * (x1 - x0), y1 - (v - lo) / (hi - lo) * (y1 - y0)) for x, v in points]
    poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in xy)
    dots = ''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in xy)
    grid = ''.join(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#d9ddd3"/>' for y in (y0, (y0+y1)/2, y1))
    labels = ''.join(f'<text x="{x0-8}" y="{y+5}" text-anchor="end" font-size="13" fill="#5b655a">{v:.2f}</text>' for y,v in ((y0,hi),((y0+y1)/2,(lo+hi)/2),(y1,lo)))
    svg = f'<svg viewBox="0 0 640 285" role="img" aria-label="{e(title)}">{grid}{labels}<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3"/>{dots}<text x="{x0}" y="268" font-size="13" fill="#5b655a">{xmin} epoch</text><text x="{x1}" y="268" text-anchor="end" font-size="13" fill="#5b655a">{xmax} epoch</text></svg>'
    return f'<section class="card"><h2>{e(title)}</h2>{svg}<small>{e(ylabel)} · chỉ các mốc đã ghi trong log, không nội suy dữ liệu gốc</small></section>'

def main():
    obs = json.loads(OBS.read_text())
    prop = json.loads(PROP.read_text())
    log = LOG.read_text()
    o = obs['variants']['dinov2_pair_mlp']
    bc = [(int(x), float(y)) for x,y in re.findall(r'bc (\d+)/200 loss ([\d.]+)', log)]
    rl = [(int(x), float(y)) for x,y in re.findall(r'rl (\d+)/120 mean return (-?[\d.]+)', log)]
    folds = [(int(x), float(y)*100) for x,y in re.findall(r'fold (\d)/5: \d+ boards, success ([\d.]+)', log)]
    obsrows=[]
    for name,key in [('Đúng 5 lớp','exact'),('Precision Đạt','good_precision'),('Recall Đạt','good_recall'),('F1 Lỗi','broken_f1')]:
        obsrows.extend([(f'{name} · CV',o['cv'][key]*100,'#4c6a37',''),(f'{name} · test',o['test'][key]*100,'#a35f1a','')])
    proprows=[]
    for name in ('net','rule','random','fixed_start'):
        label={'net':'GRU','rule':'Rule','random':'Random','fixed_start':'Cố định'}[name]
        proprows.append((f'{label} · test',prop['test'][name]['success']*100,'#4c6a37' if name=='net' else '#9ba597',''))
    proprows.append(('Trần lưới · test',prop['test']['net']['oracle']*100,'#56636b','Có ít nhất một ảnh Đạt trong 24 cấu hình'))
    data=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CamouBlend · metrics train 14/10/2026</title><style>
    :root{{--ink:#1b211b;--muted:#5b655a;--bg:#f4f5f1;--line:#d9ddd3}}*{{box-sizing:border-box}}body{{font:17px "Be Vietnam Pro","Segoe UI",sans-serif;color:var(--ink);background:var(--bg);margin:0;padding:28px}}main{{max-width:1320px;margin:auto}}h1{{font-size:38px;margin:0 0 8px}}h2{{font-size:25px;margin:0 0 18px}}p{{color:var(--muted);line-height:1.5}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:white;border:1px solid var(--line);border-radius:10px;padding:22px;margin:12px 0}}.bar{{display:grid;grid-template-columns:190px 1fr 78px;gap:12px;align-items:center;margin:12px 0;font-size:15px}}.track{{height:20px;background:#e6eae1;border-radius:4px;overflow:hidden}}.track>div{{height:100%}}small{{display:block;color:var(--muted);margin:0 0 12px 202px}}svg{{width:100%;height:auto}}.stat{{font-size:30px;font-weight:700;color:#4c6a37}}.warn{{color:#a3402f;font-weight:700}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}.bar{{grid-template-columns:120px 1fr 70px}}small{{margin-left:0}}}}
    </style><main><h1>CamouBlend · metrics sau nhãn đợt 2</h1><p>Observer: {obs['splits']['train/own']:,} ảnh train trực tiếp, {o['test']['n']:,} ảnh test. Proposer: {prop['boards']['train']} cặp train, {prop['boards']['test']} cặp test. DINOv2 frozen; train MLP và GRU trên CPU, chưa LoRA.</p>
    <div class="grid">{bars('Observer · CV và test',obsrows)}{bars('Proposer · success@5 trên test',proprows)}</div>
    <div class="grid">{linechart('Proposer · behavior cloning',bc,'#4c6a37','Loss; 200 epoch')}{linechart('Proposer · REINFORCE',rl,'#a35f1a','Mean return; 120 epoch')}</div>
    <div class="grid">{linechart('Proposer · 5 fold CV',folds,'#56636b','Success@5 trên fold holdout (%)')}<section class="card"><h2>Chỉ số cần đọc đúng</h2><p><span class="stat">{percent(prop['test']['net']['success'])}</span> Proposer trên test khi <b>nhãn người</b> quyết định dừng.</p><p><span class="stat">{percent(prop['cv']['net_model1_judge']['success'])}</span> hai agent tự chạy trên <b>CV train</b>. Chưa đo end-to-end trên test.</p><p class="warn">Hai tỷ lệ dùng hai tập khác nhau; không lấy hiệu số làm mức suy giảm trên cùng tập.</p><p>Observer test precision Đạt {percent(o['test']['good_precision'])}: chỉ một phần tư dự đoán Đạt khớp nhãn người. Đây là điểm nghẽn hiện tại.</p></section></div>
    <p>Nguồn: outputs/observer/report.json, outputs/proposer/report.json, outputs/demo_training_20261014/after_b2/proposer_train.log. Observer script không ghi loss theo bước, nên không có đường loss Observer trong biểu đồ.</p></main></html>'''
    OUT.write_text(data)
    print(OUT)

if __name__ == '__main__':
    main()
