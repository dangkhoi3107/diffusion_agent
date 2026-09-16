"""Interactive layout canvas (iframe) that stays in sync with the four layout sliders."""
from __future__ import annotations

import base64
import html
import io
import json

from PIL import Image

from .config import LayoutConfig

LAYOUT_ELEM_IDS = {
    "layout.scale": "camo-layout-scale",
    "layout.offset_x": "camo-layout-offset-x",
    "layout.offset_y": "camo-layout-offset-y",
    "layout.rotation_deg": "camo-layout-rotation",
}

# Copies the canvas state into the 4 layout inputs right before a handler runs.
JS_SYNC_LAYOUT = """
(...args) => {
  const s = window.camoPlacementState;
  if (s && args.length >= 4) {
    const n = args.length;
    args[n - 4] = Number(s.scale);
    args[n - 3] = Number(s.offsetX);
    args[n - 2] = Number(s.offsetY);
    args[n - 1] = Number(s.rotationDeg);
  }
  return args;
}
"""

CANVAS_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"/><style>
html,body{margin:0;background:transparent;font-family:ui-sans-serif,system-ui,sans-serif;color:#e5e7eb;overflow:hidden}
.wrap{border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:12px;background:#111827;min-height:100vh;box-sizing:border-box}
.hint{font-size:13px;line-height:1.5;color:#cbd5e1;margin-bottom:10px}
canvas{width:100%;max-width:__DISPLAY_W__px;height:auto;display:block;border-radius:12px;cursor:grab;background:#0f172a}
.bar{margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{border:0;border-radius:10px;padding:8px 12px;background:#1f2937;color:#f9fafb;cursor:pointer}
button:hover{background:#374151}.badge{font-size:12px;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.06)}
</style></head><body><div class="wrap">
<div class="hint">Drag the box to move the subject, the <b style="color:#60a5fa">blue</b> handle to resize, the
<b style="color:#fbbf24">yellow</b> handle to rotate, the mouse wheel to zoom. Layout sliders stay in sync.</div>
<canvas id="c" width="__W__" height="__H__"></canvas>
<div class="bar"><button data-a="center">Center</button><button data-a="fit">Scale 0.7</button><button data-a="reset">Reset</button>
<span class="badge" id="stats"></span></div></div>
<script>(() => {
  const IDS = __IDS__, MIN_S = 0.1, MAX_S = 4.0, HANDLE = 11, GAP = 42;
  const cv = document.getElementById('c'), ctx = cv.getContext('2d'), stats = document.getElementById('stats');
  const bg = new Image(), ov = new Image(); bg.src = '__BG__'; ov.src = '__OV__';
  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
  const wrapDeg = d => { d = Number(d) || 0; while (d > 180) d -= 360; while (d <= -180) d += 360; return d; };
  const state = {scale: clamp(__S__, MIN_S, MAX_S), offsetX: clamp(__X__, -1, 1), offsetY: clamp(__Y__, -1, 1), rotationDeg: wrapDeg(__R__)};
  let doc = null; try { doc = window.parent.document; } catch (e) {}
  const publish = () => { try { window.parent.camoPlacementState = {...state}; } catch (e) {} };

  function writeSlider(id, value) {
    const host = doc && doc.getElementById(id); if (!host) return;
    const text = String(Number(value).toFixed(id === IDS.rotationDeg ? 1 : 4));
    host.querySelectorAll('input').forEach(inp => { if (inp.value !== text) { inp.value = text;
      inp.dispatchEvent(new Event('input', {bubbles: true})); inp.dispatchEvent(new Event('change', {bubbles: true})); } });
  }
  function readSlider(id, fallback) {
    const inp = doc && doc.getElementById(id) && doc.getElementById(id).querySelector('input');
    const v = inp ? Number(inp.value) : NaN; return Number.isFinite(v) ? v : fallback;
  }
  function geometry() {
    const w = cv.width, h = cv.height, dw = w * state.scale, dh = h * state.scale;
    const cx = w / 2 + state.offsetX * w, cy = h / 2 + state.offsetY * h, rad = state.rotationDeg * Math.PI / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    const toWorld = (x, y) => ({x: cx + x * cos - y * sin, y: cy + x * sin + y * cos});
    const toLocal = (px, py) => { const dx = px - cx, dy = py - cy; return {x: dx * cos + dy * sin, y: -dx * sin + dy * cos}; };
    return {w, h, dw, dh, cx, cy, rad, toWorld, toLocal, resize: toWorld(dw / 2, dh / 2), rotate: toWorld(0, -dh / 2 - GAP)};
  }
  function draw() {
    const g = geometry(); ctx.clearRect(0, 0, g.w, g.h);
    if (bg.complete) ctx.drawImage(bg, 0, 0, g.w, g.h);
    if (ov.complete) { ctx.save(); ctx.translate(g.cx, g.cy); ctx.rotate(g.rad); ctx.drawImage(ov, -g.dw / 2, -g.dh / 2, g.dw, g.dh); ctx.restore(); }
    ctx.save(); ctx.setLineDash([8, 8]); ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 2; ctx.beginPath();
    [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([sx, sy], i) => { const p = g.toWorld(sx * g.dw / 2, sy * g.dh / 2); i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); });
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    const top = g.toWorld(0, -g.dh / 2); ctx.strokeStyle = 'rgba(251,191,36,.9)'; ctx.beginPath(); ctx.moveTo(top.x, top.y); ctx.lineTo(g.rotate.x, g.rotate.y); ctx.stroke();
    [[g.resize, '#60a5fa'], [g.rotate, '#fbbf24']].forEach(([p, c]) => { ctx.beginPath(); ctx.fillStyle = c; ctx.arc(p.x, p.y, HANDLE, 0, 2 * Math.PI); ctx.fill(); });
    ctx.restore();
    stats.textContent = `scale=${state.scale.toFixed(3)} x=${state.offsetX.toFixed(3)} y=${state.offsetY.toFixed(3)} rot=${state.rotationDeg.toFixed(1)}°`;
  }
  function commit() { publish(); Object.keys(IDS).forEach(k => writeSlider(IDS[k], state[k])); draw(); }
  function point(e) { const r = cv.getBoundingClientRect(); return {x: (e.clientX - r.left) * cv.width / r.width, y: (e.clientY - r.top) * cv.height / r.height}; }

  let mode = null, start = null;
  cv.addEventListener('mousedown', e => {
    e.preventDefault(); const p = point(e), g = geometry(), local = g.toLocal(p.x, p.y);
    if (Math.hypot(p.x - g.rotate.x, p.y - g.rotate.y) <= HANDLE * 1.6) { mode = 'rotate'; start = {rot: state.rotationDeg, ang: Math.atan2(p.y - g.cy, p.x - g.cx)}; }
    else if (Math.hypot(p.x - g.resize.x, p.y - g.resize.y) <= HANDLE * 1.6) { mode = 'resize'; }
    else if (Math.abs(local.x) <= g.dw / 2 && Math.abs(local.y) <= g.dh / 2) { mode = 'drag'; start = {x: p.x, y: p.y, ox: state.offsetX, oy: state.offsetY}; }
  });
  window.addEventListener('mousemove', e => {
    if (!mode) return; const p = point(e), g = geometry();
    if (mode === 'drag') { state.offsetX = clamp(start.ox + (p.x - start.x) / g.w, -1, 1); state.offsetY = clamp(start.oy + (p.y - start.y) / g.h, -1, 1); }
    else if (mode === 'resize') { const l = g.toLocal(p.x, p.y); state.scale = clamp(Math.max(Math.abs(l.x) / (g.w / 2), Math.abs(l.y) / (g.h / 2)), MIN_S, MAX_S); }
    else { const d = Math.atan2(p.y - g.cy, p.x - g.cx) - start.ang; state.rotationDeg = wrapDeg(start.rot + d * 180 / Math.PI); }
    commit();
  });
  const stop = () => { mode = null; start = null; };
  window.addEventListener('mouseup', stop); cv.addEventListener('mouseleave', stop);
  cv.addEventListener('wheel', e => { e.preventDefault(); state.scale = clamp(state.scale * (e.deltaY < 0 ? 1.05 : 0.95), MIN_S, MAX_S); commit(); }, {passive: false});
  document.querySelectorAll('button[data-a]').forEach(b => b.addEventListener('click', () => {
    const a = b.dataset.a;
    if (a !== 'fit') { state.offsetX = 0; state.offsetY = 0; }
    if (a === 'fit') state.scale = 0.7;
    if (a === 'reset') { state.scale = 1; state.rotationDeg = 0; }
    commit();
  }));
  setInterval(() => {  // slider -> canvas
    if (mode) return;
    const next = {scale: clamp(readSlider(IDS.scale, state.scale), MIN_S, MAX_S), offsetX: clamp(readSlider(IDS.offsetX, state.offsetX), -1, 1),
                  offsetY: clamp(readSlider(IDS.offsetY, state.offsetY), -1, 1), rotationDeg: wrapDeg(readSlider(IDS.rotationDeg, state.rotationDeg))};
    if (Object.keys(next).some(k => Math.abs(next[k] - state[k]) > 1e-4)) { Object.assign(state, next); publish(); draw(); }
  }, 150);
  bg.onload = draw; ov.onload = draw; publish(); draw();
})();</script></body></html>"""


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def canvas_html(background: Image.Image, overlay: Image.Image, width: int, height: int, layout: LayoutConfig) -> str:
    ids = {"scale": LAYOUT_ELEM_IDS["layout.scale"], "offsetX": LAYOUT_ELEM_IDS["layout.offset_x"],
           "offsetY": LAYOUT_ELEM_IDS["layout.offset_y"], "rotationDeg": LAYOUT_ELEM_IDS["layout.rotation_deg"]}
    display_w = min(max(width, 320), 640)
    doc = (
        CANVAS_TEMPLATE.replace("__IDS__", json.dumps(ids))
        .replace("__DISPLAY_W__", str(display_w)).replace("__W__", str(width)).replace("__H__", str(height))
        .replace("__BG__", _data_uri(background.convert("RGB"))).replace("__OV__", _data_uri(overlay))
        .replace("__S__", f"{layout.scale:.6f}").replace("__X__", f"{layout.offset_x:.6f}")
        .replace("__Y__", f"{layout.offset_y:.6f}").replace("__R__", f"{layout.rotation_deg:.6f}")
    )
    frame_h = int(display_w * height / max(width, 1)) + 140
    return f'<iframe srcdoc="{html.escape(doc, quote=True)}" style="width:100%;height:{frame_h}px;border:0;background:transparent"></iframe>'
