from __future__ import annotations

AWAKE_HTML = """
<div style="
  background: rgba(10,10,15,0.88);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 20px;
  font-family: -apple-system, 'SF Pro Display', sans-serif;
  color: white;
  width: 340px;
">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
    <div id="orb" style="width:10px; height:10px; border-radius:50%; background:#4ade80;"></div>
    <span style="font-size:13px; font-weight:600; opacity:0.9; letter-spacing:0.05em;">SID</span>
    <span style="font-size:12px; opacity:0.4; margin-left:auto;">listening</span>
  </div>
  <div id="waveform" style="display:flex; gap:3px; align-items:center; height:32px; margin-bottom:12px;"></div>
  <div id="transcript" style="font-size:14px; opacity:0.7; min-height:20px; font-style:italic;"></div>
</div>
<script>
  const waveform = document.getElementById('waveform');
  for (let i = 0; i < 20; i++) {
    const bar = document.createElement('div');
    bar.style.width = '4px';
    bar.style.borderRadius = '4px';
    bar.style.background = 'rgba(74, 222, 128, 0.85)';
    bar.style.height = '8px';
    waveform.appendChild(bar);
  }
  let t = 0;
  setInterval(() => {
    t += 0.2;
    [...waveform.children].forEach((bar, i) => {
      const h = 8 + Math.abs(Math.sin(t + i * 0.45)) * 24;
      bar.style.height = `${h}px`;
    });
  }, 80);
</script>
"""

WORKING_HTML = """
<div style="
  background: rgba(10,10,15,0.88);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 20px;
  font-family: -apple-system, 'SF Pro Display', sans-serif;
  color: white;
  width: 340px;
  max-height: 200px;
  overflow: hidden;
">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
    <div id="orb" style="width:10px;height:10px;border-radius:50%;background:#60a5fa;"></div>
    <span style="font-size:13px; font-weight:600; letter-spacing:0.05em;">SID</span>
    <span style="font-size:12px; opacity:0.4; margin-left:auto;">working</span>
  </div>
  <div id="task-stream" style="font-size:13px; line-height:1.8; opacity:0.75;"></div>
</div>
"""

PROACTIVE_HTML = """
<div style="
  background: rgba(10,10,15,0.88);
  border: 1px solid rgba(250,204,21,0.2);
  border-radius: 16px;
  padding: 14px 18px;
  font-family: -apple-system, 'SF Pro Display', sans-serif;
  color: white;
  width: 340px;
">
  <div style="display:flex; gap:10px; align-items:flex-start;">
    <div style="width:8px;height:8px;border-radius:50%;background:#facc15;margin-top:5px;flex-shrink:0;"></div>
    <span id="proactive-text" style="font-size:14px; line-height:1.6; opacity:0.85;"></span>
  </div>
</div>
"""


def get_state_html(state: str) -> str:
    state = state.upper()
    if state == "AWAKE":
        return AWAKE_HTML
    if state == "WORKING":
        return WORKING_HTML
    if state == "PROACTIVE":
        return PROACTIVE_HTML
    return WORKING_HTML
