const cfg = window.APP_CONFIG;
let state = {
  imageB64: null,
  filename: null,
  maskB64: null,
  accuracy: null,
  lesions: [],
  reportUrl: null
};

const el = (id) => document.getElementById(id);
const imgOriginal = el('img-original');
const imgMask = el('img-mask');
const canvas = el('canvas-overlay');
const ctx = canvas.getContext('2d');

/* ---------- helpers ---------- */
function showError(msg) {
  const t = el('error-toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 4000);
}

function setButtonsEnabled(v) {
  el('btn-run').disabled = !v;
  el('btn-analyze').disabled = !v || !state.maskB64;
}

function fitCanvasToImage() {
  if (!imgOriginal.naturalWidth) return;
  canvas.width = imgOriginal.naturalWidth;
  canvas.height = imgOriginal.naturalHeight;
}

function drawOverlay(opacity = 0.6) {
  if (!state.imageB64) return;
  fitCanvasToImage();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1.0;
  ctx.drawImage(imgOriginal, 0, 0, canvas.width, canvas.height);
  if (state.maskB64) {
    const tmp = new Image();
    tmp.onload = () => {
      ctx.globalAlpha = opacity;
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
      fadeCanvasOnce(); // 平滑显影
    };
    tmp.src = state.maskB64;
  } else {
    fadeCanvasOnce(); // 没 mask 时也做一次淡入
  }
}

function dataURLToBlob(dataUrl) {
  const [hdr, b64] = dataUrl.split(',');
  const mime = hdr.match(/:(.*?);/)[1];
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

function downloadDataURL(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ---------- Global Loading & 微交互 ---------- */
const $overlay = document.getElementById('loading-overlay');
const $overlayText = document.getElementById('loading-text');

function showLoading(text = 'Working…') {
  if ($overlayText) $overlayText.textContent = text;
  if ($overlay) {
    $overlay.classList.remove('hidden');
    requestAnimationFrame(() => $overlay.classList.add('show'));
  }
}
function hideLoading() {
  if ($overlay) {
    $overlay.classList.remove('show');
    setTimeout(() => $overlay.classList.add('hidden'), 220);
  }
}
async function withLoading(text, fn) {
  try { showLoading(text); return await fn(); }
  finally { hideLoading(); }
}

// 按钮 ripple
function attachRipple(el) {
  if (!el) return;
  el.classList.add('ripple');
  el.addEventListener('click', (e) => {
    const rect = el.getBoundingClientRect();
    const ink = document.createElement('span');
    const size = Math.max(rect.width, rect.height);
    ink.className = 'ripple-ink';
    ink.style.width = ink.style.height = size + 'px';
    ink.style.left = (e.clientX - rect.left - size/2) + 'px';
    ink.style.top  = (e.clientY - rect.top  - size/2) + 'px';
    el.appendChild(ink);
    setTimeout(() => ink.remove(), 650);
  });
}
['btn-run','btn-analyze','btn-download-mask','btn-download-overlay'].forEach(id => attachRipple(el(id)));

// 图片 skeleton + 渐显
function wrapSkeleton(img) {
  if (!img) return;
  const parent = img.parentElement;
  if (!parent) return;
  parent.classList.add('skeleton');
  img.addEventListener('load', () => {
    parent.classList.remove('skeleton');
    img.classList.add('media-fade-in');
  }, { once: true });
}
wrapSkeleton(imgOriginal);
wrapSkeleton(imgMask);

// 滚动入场动画
const io = new IntersectionObserver((entries) => {
  entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add('in'); });
}, { threshold: .12 });
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-animate]').forEach(node => io.observe(node));
});

// 叠加层绘制时的淡入
function fadeCanvasOnce() {
  canvas.style.opacity = 0;
  canvas.style.transition = 'opacity .2s cubic-bezier(.2,.8,.2,1)';
  requestAnimationFrame(() => { canvas.style.opacity = 1; });
}

/* ---------- DnD & File ---------- */
const dropZone = el('drop-zone');
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files[0]);
});

el('file-input').addEventListener('change', (e) => {
  handleFiles(e.target.files[0]);
});

function handleFiles(file) {
  if (!file) return;
  const mb = file.size / 1024 / 1024;
  if (mb > cfg.MAX_IMAGE_MB) {
    showError(`Image too large (${mb.toFixed(1)}MB). Limit ${cfg.MAX_IMAGE_MB}MB.`);
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    state.imageB64 = e.target.result;
    state.filename = file.name;
    imgOriginal.src = state.imageB64;
    setButtonsEnabled(true);
    drawOverlay(parseFloat(el('opacity-range').value));
  };
  reader.readAsDataURL(file);
}

/* ---------- Controls ---------- */
el('opacity-range').addEventListener('input', (e) => drawOverlay(parseFloat(e.target.value)));

el('btn-download-overlay').addEventListener('click', () => {
  if (!state.imageB64) return;
  drawOverlay(parseFloat(el('opacity-range').value));
  downloadDataURL(canvas.toDataURL('image/png'), 'overlay.png');
});

el('btn-download-mask').addEventListener('click', () => {
  if (!state.maskB64) return;
  downloadDataURL(state.maskB64, 'mask.png');
});

/* ---------- Run Segmentation（带全屏 Loading） ---------- */
el('btn-run').addEventListener('click', async () => {
  if (!state.imageB64) return;
  await withLoading('Running segmentation…', async () => {
    setButtonsEnabled(false);
    try {
      if (imgMask && imgMask.parentElement) imgMask.parentElement.classList.add('skeleton');

      const blob = dataURLToBlob(state.imageB64);
      const form = new FormData();
      form.append('image', new File([blob], state.filename || 'image.png', { type: blob.type }));

      const url = (cfg.BASE_URL || '') + '/api/predict/';
      const resp = await fetch(url, { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();

      state.maskB64 = json.mask;
      state.accuracy = json.accuracy ?? null;
      state.lesions = json.lesions || [];

      imgMask.src = state.maskB64;
      drawOverlay(parseFloat(el('opacity-range').value));

      const accText = state.accuracy ? `🎯 Accuracy: ${(state.accuracy * 100).toFixed(2)}%` : "Accuracy: N/A";
      const lesionsText = state.lesions.map(
        (l, i) => `Lesion ${i + 1}: (x=${l.x}, y=${l.y}), size=${l.width}×${l.height}, area=${l.area}`
      ).join('\n');
      el('pred-info').textContent = `${accText}\n${lesionsText}`;

    } catch (err) {
      console.error(err);
      showError(String(err.message || err));
    } finally {
      if (imgMask && imgMask.parentElement) imgMask.parentElement.classList.remove('skeleton');
      setButtonsEnabled(true);
    }
  });
});

/* ---------- Vision Agent（带全屏 Loading） ---------- */
el('btn-analyze').addEventListener('click', async () => {
  if (!state.imageB64) return;
  await withLoading('Asking vision agent…', async () => {
    setButtonsEnabled(false);
    try {
      const payload = {
        image: state.imageB64,
        mask: state.maskB64,
        accuracy: state.accuracy,
        lesions: state.lesions
      };
      const url = (cfg.BASE_URL || '') + '/api/vision';
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      console.log("Vision API response:", data);

      el('agent-output').textContent = "";
      if (data.report_url) {
        state.reportUrl = data.report_url;
        const a = document.createElement('a');
        a.href = (cfg.BASE_URL || '') + data.report_url;
        a.textContent = "📄 Download PDF Report";
        a.target = "_blank";
        a.classList.add("btn", "btn-primary");
        el('agent-output').appendChild(a);
      } else {
        el('agent-output').textContent = "(No report generated)";
      }
    } catch (err) {
      console.error(err);
      showError(String(err.message || err));
    } finally {
      setButtonsEnabled(true);
    }
  });
});
