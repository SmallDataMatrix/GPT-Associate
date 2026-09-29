'use strict';

const $ = (sel) => document.querySelector(sel);
const SPEAKERS = { them: 'Them', me: 'Me', room: 'Room' };
const KIND_LABELS = { auto: 'heard', manual: 'answer now', typed: 'typed', detail: 'in detail', search: 'web', improve: 'improve' };
const FIELDS = ['about_me', 'goal', 'other_side', 'keywords', 'base', 'brief'];
const MAX_SEGMENTS = 600;
const codeParam = new URLSearchParams(location.search).get('code');
const canCapture = window.isSecureContext && !!navigator.mediaDevices?.getDisplayMedia;

const state = { cards: new Map(), partials: new Map(), session: '' };

// ---------------------------------------------------------------- helpers

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function withCode(path) {
  return codeParam ? `${path}${path.includes('?') ? '&' : '?'}code=${encodeURIComponent(codeParam)}` : path;
}

function wsUrl(path) {
  return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${withCode(path)}`;
}

async function api(path, { method = 'GET', json, body } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.body = JSON.stringify(json);
    opts.headers['Content-Type'] = 'application/json';
  } else if (body) {
    opts.body = body;
  }
  const res = await fetch(withCode(path), opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `Request failed (${res.status})`);
  return data;
}

function toast(message, level = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${level}`;
  el.textContent = message;
  $('#toasts').append(el);
  setTimeout(() => el.remove(), level === 'error' ? 9000 : 5000);
}

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {
    return null;
  }
  return null;
}

const sec = (ms) => (ms == null ? '–' : `${(ms / 1000).toFixed(1)} s`);

function inline(s) {
  return s
    .replace(/^#+\s*/, '')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]{1,80})\]/g, '<mark>[$1]</mark>');
}

function renderText(text) {
  let html = '';
  let inList = false;
  for (const line of esc(text).split('\n')) {
    const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      if (!inList) html += '<ul>';
      inList = true;
      html += `<li>${inline(item[1])}</li>`;
      continue;
    }
    if (inList) html += '</ul>';
    inList = false;
    if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  return inList ? `${html}</ul>` : html;
}

// ---------------------------------------------------------------- live socket

let ui = null;
let uiRetry = 0;

function connectUi() {
  ui = new WebSocket(wsUrl('/ws/ui'));
  ui.onopen = () => {
    uiRetry = 0;
    $('#conn').classList.add('on');
  };
  ui.onclose = () => {
    $('#conn').classList.remove('on');
    setTimeout(connectUi, Math.min(5000, 500 * 2 ** uiRetry++));
  };
  ui.onmessage = (e) => handle(JSON.parse(e.data));
}

function send(message) {
  if (ui && ui.readyState === WebSocket.OPEN) ui.send(JSON.stringify(message));
  else toast('Not connected to the server yet.', 'error');
}

function handle(ev) {
  switch (ev.type) {
    case 'snapshot': applySnapshot(ev); break;
    case 'segment': addSegment(ev.segment); break;
    case 'partial': setPartial(ev.id, ev.speaker, ev.text); break;
    case 'partial_drop': dropPartial(ev.id); break;
    case 'card': upsertCard(ev.card); break;
    case 'delta': appendDelta(ev.id, ev.d); break;
    case 'notes': renderNotes(ev.notes); break;
    case 'cn': renderCn(ev.text); break;
    case 'usage': renderUsage(ev.usage); break;
    case 'stt': renderStt(ev.stt); break;
    case 'auto': $('#auto').checked = ev.value; break;
    case 'profile': renderProfileStatus(ev.profile); break;
    case 'toast': toast(ev.message, ev.level); break;
  }
}

function applySnapshot(s) {
  state.session = s.session;
  state.cards.clear();
  state.partials.clear();
  $('#cards').innerHTML = '';
  $('#segs').innerHTML = '';
  $('#parts').innerHTML = '';
  s.transcript.forEach(addSegment);
  s.partials.forEach((p) => setPartial(p.id, p.speaker, p.text));
  if (!s.cards.length) {
    $('#cards').innerHTML = '<p class="empty">Answers appear here. Ask something below, press Answer now, or start listening.</p>';
  }
  s.cards.forEach(upsertCard);
  renderNotes(s.notes);
  renderCn(s.cn);
  renderUsage(s.usage);
  renderStt(s.stt);
  renderProfileStatus(s.profile);
  $('#auto').checked = s.auto;
}

// ---------------------------------------------------------------- transcript

function withAutoscroll(fn) {
  const box = $('#transcript');
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  fn();
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function segmentEl(speaker, text) {
  const el = document.createElement('p');
  el.className = `seg ${speaker}`;
  el.innerHTML = `<span class="who">${esc(SPEAKERS[speaker] || speaker)}</span><span class="txt">${esc(text)}</span>`;
  return el;
}

function addSegment(seg) {
  withAutoscroll(() => {
    dropPartial(seg.id);
    const segs = $('#segs');
    segs.append(segmentEl(seg.speaker, seg.text));
    while (segs.childElementCount > MAX_SEGMENTS) segs.firstElementChild.remove();
  });
}

function setPartial(id, speaker, text) {
  withAutoscroll(() => {
    let el = state.partials.get(id);
    if (!el) {
      el = segmentEl(speaker, '');
      el.classList.add('partial');
      state.partials.set(id, el);
      $('#parts').append(el);
    }
    el.querySelector('.txt').textContent = text;
  });
}

function dropPartial(id) {
  const el = state.partials.get(id);
  if (el) el.remove();
  state.partials.delete(id);
}

// ---------------------------------------------------------------- answer cards

function upsertCard(card) {
  state.cards.set(card.id, card);
  let el = document.getElementById(`card-${card.id}`);
  if (!el) {
    $('#cards .empty')?.remove();
    el = document.createElement('article');
    el.id = `card-${card.id}`;
    el.innerHTML = '<div class="q"></div><div class="a"></div><footer></footer>';
    el.querySelector('footer').addEventListener('click', (e) => {
      const act = e.target.closest('button')?.dataset.act;
      if (act) send({ type: act, card_id: card.id });
    });
    $('#cards').prepend(el);
    document.querySelectorAll('#cards .card.latest').forEach((c) => c.classList.remove('latest'));
    el.classList.add('latest');
  }
  renderCard(el, card);
  if (card.status === 'done' && (card.kind === 'answer' || card.kind === 'improve')) renderLatency(card);
}

function renderCard(el, card) {
  const latest = el.classList.contains('latest');
  el.className = `card kind-${card.kind} status-${card.status}${latest ? ' latest' : ''}`;
  const q = card.question.length > 240 ? `…${card.question.slice(-240)}` : card.question;
  el.querySelector('.q').innerHTML = `<span class="kind">${esc(KIND_LABELS[card.source] || card.kind)}</span><b>${esc(q)}</b>`;
  el.querySelector('.a').innerHTML = card.status === 'error' ? esc(card.error) : renderText(card.text);
  const busy = ['pending', 'searching', 'quick', 'streaming'].includes(card.status);
  const statusText = {
    pending: card.kind === 'detail' ? 'thinking it through…' : 'thinking…',
    searching: 'searching the web…',
    quick: 'thinking took too long, answering directly…',
    streaming: `first words ${sec(card.first_ms)}`,
    done: `first words ${sec(card.first_ms)} · complete ${sec(card.done_ms)}`, cancelled: 'stopped', error: 'failed',
  }[card.status] || card.status;
  let buttons = '';
  if (busy) buttons += '<button data-act="stop">Stop</button>';
  if (!busy && (card.kind === 'answer' || card.kind === 'detail')) buttons += '<button data-act="search">Search web</button>';
  if (!busy && card.kind === 'answer') buttons += '<button data-act="detail" title="A fuller answer from a stronger model (D)">Answer in details</button>';
  if (!busy && card.kind !== 'improve') buttons += '<button data-act="improve">Improve</button>';
  el.querySelector('footer').innerHTML = `<span class="meta">${esc(statusText)}</span>${buttons}`;
}

function appendDelta(id, delta) {
  const card = state.cards.get(id);
  if (!card) return;
  card.text += delta;
  const el = document.getElementById(`card-${id}`);
  if (el) el.querySelector('.a').innerHTML = renderText(card.text);
}

function renderLatency(card) {
  if (card.first_ms == null) return;
  const el = $('#stat-latency');
  el.textContent = `⚡ ${sec(card.first_ms)} / ${sec(card.done_ms)}`;
  el.className = `stat ${card.done_ms != null && card.done_ms <= 3000 ? 'good' : 'slow'}`;
}

// ---------------------------------------------------------------- status panels

function renderUsage(u) {
  if (!u) return;
  $('#stat-cost').textContent = `$${u.cost.toFixed(u.cost < 1 ? 3 : 2)}`;
  $('#stat-cache').textContent = u.input_tokens ? `cache ${u.cached_pct}%` : '';
}

function renderStt(stt) {
  $('#stt-status').innerHTML = Object.entries(stt || {})
    .map(([src, st]) => `<span class="${esc(st)}">${esc(SPEAKERS[src] || src)}: ${esc(st)}</span>`)
    .join('');
}

function renderNotes(notes) {
  $('#notes').textContent = notes.text || 'Notes build up automatically as the conversation goes on.';
}

function renderCn(text) {
  const panel = $('#cn-panel');
  panel.hidden = !text;
  if (text) $('#cn-content').innerHTML = renderText(text);
}

function describeProfile(p) {
  if (!p || !p.slug) return 'no background loaded';
  return p.tokens ? `${p.name}: ~${p.tokens.toLocaleString()} tokens, ${p.summary}` : `${p.name}: ${p.summary}`;
}

function renderProfileStatus(p) {
  $('#profile-status').textContent = `· ${describeProfile(p)}`;
}

// ---------------------------------------------------------------- audio capture

const capture = { active: false, sources: {} };

async function getStream(kind) {
  if (kind === 'mic') {
    return navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  }
  if (kind === 'display') {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 1 } },
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      systemAudio: 'include',
      selfBrowserSurface: 'exclude',
      surfaceSwitching: 'include',
    });
    if (!stream.getAudioTracks().length) {
      stream.getTracks().forEach((t) => t.stop());
      throw new Error('No audio was shared. Pick the meeting tab and turn on "Share tab audio" (or "Share system audio" for a whole screen).');
    }
    return stream;
  }
  // A specific input device, e.g. a BlackHole loopback that carries the Teams/Zoom app audio on macOS.
  return navigator.mediaDevices.getUserMedia({
    audio: { deviceId: { exact: kind }, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
}

function openAudioSocket(entry) {
  const ws = new WebSocket(wsUrl(`/ws/audio?src=${entry.src}`));
  ws.binaryType = 'arraybuffer';
  ws.onclose = () => {
    if (!entry.closed) setTimeout(() => !entry.closed && openAudioSocket(entry), 1000);
  };
  entry.ws = ws;
}

async function startSource(src, stream) {
  const ctx = new AudioContext({ latencyHint: 'interactive' });
  const entry = { src, stream, ctx, ws: null, closed: false, level: 0 };
  capture.sources[src] = entry;
  await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
  const input = ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
  const node = new AudioWorkletNode(ctx, 'pcm16-writer', { processorOptions: { targetRate: 24000, chunkMs: 100 } });
  const sink = ctx.createGain();
  sink.gain.value = 0; // keeps the graph running without playing the audio back
  input.connect(node);
  node.connect(sink);
  sink.connect(ctx.destination);
  node.port.onmessage = (e) => {
    if (entry.ws && entry.ws.readyState === WebSocket.OPEN) entry.ws.send(e.data.pcm);
    entry.level = e.data.level;
  };
  stream.getAudioTracks()[0].addEventListener('ended', () => {
    if (entry.closed) return;
    toast(`${SPEAKERS[src]} audio stopped.`);
    stopListening();
  });
  openAudioSocket(entry);
  $('#meters').insertAdjacentHTML('beforeend', `<span class="meter" data-src="${src}">${SPEAKERS[src]} <i></i></span>`);
}

function stopListening() {
  for (const entry of Object.values(capture.sources)) {
    entry.closed = true;
    try { entry.ws?.close(); } catch { /* already closed */ }
    entry.stream.getTracks().forEach((t) => t.stop());
    entry.ctx.close().catch(() => {});
  }
  capture.sources = {};
  capture.active = false;
  $('#meters').innerHTML = '';
  renderListenButton();
}

async function startListening() {
  const mode = $('#mode').value;
  const them = $('#them-source').value;
  // The screen-share picker needs the click's user gesture, so it is requested first.
  const plan = mode === 'inperson' ? [['room', 'mic']]
    : mode === 'online-them' ? [['them', them]]
    : [['them', them], ['me', 'mic']];
  $('#btn-listen').disabled = true;
  try {
    for (const [src, kind] of plan) await startSource(src, await getStream(kind));
    capture.active = true;
    refreshDevices();
  } catch (err) {
    stopListening();
    const denied = err && err.name === 'NotAllowedError';
    toast(denied ? 'Permission to capture audio was denied or cancelled.' : err.message || String(err), 'error');
  } finally {
    $('#btn-listen').disabled = false;
    renderListenButton();
  }
}

function renderListenButton() {
  const btn = $('#btn-listen');
  btn.textContent = capture.active ? 'Stop listening' : 'Start listening';
  btn.classList.toggle('primary', !capture.active);
  btn.classList.toggle('danger', capture.active);
  $('#mode').disabled = capture.active;
  $('#them-source').disabled = capture.active;
  $('#them-source').hidden = $('#mode').value === 'inperson';
}

async function refreshDevices() {
  if (!canCapture) return;
  const select = $('#them-source');
  const keep = select.value;
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  const inputs = devices.filter((d) => d.kind === 'audioinput' && d.deviceId && !['default', 'communications'].includes(d.deviceId));
  select.innerHTML = '<option value="display">Them: share tab / screen audio</option>'
    + inputs.map((d) => `<option value="${esc(d.deviceId)}">Them: ${esc(d.label || 'audio input')}</option>`).join('');
  if ([...select.options].some((o) => o.value === keep)) select.value = keep;
}

function animateMeters() {
  for (const entry of Object.values(capture.sources)) {
    const bar = document.querySelector(`.meter[data-src="${entry.src}"] i`);
    if (bar) bar.style.setProperty('--level', `${Math.min(100, Math.round(entry.level * 400))}%`);
  }
  requestAnimationFrame(animateMeters);
}

// ---------------------------------------------------------------- background profiles

let currentSlug = '';
let profileList = [];
let docKinds = {};
let baseDocs = [];

function kindSelect(d) {
  const options = Object.entries(docKinds)
    .map(([k, label]) => `<option value="${esc(k)}"${k === d.kind ? ' selected' : ''}>${esc(label)}</option>`).join('');
  return `<select data-kind-for="${esc(d.name)}" title="How this file is used">${options}</select>`;
}

function renderDocs(docs) {
  const size = (n) => (n < 1000 ? `${n} chars` : `${Math.round(n / 1000)}k chars`);
  const own = docs.map((d) => `<li><span class="doc-name">${esc(d.name)}</span> <span class="muted">${size(d.chars)}</span>`
    + `${kindSelect(d)}<button type="button" data-doc="${esc(d.name)}" title="Remove">✕</button></li>`);
  const inherited = baseDocs.map((d) => `<li class="inherited"><span class="doc-name">${esc(d.name)}</span>`
    + ` <span class="muted">from ${esc(d.from)} · ${esc(docKinds[d.kind] || d.kind)}</span></li>`);
  $('#doc-list').innerHTML = own.concat(inherited).join('');
}

function renderBaseOptions(slug, base) {
  const others = profileList.filter((p) => p.slug !== slug);
  $('#f-base').innerHTML = '<option value="">(nothing)</option>'
    + others.map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join('');
  $('#f-base').value = others.some((p) => p.slug === base) ? base : '';
}

function applyProfile(p) {
  docKinds = p.kinds || docKinds;
  baseDocs = p.base_docs || [];
  renderBaseOptions(p.slug, p.base);
  for (const f of FIELDS) if (f !== 'base') $(`#f-${f}`).value = p[f] || '';
  $('#brief-wrap').hidden = !p.brief;
  renderDocs(p.docs);
}

async function loadProfile(slug) {
  currentSlug = slug;
  applyProfile(await api(`/api/profiles/${encodeURIComponent(slug)}`));
}

async function loadProfiles(selectSlug) {
  const data = await api('/api/profiles');
  let list = data.profiles;
  if (!list.length) {
    const created = await api('/api/profiles', { method: 'POST', json: { name: 'Default' } });
    list = [{ slug: created.slug, name: 'Default' }];
  }
  profileList = list;
  $('#profile-select').innerHTML = list.map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join('');
  const slug = [selectSlug, data.active, list[0].slug].find((s) => s && list.some((p) => p.slug === s));
  $('#profile-select').value = slug;
  await loadProfile(slug);
}

function profileFields() {
  const out = {};
  for (const f of FIELDS) out[f] = $(`#f-${f}`).value;
  if ($('#brief-wrap').hidden) delete out.brief;
  return out;
}

async function prepare() {
  const btn = $('#btn-prepare');
  const status = $('#prepare-status');
  btn.disabled = true;
  status.textContent = 'Saving…';
  try {
    await api(`/api/profiles/${encodeURIComponent(currentSlug)}`, { method: 'PUT', json: profileFields() });
    status.textContent = 'Preparing… (the prep pack is rewritten when the material changed, this can take up to a minute)';
    const info = await api(`/api/profiles/${encodeURIComponent(currentSlug)}/prepare`, { method: 'POST' });
    await loadProfile(currentSlug);
    renderProfileStatus(info);
    status.textContent = `Ready: ${describeProfile(info)}`;
    if (info.warning) toast(info.warning, 'error');
  } catch (err) {
    status.textContent = '';
    toast(err.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  $('#prepare-status').textContent = 'Uploading…';
  try {
    const result = await api(`/api/profiles/${encodeURIComponent(currentSlug)}/docs`, { method: 'POST', body: form });
    renderDocs(result.docs);
    result.errors.forEach((m) => toast(m, 'error'));
    $('#prepare-status').textContent = 'Uploaded. Press Save & prepare to use it.';
  } catch (err) {
    $('#prepare-status').textContent = '';
    toast(err.message, 'error');
  }
}

// ---------------------------------------------------------------- past meetings

async function loadSessions() {
  const data = await api('/api/sessions');
  const when = (t) => (t ? new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');
  $('#session-select').innerHTML = data.sessions.length
    ? data.sessions.map((m) => `<option value="${esc(m.id)}">${esc(when(m.started))}`
      + `${m.id === data.current ? ' (this meeting)' : ''} · ${esc(m.profile || 'no profile')} · ${m.questions} answers</option>`).join('')
    : '<option value="">No saved meetings yet</option>';
  $('#btn-import').disabled = $('#btn-download').disabled = !data.sessions.length;
}

async function downloadSummary(sessionId) {
  if (!sessionId) return;
  toast('Writing the summary…');
  const res = await fetch(withCode(`/api/sessions/${encodeURIComponent(sessionId)}/summary.md`));
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Download failed (${res.status})`);
  }
  const link = document.createElement('a');
  link.href = URL.createObjectURL(await res.blob());
  link.download = `meeting-${sessionId}.md`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 10000);
}

async function importMeeting() {
  const sessionId = $('#session-select').value;
  if (!sessionId) return;
  const status = $('#prepare-status');
  status.textContent = 'Adding the meeting (writing its summary first)…';
  try {
    const p = await api(`/api/profiles/${encodeURIComponent(currentSlug)}/import`, { method: 'POST', json: { session_id: sessionId } });
    renderDocs(p.docs);
    status.textContent = 'Added as a past interview. Press Save & prepare to use it.';
  } catch (err) {
    status.textContent = '';
    toast(err.message, 'error');
  }
}

// ---------------------------------------------------------------- wiring

function wire() {
  $('#btn-answer').onclick = () => send({ type: 'answer_now' });
  $('#btn-improve').onclick = () => send({ type: 'improve' });
  $('#btn-new').onclick = () => {
    if (confirm('Save this meeting and start a fresh one?')) send({ type: 'new_meeting' });
  };
  $('#btn-summary').onclick = () => downloadSummary(state.session).catch((err) => toast(err.message, 'error'));
  $('#auto').onchange = (e) => send({ type: 'auto', value: e.target.checked });

  $('#ask-form').onsubmit = (e) => {
    e.preventDefault();
    const text = $('#ask-input').value.trim();
    if (!text) return;
    send({ type: 'ask', text });
    $('#ask-input').value = '';
  };
  document.addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea, select') || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === 'Space') {
      e.preventDefault();
      send({ type: 'answer_now' });
    } else if (e.key === 'i' || e.key === 'I') {
      send({ type: 'improve' });
    } else if (e.key === 'd' || e.key === 'D') {
      send({ type: 'detail' });
    }
  });

  // Background profiles
  $('#profile-select').onchange = (e) => loadProfile(e.target.value).catch((err) => toast(err.message, 'error'));
  $('#btn-new-profile').onclick = async () => {
    const name = prompt('Name for the new profile, e.g. "Acme interview"');
    if (!name) return;
    try {
      const created = await api('/api/profiles', { method: 'POST', json: { name } });
      await loadProfiles(created.slug);
    } catch (err) {
      toast(err.message, 'error');
    }
  };
  $('#file-input').onchange = async (e) => {
    await uploadFiles([...e.target.files]);
    e.target.value = '';
  };
  $('#doc-list').onchange = async (e) => {
    const name = e.target.dataset.kindFor;
    if (!name) return;
    try {
      const result = await api(`/api/profiles/${encodeURIComponent(currentSlug)}/docs/${encodeURIComponent(name)}`,
        { method: 'PATCH', json: { kind: e.target.value } });
      renderDocs(result.docs);
      $('#prepare-status').textContent = 'Type changed. Press Save & prepare to use it.';
    } catch (err) {
      toast(err.message, 'error');
    }
  };
  $('#doc-list').onclick = async (e) => {
    const name = e.target.closest('button')?.dataset.doc;
    if (!name) return;
    try {
      const result = await api(`/api/profiles/${encodeURIComponent(currentSlug)}/docs/${encodeURIComponent(name)}`, { method: 'DELETE' });
      renderDocs(result.docs);
    } catch (err) {
      toast(err.message, 'error');
    }
  };
  $('#btn-prepare').onclick = prepare;
  $('#f-base').onchange = async () => {
    // Saved right away so the documents it brings in show up in the list.
    try {
      applyProfile(await api(`/api/profiles/${encodeURIComponent(currentSlug)}`, { method: 'PUT', json: profileFields() }));
      $('#prepare-status').textContent = 'Press Save & prepare to use it.';
    } catch (err) {
      toast(err.message, 'error');
    }
  };
  $('#setup').addEventListener('toggle', () => {
    if ($('#setup').open) loadSessions().catch((err) => toast(err.message, 'error'));
  });
  $('#btn-import').onclick = importMeeting;
  $('#btn-download').onclick = () => downloadSummary($('#session-select').value).catch((err) => toast(err.message, 'error'));

  // Capture (only where the browser allows it: the meeting computer on localhost)
  if (!canCapture) {
    $('#capture-controls').hidden = true;
    $('#capture-note').hidden = false;
  } else {
    const savedMode = store('ga-mode');
    if (savedMode) $('#mode').value = savedMode;
    $('#mode').onchange = (e) => {
      store('ga-mode', e.target.value);
      renderListenButton();
    };
    $('#them-source').onchange = (e) => store('ga-them', e.target.value);
    $('#btn-listen').onclick = () => (capture.active ? stopListening() : startListening());
    navigator.mediaDevices.addEventListener?.('devicechange', refreshDevices);
    refreshDevices().then(() => {
      const savedThem = store('ga-them');
      if (savedThem && [...$('#them-source').options].some((o) => o.value === savedThem)) $('#them-source').value = savedThem;
    });
    renderListenButton();
    requestAnimationFrame(animateMeters);
  }
}

wire();
connectUi();
loadProfiles().catch((err) => toast(err.message, 'error'));
