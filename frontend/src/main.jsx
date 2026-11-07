import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDownLeft, ArrowRight, ArrowUpRight, Bookmark, BookOpen, Check, CheckCircle2, ChevronDown, CircleHelp, Clock3, FileText, FolderOpen, Gauge, Layers3, Library, LockKeyhole, Menu, MessageSquare, MoreHorizontal, Play, Plus, Search, ShieldCheck, Sparkles, Trash2, Upload, Video, X } from 'lucide-react';
import { uploadLecture } from './upload.js';
import './style.css';

const formatTime = value => {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = response.status === 204 ? {} : await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const json = (method, value) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });

function UploadDialog({ onClose, onImported, notify }) {
  const [video, setVideo] = useState(null);
  const [subtitle, setSubtitle] = useState(null);
  const [permission, setPermission] = useState(false);
  const [storage, setStorage] = useState(false);
  const [language, setLanguage] = useState('en');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState('');
  const dialogRef = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    const dialog = dialogRef.current;
    dialog?.querySelector('button')?.focus();
    const onKeyDown = event => {
      if (event.key === 'Escape' && !busy) { event.preventDefault(); onClose(); }
      if (event.key !== 'Tab' || !dialog) return;
      const controls = [...dialog.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled)')];
      if (!controls.length) return;
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => { document.removeEventListener('keydown', onKeyDown); previous?.focus?.(); };
  }, [busy, onClose]);
  const submit = async event => {
    event.preventDefault();
    if (!video || !permission || !storage) return;
    if (video.size > 2 * 1024 * 1024 * 1024) { setError('Videos must be no larger than 2 GiB.'); return; }
    const data = new FormData();
    data.append('video', video); if (subtitle) data.append('subtitle', subtitle);
    data.append('permission', 'yes'); data.append('local_storage', 'yes'); data.append('language', language);
    setBusy(true); setError(''); setProgress(null);
    try {
      const result = await uploadLecture(data, setProgress);
      try { await onImported(result); notify(result.state === 'ready' ? 'Video added to your local library.' : 'Local transcription started.'); }
      catch (_) { notify('Import was accepted, but the library could not refresh. Reload the page before importing again.'); }
      onClose();
    } catch (err) { setError(err.message); } finally { setBusy(false); setProgress(null); }
  };
  return <div className="modal-layer" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose(); }}>
    <section ref={dialogRef} className="modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
      <div className="modal-heading"><div><div className="eyebrow">LOCAL IMPORT <span> / 01</span></div><h2 id="upload-title">Add a lecture</h2><p>Bring your own authorized video into your private workspace.</p></div><button className="icon-button" onClick={onClose} disabled={busy} aria-label="Close"><X size={19}/></button></div>
      <form onSubmit={submit}>
        <label className="dropzone"><input type="file" accept=".mp4,.webm,.ogv,.mov,.m4v" onChange={e => setVideo(e.target.files[0] || null)} required/><span className="drop-icon"><Upload size={21}/></span><strong>{video ? video.name : 'Choose a lecture video'}</strong><small>MP4, WebM, OGV, MOV or M4V · up to 2 GiB</small></label>
        <div className="form-row"><label className="file-field"><span>Timed subtitles <em>optional</em></span><input type="file" accept=".srt,.vtt" onChange={e => setSubtitle(e.target.files[0] || null)}/><small>{subtitle ? subtitle.name : 'SRT or VTT · up to 10 MiB'}</small></label><label className="select-field"><span>Lecture language</span><select value={language} onChange={e => setLanguage(e.target.value)}><option value="en">English</option><option value="zh">中文</option><option value="">Auto detect</option></select></label></div>
        <div className="privacy-box"><div className="privacy-title"><ShieldCheck size={17}/> Before you import</div><p>Your browser sends files to the RewindAI server on this computer. Video copies, transcripts, indexes and notes are stored locally; the browser or operating system may also create temporary copies. This app does not send lecture content to a hosted AI service. Restricted material must be kept out of shared or cloud-synced folders and backups.</p><p>Removing an index does not delete the uploaded video. Failed or cancelled transcription can leave local copies. Qwen3 is pre-trained: importing a video creates an index, not new model weights. Without subtitles, local transcription requires a cached Whisper model and may take time.</p>
          <label className="check"><input type="checkbox" checked={permission} onChange={e => setPermission(e.target.checked)}/><span>I have permission to process this recording and subtitles under the applicable course and institutional rules.</span></label>
          <label className="check"><input type="checkbox" checked={storage} onChange={e => setStorage(e.target.checked)}/><span>I understand local copies and derived data will be stored on this computer.</span></label>
        </div>
        {busy && <div className="upload-progress" role="status"><span>{progress === null ? 'Preparing local upload…' : progress < 100 ? `Copying to this computer… ${progress}%` : 'Copy complete. Processing locally; please keep this window open…'}</span>{progress !== null && <progress value={progress} max="100" aria-label="Local upload progress"/>}<small>After copying, indexing may take additional time.</small></div>}
        {error && <div className="error" role="alert">{error}</div>}
        <div className="modal-actions"><button type="button" className="btn quiet" onClick={onClose} disabled={busy}>Cancel</button><button className="btn primary" disabled={!video || !permission || !storage || busy}>{busy ? 'Importing locally…' : 'Import to workspace'} <ArrowRight size={16}/></button></div>
      </form>
    </section>
  </div>;
}

function SubtitleDialog({ video, onClose, onUpdated, notify }) {
  const [subtitle, setSubtitle] = useState(null), [permission, setPermission] = useState(false);
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  const dialogRef = useRef(null);
  useEffect(() => {
    const previous = document.activeElement, dialog = dialogRef.current;
    dialog?.querySelector('button')?.focus();
    const onKeyDown = event => {
      if (event.key === 'Escape' && !busy) { event.preventDefault(); onClose(); }
      if (event.key !== 'Tab' || !dialog) return;
      const controls = [...dialog.querySelectorAll('button:not(:disabled), input:not(:disabled)')];
      if (!controls.length) return;
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => { document.removeEventListener('keydown', onKeyDown); previous?.focus?.(); };
  }, [busy, onClose]);
  async function submit(event) {
    event.preventDefault();
    if (!subtitle || !permission || busy) return;
    if (subtitle.size > 10 * 1024 * 1024) { setError('Subtitles must be no larger than 10 MiB.'); return; }
    const data = new FormData(); data.append('subtitle', subtitle); data.append('permission', 'yes');
    setBusy(true); setError('');
    try {
      await api(`/api/library/videos/${encodeURIComponent(video.id)}/subtitle-upload`, { method: 'POST', body: data });
      try { await onUpdated(); notify('Subtitles updated. Search again to see the new evidence.'); }
      catch (_) { notify('Subtitles were updated, but the library could not refresh. Reload the page.'); }
      onClose();
    } catch (err) { setError(`${err.message} If the connection was interrupted, reload and check before retrying.`); }
    finally { setBusy(false); }
  }
  return <div className="modal-layer" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose(); }}>
    <section ref={dialogRef} className="modal feedback-modal" role="dialog" aria-modal="true" aria-labelledby="subtitle-title">
      <div className="modal-heading"><div><div className="eyebrow">LOCAL SUBTITLES</div><h2 id="subtitle-title">Update subtitles</h2><p>{video.title}</p></div><button className="icon-button" onClick={onClose} disabled={busy} aria-label="Close"><X size={19}/></button></div>
      <form onSubmit={submit}><label className="file-field"><span>Corrected timed subtitles</span><input type="file" accept=".srt,.vtt" onChange={event => setSubtitle(event.target.files[0] || null)} required/><small>SRT or VTT · up to 10 MiB</small></label><div className="privacy-box"><p>The server rebuilds the local index from this file. Notes remain saved; bookmarks with changed time ranges may no longer be playable. The original video and subtitle files are not changed. The browser and system may create temporary copies.</p><label className="check"><input type="checkbox" checked={permission} onChange={event => setPermission(event.target.checked)}/><span>I have permission to process these subtitles on this computer.</span></label></div>{error && <div className="error" role="alert">{error}</div>}<div className="modal-actions"><button type="button" className="btn quiet" onClick={onClose} disabled={busy}>Cancel</button><button className="btn primary" disabled={!subtitle || !permission || busy}>{busy ? 'Updating locally…' : 'Update index'} <ArrowRight size={16}/></button></div></form>
    </section>
  </div>;
}

function FeedbackDialog({ feedback, onClose, onSaved, notify }) {
  const [comment, setComment] = useState(''), [saving, setSaving] = useState(false), [error, setError] = useState('');
  const dialogRef = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    const dialog = dialogRef.current;
    dialog?.querySelector('textarea')?.focus();
    const onKeyDown = event => {
      if (event.key === 'Escape' && !saving) { event.preventDefault(); onClose(); }
      if (event.key !== 'Tab' || !dialog) return;
      const controls = [...dialog.querySelectorAll('button:not(:disabled), textarea:not(:disabled)')];
      if (!controls.length) return;
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => { document.removeEventListener('keydown', onKeyDown); previous?.focus?.(); };
  }, [saving, onClose]);
  async function submit(event) {
    event.preventDefault();
    if (saving) return;
    setSaving(true); setError('');
    try {
      await api('/api/feedback', json('POST', { ...feedback, comment }));
      onSaved(); notify('Feedback saved locally.'); onClose();
    } catch (err) { setError(err.message); }
    finally { setSaving(false); }
  }
  return <div className="modal-layer" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget && !saving) onClose(); }}>
    <section ref={dialogRef} className="modal feedback-modal" role="dialog" aria-modal="true" aria-labelledby="feedback-title">
      <div className="modal-heading"><div><div className="eyebrow">LOCAL FEEDBACK</div><h2 id="feedback-title">Share your feedback</h2><p>{feedback.target_type === 'answer' ? 'Response' : feedback.target_type === 'visual' ? 'Visual candidate' : 'Transcript evidence'} · {feedback.rating === 'helpful' ? 'Useful' : 'Not useful'}</p></div><button className="icon-button" onClick={onClose} disabled={saving} aria-label="Close"><X size={19}/></button></div>
      <form onSubmit={submit}><label className="feedback-label">Add a note <span>Optional</span><textarea maxLength={4000} value={comment} onChange={event => setComment(event.target.value)} placeholder="Add a brief note…"/></label><p className="feedback-privacy"><LockKeyhole size={13}/> Your question, rating and note are stored locally.</p>{error && <div className="error" role="alert">{error}</div>}<div className="modal-actions"><button type="button" className="btn quiet" disabled={saving} onClick={onClose}>Cancel</button><button className="btn primary" disabled={saving}>{saving ? 'Saving…' : 'Save feedback'} <ArrowRight size={16}/></button></div></form>
    </section>
  </div>;
}

function EvidenceContext({ segmentId, onSeek }) {
  const [open, setOpen] = useState(false), [items, setItems] = useState(null), [error, setError] = useState('');
  useEffect(() => {
    if (!open) return undefined;
    let active = true;
    setItems(null); setError('');
    api(`/api/segments/${encodeURIComponent(segmentId)}/context`)
      .then(data => { if (active) setItems(data.context || []); })
      .catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [open, segmentId]);
  return <div className="evidence-context"><button type="button" className="context-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? 'Hide surrounding subtitles' : 'View surrounding subtitles'} <ChevronDown size={13}/></button>{open && <div className="context-content"><small>For context only; these excerpts may not have been supplied to the answer model.</small>{error ? <p className="context-error" role="alert">{error}</p> : items === null ? <p>Loading local context…</p> : items.length ? items.map(item => <div className="context-item" key={item.segment_id}><button type="button" onClick={() => onSeek(item)}>{item.relation === 'previous' ? 'Previous' : 'Next'} · {formatTime(item.start_time)}–{formatTime(item.end_time)}</button><p>{item.transcript}</p></div>) : <p>No neighboring segments are available.</p>}</div>}</div>;
}

function FeedbackHistory({ revision, notify }) {
  const [open, setOpen] = useState(false), [items, setItems] = useState(null), [error, setError] = useState('');
  const [deleting, setDeleting] = useState(null);
  useEffect(() => {
    if (!open) return undefined;
    let active = true;
    setError(''); setItems(null);
    api('/api/feedback').then(data => { if (active) setItems(data.feedback || []); })
      .catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [open, revision]);
  async function remove(item) {
    if (deleting !== null || !window.confirm('Delete this locally saved feedback?')) return;
    setDeleting(item.id);
    try { await api(`/api/feedback/${item.id}`, { method: 'DELETE' }); setItems(previous => previous.filter(row => row.id !== item.id)); notify('Feedback deleted locally.'); }
    catch (err) { notify(err.message); }
    finally { setDeleting(null); }
  }
  return <section className="panel feedback-history"><div className="panel-head"><div><div className="eyebrow">REFLECT / 06</div><h2>Feedback history</h2></div><MessageSquare size={17}/></div><p className="panel-desc">Your recent ratings and notes, stored locally.</p><button className="history-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Hide feedback' : 'View feedback'} <ChevronDown size={14}/></button>{open && <div className="feedback-list">{error ? <div className="error" role="alert">{error}</div> : items === null ? <p>Loading feedback…</p> : items.length === 0 ? <p>No feedback saved yet.</p> : items.map(item => <article className="feedback-record" key={item.id}><div className="saved-meta"><span>{item.target_type === 'visual' ? 'FRAME' : item.target_type === 'transcript' ? 'TRANSCRIPT' : item.target_type === 'answer' ? 'RESPONSE' : 'EVIDENCE'} · {item.rating === 'helpful' ? 'USEFUL' : 'NOT USEFUL'}</span><button type="button" disabled={deleting !== null} onClick={() => remove(item)} aria-label="Delete feedback"><Trash2 size={13}/></button></div><strong>{item.question}</strong>{item.comment && <p>{item.comment}</p>}</article>)}{items?.length === 100 && <small>Showing the 100 most recent entries.</small>}</div>}<div className="panel-footer"><span>RECENT LOCAL FEEDBACK</span><span><LockKeyhole size={11}/> LOCAL</span></div></section>;
}

function App() {
  const [health, setHealth] = useState(null), [library, setLibrary] = useState({ videos: [], courses: [] });
  const [selectedId, setSelectedId] = useState(null), [bookmarks, setBookmarks] = useState([]);
  const [question, setQuestion] = useState(''), [scope, setScope] = useState('video'), [result, setResult] = useState(null), [searchMode, setSearchMode] = useState('answer');
  const [lastQuestion, setLastQuestion] = useState(''), [busy, setBusy] = useState(false), [uploadOpen, setUploadOpen] = useState(false);
  const [toast, setToast] = useState(''), [transcription, setTranscription] = useState(null), [courseName, setCourseName] = useState('');
  const [noteDrafts, setNoteDrafts] = useState({}), [savingNotes, setSavingNotes] = useState({}), [menuOpen, setMenuOpen] = useState(false), [removingVideo, setRemovingVideo] = useState(false);
  const [feedbackDraft, setFeedbackDraft] = useState(null);
  const [feedbackRevision, setFeedbackRevision] = useState(0);
  const [subtitleTarget, setSubtitleTarget] = useState(null);
  const playerRef = useRef(null), pendingTime = useRef(null), toastTimer = useRef(null);
  const selected = library.videos.find(v => v.id === selectedId) || null;
  const notify = message => { setToast(message); clearTimeout(toastTimer.current); toastTimer.current = setTimeout(() => setToast(''), 4500); };
  async function load() {
    const [h, l, b] = await Promise.all([api('/api/health'), api('/api/library'), api('/api/bookmarks')]);
    setHealth(h); setLibrary(l); setBookmarks(b.bookmarks || []);
    setSelectedId(previous => l.videos.some(v => v.id === previous) ? previous : (l.videos[0]?.id || null));
  }
  useEffect(() => { load().catch(err => notify(`Could not load workspace: ${err.message}`)); return () => clearTimeout(toastTimer.current); }, []);
  useEffect(() => {
    const warn = event => { if (Object.keys(noteDrafts).length) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', warn); return () => window.removeEventListener('beforeunload', warn);
  }, [noteDrafts]);
  useEffect(() => {
    if (!health || health.mode !== 'transcript') return undefined;
    let active = true, timer;
    const poll = async () => { try { const data = await api('/api/transcription'); if (!active) return; setTranscription(previous => { if (data.state === 'complete' && previous?.state !== 'complete') load().catch(() => {}); return data; }); } catch (_) { /* no running local job */ } if (active) timer = setTimeout(poll, 4000); };
    poll(); return () => { active = false; clearTimeout(timer); };
  }, [health?.mode]);
  function seek(item) {
    const video = library.videos.find(v => v.id === item.video_id);
    if (!video || item.available === false) { notify('This video or saved segment is not currently available.'); return; }
    pendingTime.current = Number(item.start_time) || 0;
    if (selectedId === video.id && playerRef.current) {
      if (Number.isFinite(playerRef.current.duration) && pendingTime.current > playerRef.current.duration) { pendingTime.current = null; notify('This timestamp exceeds the video duration.'); return; }
      playerRef.current.currentTime = pendingTime.current; pendingTime.current = null; playerRef.current.play().catch(() => {});
    }
    else setSelectedId(video.id);
    document.getElementById('player')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  async function ask(event) {
    event.preventDefault(); const q = question.trim(); if (!q) return;
    if (scope !== 'all' && !selected) { notify('Select a video first, or search all videos.'); return; }
    const body = { question: q };
    if (scope === 'video') body.video_id = selected.id;
    if (scope === 'course') { if (selected.course_id == null) { notify('Assign the selected video to a course first.'); return; } body.course_id = selected.course_id; }
    setBusy(true); setResult(null); setLastQuestion(q);
    try { const data = await api(searchMode === 'visual' ? '/api/visual-search' : searchMode === 'hybrid' ? '/api/hybrid-search' : '/api/ask', json('POST', body)); setResult({ ...data, searchMode }); } catch (err) { notify(err.message); } finally { setBusy(false); }
  }
  async function saveEvidence(item) { try { await api('/api/bookmarks', json('POST', { video_id: item.video_id, segment_id: item.segment_id, note: '' })); setBookmarks((await api('/api/bookmarks')).bookmarks); notify('Moment saved for review.'); } catch (err) { notify(err.message); } }
  function feedback(item, rating, source = null) { setFeedbackDraft({ question: lastQuestion, segment_id: item?.segment_id || null, rating, target_type: item ? (source || (result?.searchMode === 'visual' ? 'visual' : 'transcript')) : 'answer' }); }
  async function createCourse(event) { event.preventDefault(); if (!courseName.trim()) return; try { await api('/api/courses', json('POST', { name: courseName.trim() })); setCourseName(''); await load(); notify('Course created.'); } catch (err) { notify(err.message); } }
  async function assignCourse(value) { if (!selected) return; try { await api(`/api/videos/${encodeURIComponent(selected.id)}/course`, json('PUT', { course_id: value ? Number(value) : null })); await load(); } catch (err) { notify(err.message); } }
  async function removeFromLibrary() {
    if (!selected || removingVideo || busy || busyTranscribing) return;
    if (!window.confirm(`Remove “${selected.title}” from the searchable library? Its uploaded video, subtitles, saved moments and notes remain on this computer. Delete local copies separately if needed.`)) return;
    setRemovingVideo(true);
    try {
      await api(`/api/library/videos/${encodeURIComponent(selected.id)}`, { method: 'DELETE' });
      setResult(null);
      try { await load(); notify('Lecture removed from the index. Local copies and saved moments remain.'); }
      catch (_) { notify('Index removed, but the library could not refresh. Reload the page.'); }
    } catch (err) { notify(err.message); }
    finally { setRemovingVideo(false); }
  }
  async function updateBookmark(item, operation) { const submittedNote = noteDrafts[item.id] ?? item.note; if (operation === 'note') setSavingNotes(previous => ({ ...previous, [item.id]: true })); try {
    if (operation === 'note') await api(`/api/bookmarks/${item.id}`, json('PATCH', { note: submittedNote }));
    if (operation === 'review') await api(`/api/bookmarks/${item.id}/reviewed`, json('PUT', { reviewed: !item.reviewed }));
    if (operation === 'delete') { if (!window.confirm('Remove this saved moment and its note? The original video is kept.')) return; await api(`/api/bookmarks/${item.id}`, { method: 'DELETE' }); }
    setBookmarks((await api('/api/bookmarks')).bookmarks); if (operation === 'note' || operation === 'delete') setNoteDrafts(previous => { const next = { ...previous }; if (operation === 'delete' || previous[item.id] === submittedNote) delete next[item.id]; return next; });
    notify(operation === 'delete' ? 'Saved moment removed.' : 'Review list updated.');
  } catch (err) { notify(err.message); } finally { if (operation === 'note') setSavingNotes(previous => { const next = { ...previous }; delete next[item.id]; return next; }); } }
  const modelLabel = health?.retrieval_type === 'semantic_experimental' ? 'QWEN3 SEMANTIC' : health ? 'KEYWORD SEARCH' : 'CONNECTING';
  const busyTranscribing = ['transcribing', 'cancelling', 'indexing'].includes(transcription?.state);
  return <div className={`app ${menuOpen ? 'menu-open' : ''}`}>
    <aside className="sidebar"><div className="brand"><div className="brandmark"><span/><span/><span/></div><div><strong>REWIND<span>AI</span></strong><small>LEARNING WORKSPACE</small></div></div>
      <div className="sidebar-section"><div className="side-label">WORKSPACE <span>01</span></div><a className="nav-item active" href="#workspace" onClick={() => setMenuOpen(false)}><Layers3 size={17}/> Overview <span className="nav-arrow">↗</span></a><a className="nav-item" href="#library" onClick={() => setMenuOpen(false)}><Library size={17}/> My library <span className="nav-count">{String(library.videos.length).padStart(2, '0')}</span></a><a className="nav-item" href="#saved" onClick={() => setMenuOpen(false)}><Bookmark size={17}/> Saved moments <span className="nav-count">{String(bookmarks.length).padStart(2, '0')}</span></a></div>
      <div className="sidebar-section course-nav"><div className="side-label">COURSES <span>{String(library.courses.length).padStart(2, '0')}</span></div>{library.courses.length ? library.courses.map(course => <div className="course-line" key={course.id}><span className="course-dot"/>{course.name}</div>) : <p className="side-empty">Organize your lectures into courses.</p>}<form className="course-create" onSubmit={createCourse}><input aria-label="New course name" maxLength="120" placeholder="New course" value={courseName} onChange={e => setCourseName(e.target.value)}/><button type="submit" aria-label="Create course"><Plus size={16}/></button></form></div>
      <div className="side-bottom"><div className="secure-icon"><LockKeyhole size={16}/></div><div><strong>LOCAL WORKSPACE</strong><small>Your content stays on this device.</small></div><span className="secure-dot"/></div>
    </aside>
    <div className="main-shell" id="workspace"><header className="topbar"><div className="top-left"><button className="icon-button mobile-menu" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation"><Menu size={20}/></button><span className="crumb-muted">WORKSPACE</span><span className="crumb-slash">/</span><strong>Overview</strong></div><div className="top-right"><span className="status-pill"><span className={health ? 'status-dot' : 'status-dot offline'}/>{health ? 'LOCAL SERVER ONLINE' : 'CONNECTING'}</span><a className="help-link" href="/classic" title="Open the original workspace"><CircleHelp size={17}/><span>Classic view</span></a><div className="profile">R</div></div></header>
      <main className="content"><div className="page-head"><div><div className="eyebrow">YOUR LEARNING SPACE <span>/ OVERVIEW</span></div><h1>Pick up where<br/><span>you left off.</span></h1><p>Find the moment, understand the context, and make it stick.</p></div><div className="head-actions"><div className="date-stamp"><span>YOUR WORKSPACE</span><strong>Built for deeper learning</strong></div><button className="btn primary add-btn" onClick={() => setUploadOpen(true)} disabled={health?.mode !== 'transcript'}><Plus size={17}/> Add lecture <ArrowUpRight size={15}/></button></div></div>
      <div className="stat-strip"><div className="stat"><span className="stat-icon"><Video size={17}/></span><div><small>IN YOUR LIBRARY</small><strong>{String(library.videos.length).padStart(2, '0')} <em>lectures</em></strong></div><span className="stat-index">01</span></div><div className="stat"><span className="stat-icon"><Bookmark size={17}/></span><div><small>SAVED FOR LATER</small><strong>{String(bookmarks.length).padStart(2, '0')} <em>moments</em></strong></div><span className="stat-index">02</span></div><div className="stat"><span className="stat-icon"><Activity size={17}/></span><div><small>RETRIEVAL MODE</small><strong className="stat-mode">{modelLabel}</strong></div><span className="stat-index">03</span></div><div className="stat stat-private"><span className="stat-icon"><ShieldCheck size={17}/></span><div><small>PROCESSING</small><strong className="stat-mode">LOCAL ONLY</strong></div><span className="stat-index">04</span></div></div>
      {busyTranscribing && <div className="job-banner"><Activity size={16}/><span>{transcription?.state === 'indexing' ? 'Transcription complete. Building the local index…' : transcription?.state === 'cancelling' ? 'Stopping local transcription…' : 'Transcribing on this computer. Longer recordings can take time.'}</span><button onClick={async () => { try { await api('/api/transcription/cancel', { method: 'POST' }); } catch (err) { notify(err.message); } }}>Cancel</button></div>}
      {transcription?.state === 'failed' && <div className="error job-banner">Local transcription failed. Check that the Whisper base model is cached, the video has an audio track, and disk space is available. The uploaded copy remains in artifacts/uploads.</div>}
      <div className="workspace-grid"><section className="workspace-main"><div className="section-top"><div><div className="eyebrow">LEARN / 01</div><h2>Lecture workspace</h2></div><span className="section-note"><span className="tiny-dot"/> {selected ? 'READY TO EXPLORE' : 'AWAITING A LECTURE'}</span></div>
        <div className="player-card" id="player"><div className="player-top"><div><span className="player-kicker">NOW VIEWING</span><h3>{selected?.title || 'Your next insight starts here'}</h3><p>{selected ? `${selected.segment_count} indexed ${selected.segment_count === 1 ? "segment" : "segments"}` : 'Add a lecture to begin your learning session.'}</p></div><span className="player-count">{selected ? '01 / 01' : '— / —'}</span></div><div className="video-area">{selected?.available ? <video ref={playerRef} key={selected.id} controls preload="metadata" src={selected.video_url} onLoadedMetadata={() => { if (pendingTime.current != null && playerRef.current) { if (pendingTime.current <= playerRef.current.duration) { playerRef.current.currentTime = pendingTime.current; playerRef.current.play().catch(() => {}); } else notify('This timestamp exceeds the video duration.'); pendingTime.current = null; } }}/> : <div className="video-empty"><div className="video-empty-mark"><Play size={21} fill="currentColor"/></div><strong>NO LECTURE SELECTED</strong><span>Your video will appear here after import.</span><button onClick={() => setUploadOpen(true)} disabled={health?.mode !== 'transcript'}>ADD YOUR FIRST LECTURE <ArrowUpRight size={13}/></button></div>}</div><div className="player-footer"><div><span className="orange-square"/> {selected ? 'LOCAL VIDEO' : 'PRIVATE BY DESIGN'}</div><span><LockKeyhole size={12}/> ON THIS DEVICE</span></div></div>
        <div className="ask-card"><div className="card-head"><div><div className="eyebrow">ASK YOUR LECTURE <span>/ 02</span></div><h3>Find an answer.</h3><p>Ask a question or search for a concept in your lecture.</p></div><span className="ask-symbol"><Sparkles size={19}/></span></div>{health?.visual_enabled && <div className="search-modes" role="group" aria-label="Search mode"><button type="button" aria-pressed={searchMode === "answer"} disabled={busy} onClick={() => { setSearchMode("answer"); setResult(null); }}>Lecture question</button><button type="button" aria-pressed={searchMode === "visual"} disabled={busy} onClick={() => { setSearchMode("visual"); setResult(null); }}>Find a frame <span>EXPERIMENTAL</span></button><button type="button" aria-pressed={searchMode === "hybrid"} disabled={busy} onClick={() => { setSearchMode("hybrid"); setResult(null); }}>Text + frames <span>EXPERIMENTAL</span></button></div>}<form onSubmit={ask} className="ask-form"><Search size={19}/><input value={question} onChange={e => setQuestion(e.target.value)} placeholder={searchMode === "visual" ? "Describe a frame in English…" : searchMode === "hybrid" ? "Describe a concept or frame in English…" : "What would you like to understand?"} aria-label="Ask about your lecture" maxLength={searchMode !== "answer" || health?.answer_mode === 'local_experimental' ? 1000 : 2000}/><select aria-label="Search scope" value={scope} onChange={e => setScope(e.target.value)}><option value="video">This lecture</option><option value="course">This course</option><option value="all">All lectures</option></select><button type="submit" className="ask-submit" disabled={busy || !question.trim()}>{busy ? 'SEARCHING' : searchMode !== 'answer' ? 'FIND' : 'ASK'} <ArrowRight size={16}/></button></form><div className="ask-caption"><span><span className="tiny-dot"/> {searchMode === 'visual' ? 'LOCAL FRAME SIMILARITY SEARCH' : searchMode === 'hybrid' ? 'LOCAL TEXT + FRAME SEARCH' : health?.answer_mode === 'local_experimental' ? 'LOCAL AI ANSWER + EVIDENCE' : 'SUBTITLE EVIDENCE SEARCH'}</span><span>Always verify against the recording</span></div>
          {result?.searchMode === "visual" && <div className="result-warning visual-warning"><strong>Experimental frame matches</strong><p>Similar frames are candidates, not verified answers; unrelated results are possible. Searched {result.indexed_segments} of {result.source_segments} segment midpoints. Other moments may be missed.</p></div>}
          {result?.searchMode === "hybrid" && <div className="result-warning visual-warning"><strong>Experimental fused candidates · no answer generated</strong><p>Text and frame rankings are combined; verify every candidate. {result.visual_available ? `Searched ${result.indexed_segments} of ${result.source_segments} sampled segment midpoints.` : "Frame search failed; only transcript candidates are shown."} {result.visual_error || ""}</p></div>}
          {result && result.searchMode === "answer" && <div className="answer-result"><div className="result-head"><span><Sparkles size={15}/> {result.generation_error ? 'ANSWER UNAVAILABLE' : health?.answer_mode === 'local_experimental' ? 'AI RESPONSE' : 'SEARCH RESULT'}</span><span>LOCAL / {modelLabel}</span></div><p className="answer-text">{result.generation_error ? `No answer was generated. ${result.generation_error}` : health?.answer_mode === 'local_experimental' ? (result.answer || 'No answer was generated.') : (result.evidence?.[0]?.transcript || 'No matching transcript excerpt was found.') }</p>{(result.abstained || result.exact_model_refusal) && <p className="result-warning">{result.abstained ? 'No matching evidence was retrieved. This does not prove the video has no answer.' : 'The model reported insufficient evidence. Review the candidate excerpts below.'}</p>}{!result.generation_error && <div className="result-feedback"><span>Was this useful?</span><button onClick={() => feedback(null, 'helpful')}>Yes</button><button onClick={() => feedback(null, 'not_helpful')}>No</button></div>}</div>}
        </div>
        {result && <div className="evidence-section"><div className="section-top compact"><div><div className="eyebrow">TRACE THE SOURCE / 03</div><h2>Candidate evidence <span className="count-pill">{String((result.evidence || []).length).padStart(2, '0')}</span></h2></div><small>VERIFY AGAINST VIDEO</small></div>{!(result.evidence || []).length ? <div className="empty-panel">No candidates were returned. Try a different search.</div> : result.evidence.map((item, i) => <article className="evidence-card" key={item.segment_id}><div className="evidence-number">{String(i + 1).padStart(2, '0')}</div><div className="evidence-body"><div className="evidence-source"><Video size={13}/> {library.videos.find(video => video.id === item.video_id)?.title || "Unavailable lecture"}</div><div className="evidence-meta"><span><span className="tiny-dot"/> {result.searchMode === "visual" ? "SAMPLED FRAME" : result.searchMode === "hybrid" ? item.sources.map(source => source === "visual" ? "FRAME" : "TRANSCRIPT").join(" + ") : "TRANSCRIPT EXCERPT"}</span><button onClick={() => seek(item)}><Play size={12} fill="currentColor"/> {formatTime(item.start_time)} — {formatTime(item.end_time)}</button></div>{(result.searchMode === "visual" || (result.searchMode === "hybrid" && item.sources.includes("visual"))) && <figure className="frame-preview"><img loading="lazy" src={`/api/segments/${encodeURIComponent(item.segment_id)}/frame`} alt="Sampled midpoint frame; verify against the video"/><figcaption>Sampled midpoint · verify the full segment</figcaption></figure>}{result.searchMode !== "visual" && item.transcript && <><p>{item.transcript}</p>{health?.mode === "transcript" && <EvidenceContext segmentId={item.segment_id} onSeek={seek}/>}</>}{Object.hasOwn(item, 'model_excerpt') && <details><summary>What the answer model received</summary><p>{item.model_excerpt ?? 'This excerpt was not sent to the model.'}</p></details>}<div className="evidence-actions"><button onClick={() => seek(item)}>Watch moment <ArrowUpRight size={14}/></button><button onClick={() => saveEvidence(item)}><Bookmark size={14}/> Save</button>{result.searchMode === "hybrid" ? item.sources.map(source => <React.Fragment key={source}><button onClick={() => feedback(item, 'helpful', source)}>{source === "visual" ? "Frame" : "Text"} useful</button><button onClick={() => feedback(item, 'not_helpful', source)}>{source === "visual" ? "Frame" : "Text"} not useful</button></React.Fragment>) : <><button onClick={() => feedback(item, 'helpful')}>Useful</button><button onClick={() => feedback(item, 'not_helpful')}>Not useful</button></>}</div></div></article>)}</div>}
      </section>
      <aside className="right-column"><section className="panel library-panel" id="library"><div className="panel-head"><div><div className="eyebrow">YOUR CONTENT / 04</div><h2>My library</h2></div><button className="icon-button boxed" onClick={() => setUploadOpen(true)} aria-label="Add lecture" disabled={health?.mode !== 'transcript'}><Plus size={18}/></button></div><p className="panel-desc">Your lectures, ready when you are.</p><div className="library-list">{library.videos.length ? library.videos.map((video, i) => <button key={video.id} className={`lecture-row ${selectedId === video.id ? 'selected' : ''}`} onClick={() => { setSelectedId(video.id); setResult(null); }}><span className="lecture-thumbnail"><Video size={18}/><i>{String(i + 1).padStart(2, '0')}</i></span><span className="lecture-info"><strong title={video.title}>{video.title}</strong><small>{video.available ? `${video.segment_count} ${video.segment_count === 1 ? "segment" : "segments"}` : 'Source unavailable'} · index ends {formatTime(video.duration)}</small></span><ArrowUpRight size={15}/></button>) : <div className="library-empty"><FolderOpen size={22}/><strong>No lectures yet</strong><span>Import a video to start building your library.</span><button onClick={() => setUploadOpen(true)} disabled={health?.mode !== 'transcript'}>Add a lecture <ArrowRight size={13}/></button></div>}</div>{selected && <label className="assign-course">COURSE <select value={selected.course_id ?? ''} onChange={e => assignCourse(e.target.value)}><option value="">Unassigned</option>{library.courses.map(course => <option key={course.id} value={course.id}>{course.name}</option>)}</select></label>}{selected && health?.mode === "transcript" && <button className="update-subtitles" disabled={busy || busyTranscribing || removingVideo} onClick={() => setSubtitleTarget(selected)}><FileText size={13}/> Update subtitles</button>}{selected && health?.mode === "transcript" && <button className="remove-index" onClick={removeFromLibrary} disabled={removingVideo || busy || busyTranscribing}><Trash2 size={13}/> {removingVideo ? "Removing from index…" : "Remove from library"}</button>}<div className="panel-footer"><span>{library.videos.length} {library.videos.length === 1 ? 'LECTURE' : 'LECTURES'} INDEXED</span><span>LOCAL LIBRARY</span></div></section>
        <section className="panel saved-panel" id="saved"><div className="panel-head"><div><div className="eyebrow">PICK UP LATER / 05</div><h2>Saved moments</h2></div><span className="count-pill">{String(bookmarks.length).padStart(2, '0')}</span></div><p className="panel-desc">The ideas worth coming back to.</p>{bookmarks.length ? <div className="saved-list">{bookmarks.map(item => <article className="saved-item" key={item.id}><div className="saved-meta"><button onClick={() => seek(item)}><Play size={11} fill="currentColor"/> {formatTime(item.start_time)}</button><span>{item.reviewed ? 'REVIEWED' : 'TO REVIEW'}</span></div><strong>{library.videos.find(v => v.id === item.video_id)?.title || 'Unavailable video'}</strong><textarea maxLength={4000} aria-label="Review note" placeholder="Add a study note…" value={noteDrafts[item.id] ?? item.note ?? ''} onChange={e => setNoteDrafts(previous => ({ ...previous, [item.id]: e.target.value }))}/><div className="saved-actions"><button onClick={() => updateBookmark(item, 'note')} disabled={noteDrafts[item.id] === undefined || savingNotes[item.id]}>Save note</button><button disabled={savingNotes[item.id]} onClick={() => updateBookmark(item, 'review')}>{item.reviewed ? 'Mark pending' : 'Mark reviewed'}</button><button disabled={savingNotes[item.id]} onClick={() => updateBookmark(item, 'delete')} aria-label="Remove saved moment"><Trash2 size={14}/></button></div></article>)}</div> : <div className="saved-empty"><Bookmark size={21}/><strong>Save a moment</strong><span>Bookmarks from your search will appear here.</span></div>}<div className="panel-footer"><span>YOUR PERSONAL REVIEW LIST</span><span><LockKeyhole size={11}/> LOCAL</span></div></section>
        <FeedbackHistory revision={feedbackRevision} notify={notify}/><div className="insight-card"><div className="insight-mark"><Sparkles size={17}/></div><div className="eyebrow">THE REWIND PRINCIPLE</div><p>Good answers start with <em>good evidence.</em></p><span>Every result links back to its source.</span><ArrowUpRight className="insight-arrow" size={16}/></div>
      </aside></div><footer className="footer"><span>REWIND<span className="orange">AI</span> <i>—</i> A MORE THOUGHTFUL WAY TO LEARN</span><span>LOCAL WORKSPACE <i>·</i> BUILT FOR FOCUS</span></footer></main></div>
    {uploadOpen && <UploadDialog onClose={() => setUploadOpen(false)} onImported={async () => { await load(); }} notify={notify}/>}{subtitleTarget && <SubtitleDialog video={subtitleTarget} onClose={() => setSubtitleTarget(null)} onUpdated={async () => { setResult(null); await load(); }} notify={notify}/>}{feedbackDraft && <FeedbackDialog feedback={feedbackDraft} onClose={() => setFeedbackDraft(null)} onSaved={() => setFeedbackRevision(value => value + 1)} notify={notify}/>}{toast && <div className="toast" role="status">{toast}</div>}
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
