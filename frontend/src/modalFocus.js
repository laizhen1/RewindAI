// Keep focus inside a modal without refocusing it on unrelated application renders.
export function attachModalFocusTrap(document, dialog, getState, initialSelector) {
  const previous = document.activeElement;
  dialog?.querySelector(initialSelector)?.focus();
  const onKeyDown = event => {
    if (event.key === 'Escape') {
      const state = getState();
      if (!state.busy) { event.preventDefault(); state.onClose(); }
      return;
    }
    if (event.key !== 'Tab' || !dialog) return;
    const controls = [...dialog.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled)')];
    if (!controls.length) { event.preventDefault(); dialog.focus(); return; }
    const first = controls[0], last = controls[controls.length - 1];
    if (!controls.includes(document.activeElement)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    } else if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  };
  document.addEventListener('keydown', onKeyDown);
  return () => { document.removeEventListener('keydown', onKeyDown); previous?.focus?.(); };
}
