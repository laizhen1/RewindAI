import assert from 'node:assert/strict';
import test from 'node:test';
import { attachModalFocusTrap } from '../src/modalFocus.js';

test('modal retains focus across state changes and restores it on close', () => {
  let listener, controls = [], closes = 0, latest = { busy: false, onClose: () => { closes += 1; } };
  const document = {
    activeElement: null,
    addEventListener(type, callback) { assert.equal(type, 'keydown'); listener = callback; },
    removeEventListener(type, callback) { assert.equal(type, 'keydown'); assert.equal(callback, listener); listener = null; },
  };
  const element = () => ({ focuses: 0, focus() { this.focuses += 1; document.activeElement = this; } });
  const previous = element(), first = element(), last = element(), outside = element();
  document.activeElement = previous;
  controls = [first, last];
  const dialog = { ...element(), querySelector: () => first, querySelectorAll: () => controls };
  const detach = attachModalFocusTrap(document, dialog, () => latest, 'button');
  assert.equal(document.activeElement, first);
  const event = (key, shiftKey = false) => ({key, shiftKey, prevented: false, preventDefault() { this.prevented = true; }});
  const escapeWhileBusy = event('Escape'); latest = { ...latest, busy: true }; listener(escapeWhileBusy);
  assert.equal(closes, 0); assert.equal(first.focuses, 1);
  const escapeAfterUpdate = event('Escape'); latest = { busy: false, onClose: () => { closes += 2; } }; listener(escapeAfterUpdate);
  assert.equal(closes, 2); assert.equal(escapeAfterUpdate.prevented, true);
  last.focus(); const tab = event('Tab'); listener(tab); assert.equal(document.activeElement, first);
  first.focus(); const reverse = event('Tab', true); listener(reverse); assert.equal(document.activeElement, last);
  outside.focus(); const enterModal = event('Tab'); listener(enterModal); assert.equal(document.activeElement, first);
  controls = []; const noControls = event('Tab'); listener(noControls);
  assert.equal(noControls.prevented, true); assert.equal(document.activeElement, dialog);
  detach(); assert.equal(listener, null); assert.equal(document.activeElement, previous);
});
