import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequestGuard } from '../src/requestGuard.js';

test('an in-flight result is discarded when lecture selection changes', () => {
  const guard = createRequestGuard();
  const oldSearch = guard.start();
  assert.equal(guard.current(oldSearch), true);
  guard.invalidate();
  assert.equal(guard.current(oldSearch), false);
  const nextSearch = guard.start();
  assert.equal(guard.current(oldSearch), false);
  assert.equal(guard.current(nextSearch), true);
});
