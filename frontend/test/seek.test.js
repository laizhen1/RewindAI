import assert from 'node:assert/strict';
import test from 'node:test';
import { makePendingSeek, resolvePendingSeek } from '../src/seek.js';

test('a deferred timestamp can only be applied to its intended lecture', () => {
  const pending = makePendingSeek({ start_time: 75 }, { id: 'lecture-b', available: true });
  assert.deepEqual(resolvePendingSeek(pending, 'lecture-a', 200), { status: 'unmatched' });
  assert.deepEqual(resolvePendingSeek(pending, 'lecture-b', NaN), { status: 'waiting' });
  assert.deepEqual(resolvePendingSeek(pending, 'lecture-b', 200), { status: 'ready', time: 75 });
  assert.deepEqual(resolvePendingSeek(pending, 'lecture-b', 60), { status: 'out_of_range' });
  assert.deepEqual(resolvePendingSeek(null, 'lecture-b', 200), { status: 'unmatched' });
});

test('unavailable videos and invalid timestamps do not create a pending seek', () => {
  const video = { id: 'lecture-a', available: true };
  for (const start_time of [null, undefined, '', ' ', -1, Infinity, 'not a timestamp']) {
    assert.equal(makePendingSeek({ start_time }, video), null);
  }
  assert.equal(makePendingSeek({ start_time: 3 }, { ...video, available: false }), null);
  assert.equal(makePendingSeek({ start_time: 3, available: false }, video), null);
  assert.deepEqual(makePendingSeek({ start_time: 0 }, video), { videoId: 'lecture-a', time: 0 });
});
