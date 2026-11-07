import assert from 'node:assert/strict';
import test from 'node:test';
import { uploadLecture } from '../src/upload.js';

function fakeRequest() {
  const handlers = {}, uploadHandlers = {};
  return {
    handlers, uploadHandlers,
    upload: { addEventListener: (type, callback) => { uploadHandlers[type] = callback; } },
    addEventListener: (type, callback) => { handlers[type] = callback; },
    open(method, url) { this.method = method; this.url = url; },
    send(body) { this.body = body; },
  };
}

test('reports copy progress and returns the local import result', async () => {
  const request = fakeRequest(), progress = [], form = {};
  const result = uploadLecture(form, value => progress.push(value), () => request);
  assert.equal(request.method, 'POST');
  assert.equal(request.url, '/api/library/upload');
  assert.equal(request.body, form);
  request.uploadHandlers.progress({ lengthComputable: true, loaded: 5, total: 10 });
  request.uploadHandlers.progress({ lengthComputable: false, loaded: 0, total: 0 });
  request.uploadHandlers.progress({ lengthComputable: true, loaded: 10, total: 10 });
  request.status = 201; request.responseText = '{"state":"ready"}'; request.handlers.load();
  assert.deepEqual(progress, [50, 100]);
  assert.equal((await result).state, 'ready');
});

test('reports a server failure and warns when the outcome is unknown', async () => {
  const failed = fakeRequest();
  const first = uploadLecture({}, () => {}, () => failed);
  failed.status = 400; failed.responseText = '{"error":"Invalid subtitles"}'; failed.handlers.load();
  await assert.rejects(first, /Invalid subtitles/);
  const interrupted = fakeRequest();
  const second = uploadLecture({}, () => {}, () => interrupted);
  interrupted.handlers.error();
  await assert.rejects(second, /may have completed; reload and check the library/);
  const invalid = fakeRequest();
  const third = uploadLecture({}, () => {}, () => invalid);
  invalid.status = 201; invalid.responseText = 'not json'; invalid.handlers.load();
  await assert.rejects(third, /unexpected response/);
});
