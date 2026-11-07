// Exercise the real page script with delayed server responses and synthetic data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../src/video_rag/api/templates/answer_review.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];

async function harness() {
  const elements = {};
  const getElementById = id => elements[id] ??= {
    value: '', checked: false, disabled: false, textContent: '',
    replaceChildren() {}, append() {},
  };
  let pending, beforeUnload;
  const context = {
    document: {getElementById},
    window: {addEventListener: (name, fn) => { if (name === 'beforeunload') beforeUnload = fn; }},
    confirm: () => true,
    fetch: (url, options) => {
      if (url === '/api/review') return Promise.resolve({ok: true, json: async () => ({rows: [{question: 'Synthetic', answer: 'Synthetic', expected_answerable: true, evidence: []}]})});
      return new Promise(resolve => { pending = {url, options, resolve}; });
    },
  };
  vm.runInNewContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  const change = (id, value) => { elements[id].value = value; elements[id].onchange(); };
  const reply = body => pending.resolve({ok: true, json: async () => body});
  const isDirty = () => { let prevented = false; beforeUnload({preventDefault() {prevented = true;}}); return prevented; };
  return {elements, change, reply, isDirty, pending: () => pending};
}

(async () => {
  {
    const h = await harness();
    h.change('verdict', 'correct'); h.change('support', 'supported');
    const save = h.elements.saveProgress.onclick();
    assert.equal(JSON.parse(h.pending().options.body).judgments[0].answer, 'correct');
    h.change('verdict', 'incorrect');
    h.reply({filename: 'synthetic.json'}); await save;
    assert.match(h.elements.result.textContent, /New choices remain unsaved/);
    assert.equal(h.elements.verdict.value, 'incorrect');
    assert.equal(h.isDirty(), true);
  }
  {
    const h = await harness();
    const restore = h.elements.restoreProgress.onclick();
    h.change('verdict', 'incorrect');
    h.reply({judgments: [{answer: 'correct', evidence: 'supported'}]}); await restore;
    assert.match(h.elements.result.textContent, /nothing was replaced/);
    assert.equal(h.elements.verdict.value, 'incorrect');
    assert.equal(h.isDirty(), true);
  }
  {
    const h = await harness();
    h.change('verdict', 'correct'); h.change('support', 'supported');
    h.elements.confirmed.checked = true;
    const save = h.elements.save.onclick();
    h.change('verdict', 'incorrect');
    h.reply({filename: 'synthetic.json'}); await save;
    assert.match(h.elements.result.textContent, /scores do not describe them/);
    assert.equal(h.elements.confirmed.checked, false);
    assert.equal(h.isDirty(), true);
  }
  {
    const h = await harness();
    h.change('verdict', 'correct');
    const save = h.elements.saveProgress.onclick();
    h.reply({filename: 'synthetic.json'}); await save;
    assert.match(h.elements.result.textContent, /Progress saved locally/);
    assert.equal(h.isDirty(), false);
  }
  console.log('Answer-review race checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
