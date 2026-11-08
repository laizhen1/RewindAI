// Exercise local review choices while synthetic save and restore requests are pending.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../src/video_rag/api/templates/relevance_review.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];

async function harness() {
  const elements = {};
  const element = () => ({children: [], textContent: '', checked: false, disabled: false, scrollTop: 0,
    append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; },
    play() { return Promise.resolve(); }});
  const getElementById = id => elements[id] ??= element();
  let pending, beforeUnload;
  const context = {
    document: {getElementById, createElement: element, createTextNode: value => ({textContent: value})},
    window: {addEventListener: (name, fn) => { if (name === 'beforeunload') beforeUnload = fn; }},
    confirm: () => true,
    fetch: (url, options) => {
      if (url === '/api/data') return Promise.resolve({ok: true, json: async () => ({
        questions: [{question_id:'q1',question:'Synthetic?',expected_answer:'A',question_type:'audio',
                     evidence_start_seconds:1,evidence_end_seconds:3}],
        segments: [{id:'s1',start:1,end:3,text:'Synthetic evidence'}],unanswerable_excluded:0})});
      return new Promise(resolve => {pending = {url, options, resolve};});
    },
  };
  vm.runInNewContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  const select = checked => { const input=elements.segments.children[0].children[2].children[0]; input.checked=checked; input.onchange(); };
  const reply = body => pending.resolve({ok:true,json:async()=>body});
  const dirty = () => {let prevented=false;beforeUnload({preventDefault(){prevented=true;}});return prevented;};
  return {elements, select, reply, dirty, pending:()=>pending};
}

(async () => {
  {
    const h=await harness();h.elements.segments.scrollTop=42;
    assert.match(h.elements.excluded.textContent,/0 unanswerable/);
    const input=h.elements.segments.children[0].children[2].children[0];h.select(true);
    assert.equal(h.elements.segments.scrollTop,42);
    assert.equal(h.elements.segments.children[0].children[2].children[0],input);
    h.elements.verified.checked=true;h.elements.verified.onchange();
    h.elements.exhaustive.checked=true;h.elements.exhaustive.onchange();
    const save=h.elements.save.onclick();
    assert.equal(JSON.parse(h.pending().options.body).judgments[0].exhaustive,true);
    h.select(false);h.reply({filename:'synthetic.json'});await save;
    assert.match(h.elements.status.textContent,/newer choices are unsaved/);
    assert.equal(h.elements.verified.checked,false);
    assert.equal(h.elements.exhaustive.checked,false);
    assert.equal(h.dirty(),true);
  }
  {
    const h=await harness();const restore=h.elements.restore.onclick();h.select(true);
    h.reply({judgments:[{question_id:'q1',relevant_segment_ids:[],verified:false,exhaustive:false}]});await restore;
    assert.match(h.elements.status.textContent,/nothing was replaced/);
    assert.equal(h.elements.segments.children[0].children[2].children[0].checked,true);
    assert.equal(h.dirty(),true);
  }
  console.log('Relevance-review race checks passed');
})().catch(error=>{console.error(error);process.exitCode=1});
