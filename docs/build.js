#!/usr/bin/env node
// Builds docs/index.html from docs/source.html. Figures, tables and results come from the run files and docs/tanks.json. Some prose numbers are typed.
// Usage: node docs/build.js
const fs = require('fs');
const path = require('path');
const Sim = require('../lander/sim.js');
const Policy = require('../lander/policy.js');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const json = (p) => JSON.parse(read(p));
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const pct = (v) => (v * 100).toFixed(1) + '%';
const jpeg = (name) => `data:image/jpeg;base64,${fs.readFileSync(path.join(ROOT, 'docs/img', name)).toString('base64')}`;

const runDirs = fs.readdirSync(path.join(ROOT, 'results')).filter((d) => fs.statSync(path.join(ROOT, 'results', d)).isDirectory());
const findRun = (prefix) => {
  const hit = runDirs.filter((d) => d.startsWith(prefix) && fs.existsSync(path.join(ROOT, 'results', d, 'score.json'))).sort();
  if (!hit.length) throw new Error(`no run ${prefix}`);
  return hit;
};
const run = (dir) => ({ dir, score: json(`results/${dir}/score.json`), record: json(`results/${dir}/run.json`), answer: read(`results/${dir}/answer.md`) });

const landers = findRun('lander-pads-').map(run);
const LETTERS = 'ABCDEFGH';
const nets = landers.map((r) => json(`results/${r.dir}/network.json`));
const short = landers.find((r) => r.score.eval.fixed_target.episodes !== 3000);
if (short) throw new Error(`${short.dir} was scored on ${short.score.eval.fixed_target.episodes} flights; the manual quotes 3,000. Rescore it.`);
const exp1 = run(findRun('exp1-breakout-claude')[0]);
const control = run(findRun('exp1-breakout-shuffle')[0]);
const exp5 = run(findRun('exp5-breakout-')[0]);
const baselines = Object.fromEntries(Object.entries(Policy.PILOTS).map(([name, pilot]) => (
  [name, { fixed: Policy.evaluate(pilot), moved: Policy.evaluate(pilot, { midFlightSwitch: true }) }]
)));
const OUTCOMES = Object.keys(baselines.controller.fixed.outcomes);
const stress = json('results/lander-stress.json');
const published = json('data/exp5_write_from_scratch/eval/minatar_eval.json');
const verified = json('results/verify.json').games;

// --- Tables ---------------------------------------------------------------------------------------------------------

function table(head, rows, numericFrom = 1) {
  const cell = (tag, v, i) => `<${tag}${i >= numericFrom ? ' class="n"' : ''}>${v}</${tag}>`;
  return `<div class="scroll"><table><thead><tr>${head.map((h, i) => cell('th', h, i)).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((v, i) => cell('td', v, i)).join('')}</tr>`).join('')}</tbody></table></div>`;
}

const fixedOf = (r) => r.score.eval.fixed_target;
const movedOf = (r) => r.score.eval.mid_flight_switch;

const resultsTable = table(['Pilot', 'Fixed pad: landed', 'Pad changes: landed', 'Fuel left (fixed pad)', 'Ticks (fixed pad)'], [
  ...landers.map((r, i) => [`Claude reply ${LETTERS[i]}`, pct(fixedOf(r).landed), pct(movedOf(r).landed), Math.round(fixedOf(r).meanFuelLeft), Math.round(fixedOf(r).meanTicks)]),
  ['Plain program', pct(baselines.controller.fixed.landed), pct(baselines.controller.moved.landed), Math.round(baselines.controller.fixed.meanFuelLeft), Math.round(baselines.controller.fixed.meanTicks)],
  ['Random', pct(baselines.random.fixed.landed), pct(baselines.random.moved.landed), '–', '–'],
]);

const outcomeTable = table(['Pilot and test', ...OUTCOMES], [
  ...landers.flatMap((r, i) => [[`Reply ${LETTERS[i]}, fixed pad`, fixedOf(r)], [`Reply ${LETTERS[i]}, pad changes`, movedOf(r)]]),
  ['Plain program, fixed pad', baselines.controller.fixed], ['Plain program, pad changes', baselines.controller.moved],
  ['Random, fixed pad', baselines.random.fixed],
].map(([name, e]) => [name, ...OUTCOMES.map((k) => e.outcomes[k])]));

const replyTable = table(['Reply', 'Statements', 'Nonzero weights of 484', 'Output tokens', 'Time (s)', 'Tool-free'], landers.map((r, i) => [
  LETTERS[i], `${r.score.applied} of ${r.score.statements}`, r.score.nonzero, r.record.audit.usage.output_tokens.toLocaleString('en-US'), Math.round(r.record.wall_s), r.record.audit.tool_free ? 'Yes' : 'No',
]));

const stressPilots = Object.keys(Object.values(stress.table)[0]);
const stressTable = table(['Start state', ...stressPilots], Object.entries(stress.table).map(([k, v]) => [k, ...stressPilots.map((p) => pct(v[p].landed))]));

const costRows = [['Exp 1 Breakout', exp1], ['Exp 1 Breakout, shuffled', control], ['Exp 5 Breakout', exp5], ...landers.map((r, i) => [`Three Pads reply ${LETTERS[i]}`, r])];
const costTable = table(['Call', 'Effort', 'Output tokens', 'Time (s)', 'Cost figure from the CLI (USD)'], costRows.map(([n, r]) => [n, r.record.effort, r.record.audit.usage.output_tokens.toLocaleString('en-US'), Math.round(r.record.wall_s), r.record.audit.cost_usd.toFixed(2)]));
const totalCost = costRows.reduce((a, [, r]) => a + r.record.audit.cost_usd, 0);

const verifyTable = table(['Game', 'Published mean ± SE', 'This project mean ± SE', 'Difference (SE)'], ['breakout', 'seaquest', 'asterix'].map((g) => {
  const v = verified[g];
  return [g, `${v.published_mean.toFixed(1)} ± ${v.published_se.toFixed(1)}`, `${v.mean.toFixed(1)} ± ${v.se.toFixed(1)}`, v.gap_se.toFixed(2)];
}));

// --- Exp 1: row counts, from the prompt text Astra and Claude both received --------------------------------------------

function exp1Counts() {
  const lines = read('data/exp1_which_game/runs/breakout/prompt.txt').split('\n');
  const start = lines.findIndex((l) => l.startsWith('W1 (')) + 1;
  const W1 = lines.slice(start, start + 500).map((l) => l.split(/\s+/).map(Number));
  const b1 = lines[lines.findIndex((l) => l.startsWith('b1 (')) + 1].split(/\s+/).map(Number);
  const corr = (a, b) => {
    const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
    const [ma, mb] = [mean(a), mean(b)];
    let ab = 0, aa = 0, bb = 0;
    a.forEach((x, i) => { ab += (x - ma) * (b[i] - mb); aa += (x - ma) ** 2; bb += (b[i] - mb) ** 2; });
    return ab / Math.sqrt(aa * bb);
  };
  const big = W1.map((r) => Math.max(...r.map(Math.abs)) > 0.4);
  const biasLike = W1.map((r) => corr(r, b1) > 0.8);
  const inChannels = (cs) => big.filter((v, i) => v && cs.includes(i % 5)).length;
  return { active: big.filter(Boolean).length, biasLike: biasLike.filter(Boolean).length,
    noise: big.filter((v, i) => !v && !biasLike[i]).length, moving: inChannels([1, 4]), persistent: inChannels([0, 2]) };
}
const c = exp1Counts();
const countsTable = table(['Row type', 'Claude estimate, shuffled weights, no tools', 'Measured'], [
  ['Active rows (largest weight above 0.4)', 'About 148', c.active],
  ['Rows that follow b1 (correlation above 0.8)', 'About 36 to 37', c.biasLike],
  ['Noise rows', 'About 315', c.noise],
  ['Large group (true channels: moving objects, previous frame)', 'About 115 to 120', c.moving],
  ['Smaller group (true channels: agent, reward targets)', 'About 25 to 30', c.persistent],
]);

// --- Worked example: one forward pass of reply B -------------------------------------------------------------------------

function workedExample() {
  const state = { x: 30, y: 40, vx: 0.1, vy: -0.2, fuel: 200, target: 2 };
  const x = Sim.observe(state);
  const out = Policy.forward(nets[1], x);
  const fmt = (v) => v.toFixed(3);
  const used = (arr) => arr.map((v, i) => [i, v]).filter(([, v]) => v !== 0);
  return {
    inputs: table(['Input', 'Name', 'Value'], x.map((v, i) => [i, Policy.INPUT_NAMES[i], fmt(v)]), 2),
    h1: used(out.h1).map(([i, v]) => `h1[${i}] = ${fmt(v)}`).join(', '),
    h2: used(out.h2).map(([i, v]) => `h2[${i}] = ${fmt(v)}`).join(', '),
    outputs: table(['Action', 'Logit', 'Chance'], Sim.ACTIONS.map((a, i) => [a, fmt(out.logits[i]), pct(out.probs[i])])),
  };
}
const worked = workedExample();

// --- Figures ------------------------------------------------------------------------------------------------------------

let allLanded = true; // a weak reply must stay in the manual, so this changes the caption and does not stop the build

function flightPanel(title, net, switched) {
  const W = 340, H = 220, top = 26, bottom = 26, side = 10;
  const sx = (x) => side + (x / Sim.WIDTH) * (W - 2 * side);
  const sy = (y) => H - bottom - (y / 65) * (H - top - bottom);
  const pilot = Policy.netPilot(net);
  const parts = [`<line class="axis" x1="${side}" x2="${W - side}" y1="${H - bottom}" y2="${H - bottom}"></line>`];
  Sim.PADS.forEach((p, i) => {
    parts.push(`<rect class="s${i + 1}f" x="${sx(p - Sim.PAD_HALF_WIDTH).toFixed(1)}" y="${H - bottom}" width="${(sx(Sim.PAD_HALF_WIDTH * 2) - sx(0)).toFixed(1)}" height="5"></rect>`);
    parts.push(`<text class="lab" x="${sx(p).toFixed(1)}" y="${H - 6}" text-anchor="middle">Pad ${i + 1}</text>`);
  });
  for (let target = 0; target < 3; target++) {
    for (let k = 0; k < 3; k++) {
      const seed = 8000000 + target * 10 + k;
      const switchAt = switched ? 60 : 0;
      const points = [];
      let mark = null;
      const s = Policy.rollout(pilot, seed, target, { switchAt, onStep: (now) => {
        if (now.tick % 3 === 0 || now.outcome) points.push([now.x, now.y]);
        if (switchAt && now.tick === switchAt) mark = [now.x, now.y];
      } });
      if (s.outcome !== 'landed') allLanded = false;
      const finalTarget = s.target;
      const d = points.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
      parts.push(`<polyline class="s${finalTarget + 1}s" points="${d}"><title>Target pad ${finalTarget + 1}, landed at tick ${s.tick}, fuel left ${Math.round(s.fuel)}</title></polyline>`);
      if (mark) parts.push(`<circle class="s${finalTarget + 1}f mark" cx="${sx(mark[0]).toFixed(1)}" cy="${sy(mark[1]).toFixed(1)}" r="4"><title>Target changes to pad ${finalTarget + 1} at tick ${switchAt}</title></circle>`);
    }
  }
  return `<figure class="panel"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><text class="ttl" x="${side}" y="16">${esc(title)}</text>${parts.join('')}</svg></figure>`;
}

const flightFigure = `<div class="panels">${[
  ...nets.map((n, i) => flightPanel(`Reply ${LETTERS[i]}, fixed pad`, n, false)),
  flightPanel('Reply A, pad changes at tick 60 (dot)', nets[0], true),
].join('')}</div>`;

function breakoutBars() {
  const hand = published.breakout;
  const best = (prefix) => Object.entries(hand).filter(([k]) => k.startsWith(prefix)).sort((a, b) => b[1].mean - a[1].mean)[0][1].mean;
  const rows = [
    ['PPO, 100M steps (published, best seed)', best('ppo100'), null],
    ['PPO, 500M steps (published, best seed)', best('ppo500'), null],
    ['Claude Opus 5.5, one reply (this project)', exp5.score.eval.mean, exp5.score.eval.se],
    ['GPT-6 Astra, rescored by this project', verified.breakout.mean, verified.breakout.se],
    ['GPT-6 Astra (published)', hand.hand.mean, hand.hand.se],
  ];
  const W = 680, rowH = 46, left = 10, right = 80, top = 8, max = 600;
  const sx = (v) => left + (v / max) * (W - left - right);
  const H = top + rows.length * rowH + 26;
  const ticks = [0, 200, 400, 600].map((t) => `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${top}" y2="${H - 22}"></line><text class="lab" x="${sx(t)}" y="${H - 6}" text-anchor="middle">${t}</text>`).join('');
  const bars = rows.map(([label, v, se], i) => {
    const y = top + i * rowH;
    const text = se ? `${v.toFixed(1)} ± ${se.toFixed(1)}` : v.toFixed(1);
    return `<g><title>${esc(label)}: ${text}</title><text class="lab" x="${left}" y="${y + 13}">${esc(label)}</text><rect class="s1f" x="${left}" y="${y + 19}" width="${Math.max(2, sx(v) - left).toFixed(1)}" height="14" rx="3"></rect><text class="val" x="${(sx(v) + 6).toFixed(1)}" y="${y + 31}">${text}</text></g>`;
  }).join('');
  return `<div class="scroll"><svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean Breakout score by source of the network">${ticks}${bars}</svg></div>`;
}

function pipelineDiagram() {
  const box = (x, y, w, title, sub, cls = '') => `<g><rect class="box ${cls}" x="${x}" y="${y}" width="${w}" height="62" rx="4"></rect><text class="ttl" x="${x + w / 2}" y="${y + 26}" text-anchor="middle">${esc(title)}</text><text class="lab" x="${x + w / 2}" y="${y + 46}" text-anchor="middle">${esc(sub)}</text></g>`;
  const arrow = (x1, y1, x2, y2) => `<line class="arrow" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#head)"></line>`;
  return `<div class="scroll"><svg class="wide" viewBox="0 0 700 300" role="img" aria-label="Data flow from game rules to the prompt, to Claude, to the network, to the scorer and the page">
    <defs><marker id="head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="head" d="M0 0 L8 4 L0 8 Z"></path></marker></defs>
    ${box(10, 20, 150, 'lander/sim.js', 'game rules')}
    ${box(275, 20, 150, 'lander/prompt.md', 'task and network')}
    ${box(540, 20, 150, 'Claude', 'no tools, one reply', 'accent')}
    ${box(540, 120, 150, 'answer.md', 'edits block')}
    ${box(275, 120, 150, 'wsp.py score', 'parser')}
    ${box(10, 120, 150, 'network.json', '484 weights')}
    ${box(10, 220, 150, 'lander/eval.js', 'scorer')}
    ${box(275, 220, 150, 'networks.js', 'wsp.py lander-export')}
    ${box(540, 220, 150, 'lander/index.html', 'the page')}
    ${arrow(160, 51, 273, 51)}${arrow(425, 51, 538, 51)}${arrow(615, 82, 615, 118)}${arrow(540, 151, 427, 151)}${arrow(275, 151, 162, 151)}${arrow(85, 182, 85, 218)}${arrow(160, 251, 273, 251)}${arrow(425, 251, 538, 251)}
  </svg></div>`;
}

// --- Tank experiment. docs/tanks.json holds scores only, so the public build needs no private results. ------------------

const tanks = json('docs/tanks.json');
const GROUPS = { base: ['s1', 'Plain fix or baseline'], llm: ['s2', 'Written or edited by an LLM'], best: ['s3', 'Trained, or a detector'] };
const f3 = (v) => v.toFixed(3);
const legend = (x, y, items) => items.map(([cls, label], i) => `<circle class="${cls}f" cx="${x + i * 200 + 6}" cy="${y}" r="5"></circle><text class="lab" x="${x + i * 200 + 16}" y="${y + 4}">${esc(label)}</text>`).join('');

function intervalChart(value, ci, name) {
  const rows = tanks.rows;
  const W = 680, left = 245, right = 16, top = 30, rowH = 22;
  const sx = (v) => left + v * (W - left - right);
  const H = top + rows.length * rowH + 26;
  const grid = [0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${top - 4}" y2="${H - 22}"></line><text class="lab" x="${sx(t)}" y="${H - 6}" text-anchor="middle">${t}</text>`).join('');
  const marks = rows.map((r, i) => {
    const y = top + i * rowH + 9;
    const cls = GROUPS[r.group][0];
    return `<g><title>${esc(r.label)}: ${name} ${f3(r[value])}, 95% interval ${r[ci][0]} to ${r[ci][1]}</title><rect x="0" y="${y - 10}" width="${W}" height="${rowH}" fill="transparent"></rect><text class="lab" x="${left - 10}" y="${y + 4}" text-anchor="end">${esc(r.label)}</text><line class="${cls}s" stroke-width="2" x1="${sx(r[ci][0])}" x2="${sx(r[ci][1])}" y1="${y}" y2="${y}"></line><circle class="${cls}f mark" cx="${sx(r[value])}" cy="${y}" r="5"></circle></g>`;
  }).join('');
  return `<div class="scroll"><svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="${name} on the test cells for each comparator, with 95 percent intervals. The table above gives the same values.">${legend(14, 13, Object.values(GROUPS))}${grid}${marks}</svg></div>`;
}

function sizeChart() {
  const series = [['s3', 'Found at score >= 0.70'], ['s1', 'Found only below 0.70'], ['s2', 'Not found']];
  const W = 680, left = 110, right = 40, top = 30, rowH = 34;
  const max = Math.max(...tanks.sizes.map((r) => r.tanks));
  const sx = (v) => (v / max) * (W - left - right);
  const H = top + tanks.sizes.length * rowH + 8;
  const rows = tanks.sizes.map((r, i) => {
    const y = top + i * rowH;
    const parts = [r.at_70, r.any - r.at_70, r.tanks - r.any];
    let x = left;
    const bars = parts.map((n, j) => {
      const w = sx(n);
      const rect = n ? `<rect class="${series[j][0]}f" x="${x.toFixed(1)}" y="${y}" width="${Math.max(1, w - 2).toFixed(1)}" height="22" rx="2"><title>${esc(r.label)}: ${esc(series[j][1])}: ${n} of ${r.tanks}</title></rect>` : '';
      x += w;
      return rect;
    }).join('');
    return `<text class="lab" x="${left - 10}" y="${y + 15}" text-anchor="end">${esc(r.label)}</text>${bars}<text class="val" x="${(x + 6).toFixed(1)}" y="${y + 15}">${r.tanks}</text>`;
  }).join('');
  return `<div class="scroll"><svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="Tanks found by RT-DETRv2 by tank diameter. The table above gives the same values.">${legend(14, 13, series)}${rows}</svg></div>`;
}

function shuffleChart() {
  const b = tanks.bn1;
  const rows = [['True statistics, all channels', [b.true[0]]], ['True variance only', [b.variance[0]]], ['One scale on live variances', [b.scale[0]]],
    ['True mean only', [b.mean[0]]], ['5 shuffles, near-zero rows skipped', b.shuffle_skip.map((v) => v[0])], ['5 shuffles, all rows written', b.shuffle_all.map((v) => v[0])]];
  const W = 680, left = 245, right = 16, top = 10, rowH = 30;
  const sx = (v) => left + v * (W - left - right);
  const H = top + rows.length * rowH + 26;
  const grid = [0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => `<line class="grid" x1="${sx(t)}" x2="${sx(t)}" y1="${top}" y2="${H - 22}"></line><text class="lab" x="${sx(t)}" y="${H - 6}" text-anchor="middle">${t}</text>`).join('');
  const marks = rows.map(([label, values], i) => {
    const y = top + i * rowH + 14;
    return `<text class="lab" x="${left - 10}" y="${y + 4}" text-anchor="end">${esc(label)}</text>${values.map((v) => `<circle class="s1f mark" cx="${sx(v).toFixed(1)}" cy="${y}" r="5"><title>${esc(label)}: val IoU ${f3(v)}</title></circle>`).join('')}`;
  }).join('');
  return `<div class="scroll"><svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="Val IoU for plain edits of encoder.bn1. The table above gives the same values.">${grid}${marks}</svg></div>`;
}

function scatterChart() {
  const rows = tanks.rows.filter((r) => r.fp > 0);
  const W = 680, H = 330, left = 52, right = 20, top = 30, bottom = 44;
  const sx = (v) => left + (Math.log10(v) / 3) * (W - left - right);
  const sy = (v) => H - bottom - (v / 28) * (H - top - bottom);
  const gridX = [1, 3, 10, 30, 100, 300, 1000].map((t) => `<line class="grid" x1="${sx(t).toFixed(1)}" x2="${sx(t).toFixed(1)}" y1="${top}" y2="${H - bottom}"></line><text class="lab" x="${sx(t).toFixed(1)}" y="${H - bottom + 16}" text-anchor="middle">${t}</text>`).join('');
  const gridY = [0, 7, 14, 21, 28].map((t) => `<line class="grid" x1="${left}" x2="${W - right}" y1="${sy(t).toFixed(1)}" y2="${sy(t).toFixed(1)}"></line><text class="lab" x="${left - 8}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`).join('');
  const named = { 'RT-DETRv2-S, fixed settings': 'RT-DETRv2-S', 'ResNet18 U-Net, trained': 'Trained U-Net', 'Astra write 60743': 'Astra 60743', 'Claude write 54466': 'Claude 54466', 'No edit, seed 0': 'No edit' };
  const marks = rows.map((r) => {
    const cls = GROUPS[r.group][0];
    const x = sx(r.fp), y = sy(r.tp);
    const label = named[r.label] ? `<text class="lab" x="${(x + 9).toFixed(1)}" y="${(y - 7).toFixed(1)}">${named[r.label]}</text>` : '';
    return `<g><title>${esc(r.label)}: ${r.tp} of 28 tanks found, ${r.fp} false objects</title><circle class="${cls}f mark" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"></circle>${label}</g>`;
  }).join('');
  return `<div class="scroll"><svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="Test tanks found against false objects, for each comparator with one or more false objects. The table above gives the same values.">${legend(14, 13, Object.values(GROUPS))}${gridX}${gridY}${marks}<text class="lab" x="${(left + W - right) / 2}" y="${H - 8}" text-anchor="middle">False objects on the test cells (log scale)</text><text class="lab" transform="translate(13 ${(top + H - bottom) / 2}) rotate(-90)" text-anchor="middle">Tanks found, of 28</text></svg></div>`;
}

const boxOf = (x, y, w, title, sub, cls = '') => `<g><rect class="box ${cls}" x="${x}" y="${y}" width="${w}" height="56" rx="4"></rect><text class="ttl" x="${x + w / 2}" y="${y + 23}" text-anchor="middle">${esc(title)}</text><text class="lab" x="${x + w / 2}" y="${y + 41}" text-anchor="middle">${esc(sub)}</text></g>`;
const lineOf = (x1, y1, x2, y2) => `<line class="arrow" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#head)"></line>`;

function tankFlow() {
  return `<div class="scroll"><svg class="wide" viewBox="0 0 700 290" role="img" aria-label="Experiment flow. The 2015 image trains the source network. The source network fails on the new image. Three routes follow: repair by an LLM, write from zeros by an LLM, and a detector with no training.">
    ${boxOf(10, 20, 160, '2015 image', 'train cells only')}
    ${boxOf(270, 20, 160, 'Source network', 'ResNet18 U-Net')}
    ${boxOf(530, 20, 160, 'New image', 'the network fails', 'accent')}
    ${boxOf(10, 125, 200, 'Stage B: repair', 'LLM edits the source network')}
    ${boxOf(250, 125, 200, 'Stage C: write', 'LLM sets all weights from 0')}
    ${boxOf(490, 125, 200, 'Detector', 'RT-DETRv2, no training here')}
    ${boxOf(130, 225, 440, 'One scorer', 'same pixel grid, same cells, same metrics')}
    ${lineOf(170, 48, 268, 48)}${lineOf(430, 48, 528, 48)}${lineOf(560, 76, 110, 123)}${lineOf(610, 76, 350, 123)}${lineOf(650, 76, 590, 123)}${lineOf(110, 181, 230, 223)}${lineOf(350, 181, 350, 223)}${lineOf(590, 181, 470, 223)}
  </svg></div>`;
}

function unetDiagram() {
  const enc = [[10, 20, 'encoder.conv1', '7 x 7, stride 2'], [10, 90, 'encoder.bn1', 'BatchNorm, 64 channels'], [10, 160, 'encoder.layer1 to layer4', '11.18M weights in the encoder']];
  return `<div class="scroll"><svg class="wide" viewBox="0 0 700 240" role="img" aria-label="ResNet18 U-Net. The encoder includes encoder.bn1, which all repair replies edited. The decoder leads to segmentation_head.0, which the LDA head edits.">
    ${boxOf(enc[0][0], enc[0][1], 220, enc[0][2], enc[0][3])}${boxOf(enc[1][0], enc[1][1], 220, enc[1][2], enc[1][3], 'accent')}${boxOf(enc[2][0], enc[2][1], 220, enc[2][2], enc[2][3])}
    ${boxOf(290, 160, 180, 'decoder.blocks', '3.15M weights')}${boxOf(290, 20, 180, 'Skip connections', 'encoder to decoder')}
    ${boxOf(520, 160, 170, 'segmentation_head.0', '145 weights', 'accent')}${boxOf(520, 20, 170, 'Logit map', 'tank if logit > 0')}
    ${lineOf(120, 76, 120, 88)}${lineOf(120, 146, 120, 158)}${lineOf(230, 188, 288, 188)}${lineOf(230, 118, 330, 78)}${lineOf(380, 76, 380, 158)}${lineOf(470, 188, 518, 188)}${lineOf(605, 160, 605, 78)}
  </svg></div>`;
}

const ci = (v) => `${v[0].toFixed(3)} to ${v[1].toFixed(3)}`;
const tankTable = table(['Comparator', 'Val IoU', 'Val object F1', 'Test IoU', '95% interval', 'Test object F1', '95% interval', 'Found', 'False', 'Missed'],
  tanks.rows.map((r) => [esc(r.label), f3(r.val_iou), f3(r.val_f1), f3(r.iou), ci(r.iou_ci), f3(r.f1), ci(r.f1_ci), r.tp, r.fp, r.fn]));
const callRows = (name, group, weight) => Object.entries(group).map(([id, c]) => [`${name} ${id}`, `${c.applied} of ${c.statements}`, weight(c).toLocaleString('en-US'), c.tokens.toLocaleString('en-US'), Math.round(c.seconds), c.cost === null || c.cost === undefined ? 'not reported' : c.cost.toFixed(2)]);
const tankCalls = table(['Reply', 'Statements read', 'Weights', 'Output tokens', 'Seconds', 'Cost figure (USD)'], [
  ...callRows('Repair', tanks.calls.repair, (c) => c.changed), ...callRows('Control', tanks.calls.control, (c) => c.changed),
  ...callRows('Claude write', tanks.calls.claude, (c) => c.nonzero), ...callRows('Astra write', tanks.calls.astra, (c) => c.nonzero)]);
const tankSizes = table(['Tank diameter', 'Tanks', 'Found at score >= 0.70', 'Found at any score'], tanks.sizes.map((r) => [r.label, r.tanks, r.at_70, r.any]));
const b1 = tanks.bn1;
const span = (rows, i) => `${f3(Math.min(...rows.map((v) => v[i])))} to ${f3(Math.max(...rows.map((v) => v[i])))}`;
const tankBn1 = table(['Edit of encoder.bn1', 'Val IoU', 'Val object F1'], [['True statistics, all 64 channels (the same network as the repair)', f3(b1.true[0]), f3(b1.true[1])], ['True variance only', f3(b1.variance[0]), f3(b1.variance[1])], ['True mean only', f3(b1.mean[0]), f3(b1.mean[1])], ['One scale on all live variances (x 7.82), with no channel data', f3(b1.scale[0]), f3(b1.scale[1])], ['5 random shuffles, rows with near-zero variance skipped', span(b1.shuffle_skip, 0), span(b1.shuffle_skip, 1)], ['The same 5 shuffles, all rows written', span(b1.shuffle_all, 0), span(b1.shuffle_all, 1)]]);
const tankPlain = table(['Plain rule from the fact sheet', 'Val IoU', 'Val object F1', 'Test IoU', 'Test object F1'], [['Band LDA, midpoint', ...tanks.plain.band.map(f3)], ['Band LDA, midpoint and tank share', ...tanks.plain.band_share.map(f3)], ['Index 1.5B - 0.5R, midpoint', ...tanks.plain.index.map(f3)]]);
const tankCircles = table(['Cells', 'AP50', 'Object F1', 'Found', 'False', 'Missed'], ['train', 'val', 'test'].map((k) => { const c = tanks.circles[k]; return [k[0].toUpperCase() + k.slice(1), f3(c.ap50), f3(c.f1), c.tp, c.fp, c.fn]; }));

// --- Assemble -------------------------------------------------------------------------------------------------------------

const replyBEdits = landers[1].answer.match(/```edits\n([\s\S]*?)```/)[1].trim();
const values = {
  IMG_FLIGHT: jpeg('page-flight.jpg'), IMG_WEIGHTS: jpeg('page-weights.jpg'), IMG_PHONE: jpeg('page-phone.jpg'),
  RESULTS: resultsTable, OUTCOMES: outcomeTable, REPLIES: replyTable, STRESS: stressTable, COSTS: costTable, VERIFY: verifyTable, COUNTS: countsTable,
  TOTAL_COST: totalCost.toFixed(2), FLIGHTS: flightFigure, FLIGHTS_NOTE: allLanded ? 'All these flights landed.' : 'Some of these flights did not land.', BREAKOUT: breakoutBars(), PIPELINE: pipelineDiagram(),
  REPLY_B_EDITS: esc(replyBEdits), WORKED_INPUTS: worked.inputs, WORKED_H1: worked.h1, WORKED_H2: worked.h2, WORKED_OUTPUTS: worked.outputs,
  EXP1_LINE: esc(exp1.score.answer_line), EXP1_CONF: exp1.score.confidence, CONTROL_LINE: esc(control.score.answer_line), CONTROL_CONF: control.score.confidence,
  EXP5_MEAN: exp5.score.eval.mean.toFixed(1), EXP5_SE: exp5.score.eval.se.toFixed(1), EXP5_MAX: exp5.score.eval.max, EXP5_NONZERO: exp5.score.nonzero, EXP5_APPLIED: exp5.score.applied, EXP5_STATEMENTS: exp5.score.statements,
  SOURCE_SIM: esc(read('lander/sim.js').trim()),
  TANK_FLOW: tankFlow(), TANK_UNET: unetDiagram(), TANK_TABLE: tankTable, TANK_CALLS: tankCalls, TANK_SIZES: tankSizes, TANK_BN1: tankBn1, TANK_PLAIN: tankPlain, TANK_CIRCLES: tankCircles,
  TANK_IOU: intervalChart('iou', 'iou_ci', 'Pixel IoU'), TANK_F1: intervalChart('f1', 'f1_ci', 'Object F1'), TANK_SCATTER: scatterChart(), TANK_SIZE_CHART: sizeChart(), TANK_SHUFFLE: shuffleChart(),
  IMG_TANK_SCENE: jpeg('tank-scene.jpg'), IMG_TANK_WEIGHTS: jpeg('tank-weights.jpg'),
};
// --- References: numbered by their order in references.json. The build fails on a broken, unused or unchecked citation.

const references = json('docs/references.json');
const refCheck = json('docs/references-check.json');
const refNumber = Object.fromEntries(references.map((r, i) => [r.id, i + 1]));

function referenceTable(group) {
  const rows = references.filter((r) => r.group === group).map((r) => {
    const check = refCheck.results[r.id];
    if (!check) throw new Error(`reference ${r.id} has no check result: run docs/check_refs.py`);
    if (check.status === 'FAILED') throw new Error(`reference ${r.id} failed its check`);
    const source = [r.authors, r.year].filter(Boolean).join(', ');
    return `<tr id="ref-${r.id}"><td class="n">${refNumber[r.id]}</td><td>${esc(source)}${source ? '. ' : ''}<a href="${esc(r.url)}">${esc(r.title)}</a></td><td>${esc(check.status)}</td></tr>`;
  }).join('');
  return `<div class="scroll"><table><thead><tr><th class="n">No.</th><th>Reference</th><th>Check result</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

let html = read('docs/source.html');
const cited = new Set();
html = html.replace(/\{\{cite:([\w-]+)\}\}/g, (_, id) => {
  if (!refNumber[id]) throw new Error(`unknown citation ${id}`);
  cited.add(id);
  return `<a class="cite" href="#ref-${id}">[${refNumber[id]}]</a>`;
});
const uncited = references.filter((r) => r.group === 'manual' && !cited.has(r.id)).map((r) => r.id);
if (uncited.length) throw new Error(`references never cited: ${uncited}`);
values.REFS_MANUAL = referenceTable('manual');
values.REFS_POST = referenceTable('post');
values.REFS_REPORT = referenceTable('report');
values.REFS_TOTAL = references.length;
values.REFS_DATE = refCheck.checked;
html = html.replace(/\{\{([A-Z0-9_]+)\}\}/g, (_, key) => {
  if (!(key in values)) throw new Error(`unknown placeholder ${key}`);
  return values[key];
});
fs.writeFileSync(path.join(ROOT, 'docs/index.html'), html);
console.log(`docs/index.html ${(html.length / 1024).toFixed(0)} KB`);
