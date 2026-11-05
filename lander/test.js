// Run with: node --test lander/test.js
const test = require('node:test');
const assert = require('node:assert');
const Sim = require('./sim.js');
const Policy = require('./policy.js');

const scene = (overrides) => ({ ...Sim.reset(Policy.mulberry32(1), 1), ...overrides });
const quiet = () => 0.5; // never sticky

function touchdown(overrides) {
  return Sim.step(scene({ y: 0.001, vx: 0, vy: 0, ...overrides }), 0, quiet).outcome;
}

test('landing rules: every outcome', () => {
  assert.equal(touchdown({ x: 50, target: 1 }), 'landed');
  assert.equal(touchdown({ x: 54, target: 1 }), 'landed');
  assert.equal(touchdown({ x: 55, target: 1 }), 'off_pad');
  assert.equal(touchdown({ x: 20, target: 1 }), 'wrong_pad');
  assert.equal(touchdown({ x: 50, target: 1, vy: -0.5 }), 'crashed');
  assert.equal(touchdown({ x: 50, target: 1, vx: 0.2 }), 'crashed');
  assert.equal(Sim.step(scene({ x: 99.99, vx: 1 }), 0, quiet).outcome, 'lost');
  assert.equal(Sim.step(scene({ y: 79.99, vy: 1 }), 0, quiet).outcome, 'lost');
  assert.equal(Sim.step(scene({ tick: Sim.MAX_TICKS - 1 }), 0, quiet).outcome, 'timeout');
});

test('engines burn fuel and stop when it runs out', () => {
  const burn = Sim.step(scene({ vy: 0, vx: 0 }), 1, quiet);
  assert.equal(burn.fuel, Sim.FUEL - Sim.MAIN_BURN);
  assert.ok(Math.abs(burn.vy - (Sim.MAIN_THRUST - Sim.GRAVITY)) < 1e-12);
  const side = Sim.step(scene({ vy: 0, vx: 0 }), 2, quiet);
  assert.equal(side.fuel, Sim.FUEL - Sim.SIDE_BURN);
  assert.ok(side.vx < 0);
  const empty = Sim.step(scene({ fuel: 0, vy: 0, vx: 0 }), 1, quiet);
  assert.equal(empty.fuel, 0);
  assert.ok(Math.abs(empty.vy + Sim.GRAVITY) < 1e-12);
});

test('sticky action repeats the previous action', () => {
  const s = Sim.step(scene({ lastAction: 3, vx: 0 }), 2, () => 0);
  assert.equal(s.lastAction, 3);
  assert.ok(s.vx > 0);
});

test('observation layout matches the prompt', () => {
  const x = Sim.observe(scene({ x: 50, y: 30, vx: 0.1, vy: -0.2, fuel: 150, target: 2 }));
  assert.deepEqual(x, [0.5, 0.5, 0.1, -0.2, 0.5, 0, 0, 1]);
});

test('forward pass matches a hand-computed network', () => {
  const zero = (r, c) => Array.from({ length: r }, () => Array(c).fill(0));
  const net = { W1: zero(8, 16), b1: Array(16).fill(0), W2: zero(16, 16), b2: Array(16).fill(0), W3: zero(16, 4), b3: [0, 0, 0, 0] };
  assert.deepEqual(Policy.forward(net, Array(8).fill(1)).probs, [0.25, 0.25, 0.25, 0.25]);
  net.W1[0][0] = 1; net.W2[0][0] = 1; net.W3[0][1] = 2;
  const out = Policy.forward(net, [1, 0, 0, 0, 0, 0, 0, 0]);
  assert.ok(Math.abs(out.logits[1] - 2 * Math.tanh(Math.tanh(1))) < 1e-12);
  net.b3[1] = 1000;
  assert.equal(Policy.sample(Policy.forward(net, Array(8).fill(0)).probs, () => 0.9), 1);
});

test('rollouts repeat exactly for a seed, and mid-flight switching moves the target', () => {
  assert.deepEqual(Policy.rollout(Policy.randomPilot, 5, 0), Policy.rollout(Policy.randomPilot, 5, 0));
  const s = Policy.rollout(Policy.controllerPilot, 4, 0, { switchAt: 30 });
  assert.notEqual(s.target, 0);
});

test('baselines bracket the task: random never lands, the plain controller nearly always does', () => {
  assert.equal(Policy.evaluate(Policy.randomPilot, { episodes: 300 }).landed, 0);
  assert.ok(Policy.evaluate(Policy.controllerPilot, { episodes: 300 }).landed > 0.97);
});
