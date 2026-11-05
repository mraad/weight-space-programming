#!/usr/bin/env node
// Is the task too easy? Scores pilots from starts the prompt never described, and on seeds the scorer never uses.
// Usage: node lander/stress.js <net.json>... Prints JSON. Changes start states only, never the rules in sim.js.
const fs = require('fs');
const Policy = require('./policy.js');

const EPISODES = 3000;
const SEED = 7000000;
const SCENARIOS = {
  'fresh seeds, normal start': () => ({}),
  'fuel 150': () => ({ fuel: 150 }),
  'fuel 100': () => ({ fuel: 100 }),
  'fuel 60': () => ({ fuel: 60 }),
  'start speed vx +-0.6': (rng) => ({ vx: -0.6 + 1.2 * rng() }),
  'start speed vx +-1.0': (rng) => ({ vx: -1 + 2 * rng() }),
  'start falling vy -1.0': () => ({ vy: -1 }),
  'start low y = 15': () => ({ y: 15 }),
  'start low y = 8': () => ({ y: 8 }),
  'start edge, far pad': (rng, target) => ({ x: target === 2 ? 1 : 99 }),
};

function score(pilot, change) {
  const outcomes = {};
  for (let i = 0; i < EPISODES; i++) {
    const { outcome } = Policy.rollout(pilot, SEED + i, i % 3, { change });
    outcomes[outcome] = (outcomes[outcome] || 0) + 1;
  }
  return { landed: (outcomes.landed || 0) / EPISODES, outcomes };
}

const pilots = { 'Plain program': Policy.controllerPilot };
process.argv.slice(2).sort().forEach((path, i) => {
  pilots[`Claude reply ${'ABCDEFGH'[i]}`] = Policy.netPilot(JSON.parse(fs.readFileSync(path, 'utf8')));
});
const table = {};
for (const [name, change] of Object.entries(SCENARIOS)) {
  table[name] = Object.fromEntries(Object.entries(pilots).map(([p, pilot]) => [p, score(pilot, change)]));
}
console.log(JSON.stringify({ episodes: EPISODES, seed: SEED, table }, null, 1));
