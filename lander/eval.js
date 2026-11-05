#!/usr/bin/env node
// Usage: node lander/eval.js <net.json | random | controller> [episodes]. Prints JSON.
const fs = require('fs');
const Policy = require('./policy.js');

const [which, episodes = '3000'] = process.argv.slice(2);
if (!which) { console.error('usage: eval.js <net.json | random | controller> [episodes]'); process.exit(2); }
const pilot = Policy.PILOTS[which] || Policy.netPilot(JSON.parse(fs.readFileSync(which, 'utf8')));
const options = { episodes: Number(episodes) };
console.log(JSON.stringify({
  fixed_target: Policy.evaluate(pilot, options),
  mid_flight_switch: Policy.evaluate(pilot, { ...options, midFlightSwitch: true }),
}, null, 1));
