// Pilots and scoring. Not shown to the model that writes the weights: it gets sim.js only.
(function (root) {
  const Sim = typeof module !== 'undefined' ? require('./sim.js') : root.Sim;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function layer(input, W, b, activation) {
    return b.map((bias, j) => {
      let sum = bias;
      for (let i = 0; i < input.length; i++) sum += input[i] * W[i][j];
      return activation(sum);
    });
  }

  // h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3
  function forward(net, x) {
    const h1 = layer(x, net.W1, net.b1, Math.tanh);
    const h2 = layer(h1, net.W2, net.b2, Math.tanh);
    const logits = layer(h2, net.W3, net.b3, (v) => v);
    const top = Math.max(...logits);
    const raw = logits.map((l) => Math.exp(l - top));
    const total = raw.reduce((a, b) => a + b, 0);
    return { h1, h2, logits, probs: raw.map((p) => p / total) };
  }

  function sample(probs, rng) {
    let r = rng();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  const INPUT_NAMES = [`x / ${Sim.WIDTH}`, `y / ${Sim.HEIGHT}`, 'vx', 'vy', `fuel / ${Sim.FUEL}`,
    ...Sim.PADS.map((_, i) => `target pad ${i + 1}`)];

  const netPilot = (net) => (s, rng) => sample(forward(net, Sim.observe(s)).probs, rng);
  const randomPilot = (_s, rng) => Math.floor(rng() * Sim.ACTIONS.length);

  // A plain program, the fair baseline for written weights: steer over the pad, hold height until there, then descend.
  function controllerPilot(s) {
    const dx = Sim.PADS[s.target] - s.x;
    const wantVx = Math.max(-0.45, Math.min(0.45, dx * 0.025));
    const overPad = Math.abs(dx) <= 2 && Math.abs(s.vx) <= 0.06;
    const wantVy = overPad || s.y > 25 ? -Math.min(0.2 + 0.012 * s.y, 0.6) : 0;
    if (s.vy < wantVy) return 1;
    if (s.vx < wantVx - 0.02) return 3;
    if (s.vx > wantVx + 0.02) return 2;
    return 0;
  }

  const PILOTS = { random: randomPilot, controller: controllerPilot };

  // change overrides the start state (stress test). onStep sees the state after each tick (figures).
  function rollout(pilot, seed, target, { switchAt = 0, change, onStep } = {}) {
    const rng = mulberry32(seed);
    const s = Sim.reset(rng, target);
    if (change) Object.assign(s, change(rng, target));
    while (!s.outcome) {
      if (switchAt && s.tick === switchAt) s.target = (target + 1 + (seed % 2)) % 3;
      Sim.step(s, pilot(s, rng), rng);
      if (onStep) onStep(s);
    }
    return s;
  }

  // Episodes cycle through the three pads, so every pad gets the same share.
  function evaluate(pilot, { episodes = 3000, seed = 1000000, midFlightSwitch = false } = {}) {
    const outcomes = { landed: 0, wrong_pad: 0, crashed: 0, off_pad: 0, lost: 0, timeout: 0 };
    let fuel = 0, ticks = 0;
    for (let i = 0; i < episodes; i++) {
      const target = i % 3;
      const switchAt = midFlightSwitch ? 30 + (i * 7) % 91 : 0;
      const s = rollout(pilot, seed + i, target, { switchAt });
      outcomes[s.outcome] += 1;
      if (s.outcome === 'landed') { fuel += s.fuel; ticks += s.tick; }
    }
    const landed = outcomes.landed / episodes;
    return { episodes, landed, se: Math.sqrt(landed * (1 - landed) / episodes), outcomes,
      meanFuelLeft: outcomes.landed ? fuel / outcomes.landed : null, meanTicks: outcomes.landed ? ticks / outcomes.landed : null };
  }

  const api = { mulberry32, forward, sample, INPUT_NAMES, PILOTS, netPilot, randomPilot, controllerPilot, rollout, evaluate };
  if (typeof module !== 'undefined') module.exports = api;
  else root.Policy = api;
})(globalThis);
