// Three Pads: the complete game rules. The browser, the scorer and the prompt all use this file.
(function (root) {
  const WIDTH = 100, HEIGHT = 60, CEILING = 80;
  const PADS = [20, 50, 80], PAD_HALF_WIDTH = 4;
  const GRAVITY = 0.01, MAIN_THRUST = 0.025, SIDE_THRUST = 0.008;
  const FUEL = 300, MAIN_BURN = 1, SIDE_BURN = 0.5;
  const SAFE_VX = 0.1, SAFE_VY = 0.25;
  const MAX_TICKS = 1000, STICKY = 0.1;
  const ACTIONS = ['none', 'thrust', 'left', 'right'];

  function reset(rng, target) {
    return {
      x: 10 + 80 * rng(),
      y: 50 + 8 * rng(),
      vx: -0.2 + 0.4 * rng(),
      vy: -0.1 * rng(),
      fuel: FUEL,
      target,
      tick: 0,
      lastAction: 0,
      outcome: null,
    };
  }

  // The network's input: 8 numbers.
  function observe(s) {
    return [s.x / WIDTH, s.y / HEIGHT, s.vx, s.vy, s.fuel / FUEL,
      s.target === 0 ? 1 : 0, s.target === 1 ? 1 : 0, s.target === 2 ? 1 : 0];
  }

  // Advances one tick. An engine fires only if fuel is left. Exactly one action happens per tick.
  function step(s, action, rng) {
    if (s.outcome) return s;
    if (rng() < STICKY) action = s.lastAction;
    s.lastAction = action;
    let ax = 0, ay = -GRAVITY;
    if (action === 1 && s.fuel >= MAIN_BURN) { ay += MAIN_THRUST; s.fuel -= MAIN_BURN; }
    if (action === 2 && s.fuel >= SIDE_BURN) { ax -= SIDE_THRUST; s.fuel -= SIDE_BURN; }
    if (action === 3 && s.fuel >= SIDE_BURN) { ax += SIDE_THRUST; s.fuel -= SIDE_BURN; }
    s.vx += ax;
    s.vy += ay;
    s.x += s.vx;
    s.y += s.vy;
    s.tick += 1;
    if (s.y <= 0) {
      s.y = 0;
      const pad = PADS.findIndex((p) => Math.abs(s.x - p) <= PAD_HALF_WIDTH);
      const soft = Math.abs(s.vx) <= SAFE_VX && Math.abs(s.vy) <= SAFE_VY;
      if (pad < 0) s.outcome = 'off_pad';
      else if (!soft) s.outcome = 'crashed';
      else s.outcome = pad === s.target ? 'landed' : 'wrong_pad';
    } else if (s.x < 0 || s.x > WIDTH || s.y > CEILING) {
      s.outcome = 'lost';
    } else if (s.tick >= MAX_TICKS) {
      s.outcome = 'timeout';
    }
    return s;
  }

  const api = { WIDTH, HEIGHT, CEILING, PADS, PAD_HALF_WIDTH, GRAVITY, MAIN_THRUST, SIDE_THRUST,
    FUEL, MAIN_BURN, SIDE_BURN, SAFE_VX, SAFE_VY, MAX_TICKS, STICKY, ACTIONS, reset, observe, step };
  if (typeof module !== 'undefined') module.exports = api;
  else root.Sim = api;
})(globalThis);
