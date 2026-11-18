"use strict";

const Retina = (() => {
  const dt = 10;
  const duration = 2200;
  const steps = duration / dt + 1;
  const types = [
    { key: "cone", label: "Cone", color: "#eeb65e", depth: 1.12, tau: 30 },
    { key: "hc", label: "Horizontal", color: "#e49379", depth: 0.49, tau: 58 },
    { key: "on", label: "ON bipolar", color: "#80d8b6", depth: 0.27, tau: 54 },
    { key: "off", label: "OFF bipolar", color: "#56b7be", depth: 0.08, tau: 36 },
    { key: "ac", label: "Amacrine", color: "#b3a1f0", depth: -0.22, tau: 30 },
    { key: "rgcOn", label: "ON ganglion", color: "#8caeff", depth: -0.82, tau: 45 },
    { key: "rgcOff", label: "OFF ganglion", color: "#c4a7f2", depth: -0.93, tau: 45 }
  ];
  const coneClasses = [
    { key: "UV", color: "#ba9df7", uv: 1.0 },
    { key: "Blue", color: "#75acfc", uv: 0.20 },
    { key: "Green", color: "#68cea0", uv: 0.035 },
    { key: "Red", color: "#f0a561", uv: 0.005 }
  ];
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const distance = (a, b) => (a.x - b.x) ** 2 + (a.z - b.z) ** 2;

  function makeGraph(seed = 17) {
    let state = seed >>> 0;
    const random = () => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const nodes = [];
    const edges = [];
    function place(type, columns, rows) {
      const info = types[type];
      return Array.from({ length: columns * rows }, (_, i) => {
        const node = {
          id: nodes.length, type,
          x: ((i % columns + 0.5) / columns * 2 - 1) * 1.49 + (random() - 0.5) * 0.23,
          z: ((Math.floor(i / columns) + 0.5) / rows * 2 - 1) * 1.03 + (random() - 0.5) * 0.20,
          y: info.depth + (random() - 0.5) * 0.08,
          cone: type === 0 ? (i + Math.floor(i / columns)) % coneClasses.length : -1
        };
        nodes.push(node);
        return node;
      });
    }
    const cones = place(0, 8, 4);
    const horizontal = place(1, 3, 2);
    const on = place(2, 4, 3);
    const off = place(3, 4, 3);
    const amacrine = place(4, 4, 2);
    const ganglionOn = place(5, 3, 2);
    const ganglionOff = place(6, 3, 2);

    function connect(sources, targets, count, gain, contact) {
      for (const target of targets) {
        const nearest = sources.slice().sort((a, b) => distance(a, target) - distance(b, target)).slice(0, count);
        const strengths = nearest.map(source => Math.exp(-1.2 * distance(source, target)) * (0.85 + random() * 0.30));
        const total = strengths.reduce((sum, value) => sum + value, 0);
        nearest.forEach((source, i) => {
          edges.push({
            from: source.id, to: target.id, weight: gain * strengths[i] / total,
            contact, threshold: random()
          });
        });
      }
    }
    connect(cones, horizontal, 6, 0.69, "invaginating");
    connect(horizontal, cones, 2, -0.17, "feedback");
    connect(cones, on, 4, -1.28, "invaginating");
    connect(cones, off, 4, 1.22, "basal");
    connect(on, amacrine, 3, 0.60, "inner");
    connect(off, amacrine, 3, 0.54, "inner");
    connect(on, ganglionOn, 4, 1.46, "inner");
    connect(off, ganglionOff, 4, 1.46, "inner");
    connect(amacrine, ganglionOn, 2, -0.52, "inner");
    connect(amacrine, ganglionOff, 2, -0.33, "inner");
    return { nodes, edges };
  }

  function stimulusAt(stimulus, time, x) {
    if (time < 450 || time >= 1150) return 0;
    if (stimulus === "bar") {
      const center = -1.9 + (time - 450) / 700 * 3.8;
      return Math.exp(-0.5 * ((x - center) / 0.25) ** 2);
    }
    return 1;
  }

  function makeCase(graph, { stimulus = "white", mode = "wild", stress = 0, contactLoss = 0 } = {}) {
    const { nodes, edges } = graph;
    const n = nodes.length;
    const coneCount = nodes.filter(node => node.cone >= 0).length;
    const offBipolars = nodes.filter(node => node.type === 3);
    const count = n + coneCount + offBipolars.length;
    const alpha = nodes.map(node => dt / types[node.type].tau);
    const matrix = Array.from({ length: count }, () => Array(count).fill(0));
    const contacts = edges.map(edge => {
      const coneOutput = nodes[edge.from].cone >= 0;
      const invaginating = edge.contact === "invaginating";
      const active = !(mode === "contacts" && invaginating && edge.threshold < contactLoss);
      const efficacy = mode !== "wild" && coneOutput ? 1 - stress * (invaginating ? 0.83 : 0.27) : 1;
      if (active) matrix[edge.to][coneOutput ? n + edge.from : edge.from] += edge.weight * efficacy;
      return { ...edge, active, effective: active ? edge.weight * efficacy : 0 };
    });
    for (let i = 0; i < coneCount; i++) {
      matrix[n + i][i] = 1;
      alpha.push(dt / (25 + (mode === "wild" ? 0 : 95 * stress)));
    }
    // Slow adaptation to sustained input allows a transient response to light offset.
    offBipolars.forEach((node, i) => {
      const adaptation = n + coneCount + i;
      matrix[adaptation] = matrix[node.id].slice();
      matrix[node.id][adaptation] = -0.86;
      alpha.push(dt / 250);
    });
    const input = Array.from({ length: steps }, (_, step) => {
      const values = Array(count).fill(0);
      nodes.slice(0, coneCount).forEach(node => {
        const sensitivity = stimulus === "uv" ? coneClasses[node.cone].uv : 1;
        values[node.id] = -stimulusAt(stimulus, step * dt, node.x) * sensitivity;
      });
      return values;
    });
    return { nodes, contacts, alpha, matrix, input };
  }

  function simulate(scenario) {
    const { nodes, alpha, matrix, input } = scenario;
    const sparse = matrix.map(row => row.map((weight, index) => ({ weight, index })).filter(term => term.weight));
    let previous = new Float64Array(alpha.length);
    const frames = input.map(values => {
      const next = Float64Array.from(previous, (value, id) => {
        const drive = sparse[id].reduce((sum, term) => sum + term.weight * previous[term.index], values[id]);
        return clamp(value + alpha[id] * (clamp(drive, -2, 2) - value), -2, 2);
      });
      previous = next;
      return nodes.map(node => node.type >= 5 ? clamp(6 + 33 * next[node.id], 0, 80) : next[node.id]);
    });
    return { ...scenario, frames };
  }

  function summarize(result) {
    const group = type => result.nodes.filter(node => node.type === type).map(node => node.id);
    const peak = (ids, lo, hi) => Math.max(...result.frames.slice(lo / dt, hi / dt + 1).map(frame =>
      ids.reduce((sum, id) => sum + frame[id], 0) / ids.length
    ));
    return {
      total: result.contacts.filter(edge => edge.active).length,
      invaginating: result.contacts.filter(edge => edge.active && edge.contact === "invaginating").length,
      on: peak(group(5), 450, 1150),
      off: peak(group(6), 450, 1800)
    };
  }

  return { dt, duration, steps, types, coneClasses, makeGraph, stimulusAt, makeCase, simulate, summarize };
})();

if (typeof module === "object" && module.exports) module.exports = Retina;
