"use strict";

const Retina = require("./model.js");
const stimulusNames = ["white", "uv", "bar"];
const cases = [];

for (let seed = 17; seed < 33; seed++) {
  const graph = Retina.makeGraph(seed);
  for (const stimulus of stimulusNames) {
    const configurations = [
      { mode: "wild", stress: 0, contactLoss: 0 },
      { mode: "synaptic", stress: 0, contactLoss: 0 },
      { mode: "synaptic", stress: 0.65, contactLoss: 0 },
      { mode: "contacts", stress: 0.65, contactLoss: 0.40 }
    ];
    for (const config of configurations) {
      const scenario = Retina.makeCase(graph, { stimulus, ...config });
      const result = Retina.simulate(scenario);
      cases.push({
        seed, stimulus, ...config,
        alpha: scenario.alpha, matrix: scenario.matrix,
        input: scenario.input.map(frame => frame.slice(0, 32)),
        reference: Retina.summarize(result)
      });
    }
  }
}
const nodes = Retina.makeGraph().nodes;
process.stdout.write(JSON.stringify({
  dt: Retina.dt,
  on: nodes.filter(node => node.type === 5).map(node => node.id),
  off: nodes.filter(node => node.type === 6).map(node => node.id),
  cases
}));
