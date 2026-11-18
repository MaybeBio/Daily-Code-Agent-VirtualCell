"use strict";

const $ = id => document.getElementById(id);
const graph = Retina.makeGraph();
const wildColor = "#71dec2";
const mutantColor = "#f4ad78";
const abbreviations = ["Cone", "HC", "ON BC", "OFF BC", "AC", "ON GC", "OFF GC"];
const retinaCanvases = [$("wildCanvas"), $("mutantCanvas")];
const initialView = { yaw: -0.37, pitch: 0.40 };
const view = { ...initialView };
const positions = new Map();
let results = [];
let pathway = null;
let selectedCell = null;
let animation = 0;
let scheduled = 0;
let lastDraw = null;

function sizeCanvas(canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  const pixelsWide = Math.round(width * ratio);
  const pixelsHigh = Math.round(height * ratio);
  if (canvas.width !== pixelsWide || canvas.height !== pixelsHigh) {
    canvas.width = pixelsWide;
    canvas.height = pixelsHigh;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function stroke(ctx, points, color, width = 1) {
  ctx.beginPath();
  points.forEach(({ x, y }, i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function alpha(color, opacity) {
  return color + Math.round(255 * Math.min(1, Math.max(0, opacity))).toString(16).padStart(2, "0");
}

function cellColor(node) {
  return node.cone >= 0 ? Retina.coneClasses[node.cone].color : Retina.types[node.type].color;
}

function drawRetina(canvas, result, frameIndex) {
  const { context: ctx, width, height } = sizeCanvas(canvas);
  const margin = width < 380 ? 36 : 57;
  const scale = Math.min((width - 2 * margin) / 3.55, (height - 38) / 3.15);
  const project = (x, y, z) => {
    const turnedX = x * Math.cos(view.yaw) - z * Math.sin(view.yaw);
    const turnedZ = x * Math.sin(view.yaw) + z * Math.cos(view.yaw);
    return {
      x: width * 0.49 + scale * turnedX,
      y: height * 0.5 + scale * (turnedZ * Math.sin(view.pitch) - y * Math.cos(view.pitch)),
      depth: turnedZ * Math.cos(view.pitch) + y * Math.sin(view.pitch)
    };
  };

  const planes = [
    { depth: 1.10, name: "Cones", color: "#eeb65e" },
    { depth: 0.76, name: "OPL", color: "#eab682" },
    { depth: 0.35, name: "INL", color: "#80d8b6" },
    { depth: -0.49, name: "IPL", color: "#b3a1f0" },
    { depth: -0.91, name: "GCL", color: "#8caeff" }
  ];
  for (const plane of planes.slice().reverse()) {
    const points = [[-1.62,-1.09], [1.62,-1.09], [1.62,1.09], [-1.62,1.09], [-1.62,-1.09]]
      .map(([x,z]) => project(x, plane.depth, z));
    ctx.beginPath();
    points.forEach(({x,y}, i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    ctx.fillStyle = alpha(plane.color, 0.045);
    ctx.fill();
    ctx.strokeStyle = alpha(plane.color, 0.32);
    ctx.stroke();
    for (const column of [-0.81, 0, 0.81]) {
      stroke(ctx, [project(column, plane.depth, -1.09), project(column, plane.depth, 1.09)], alpha(plane.color, 0.12));
    }
  }

  const mode = $("connections").value;
  const points = result.nodes.map(node => project(node.x, node.y, node.z));
  if (mode !== "none") {
    for (const edge of result.contacts) {
      if (!edge.active) continue;
      const from = result.nodes[edge.from];
      const to = result.nodes[edge.to];
      const highlighted = pathway ? from.type === pathway.from && to.type === pathway.to
        : selectedCell !== null && (from.id === selectedCell || to.id === selectedCell);
      const dimmed = (pathway || selectedCell !== null) && !highlighted;
      const magnitude = mode === "effective" ? Math.min(1, Math.abs(edge.effective) * 3.5) : 0.75;
      const opacity = magnitude * (highlighted ? 0.85 : dimmed ? 0.055 : 0.20);
      const depth = edge.contact === "inner" ? -0.49 : 0.76;
      const middle = project((from.x + to.x) / 2, depth, (from.z + to.z) / 2);
      stroke(ctx, [points[from.id], middle, points[to.id]], alpha(cellColor(from), opacity), highlighted ? 1.65 : 0.9);
    }
  }

  const cells = result.nodes.map((node, id) => ({ node, ...points[id] })).sort((a,b) => a.depth - b.depth);
  const values = result.frames[frameIndex];
  for (const cell of cells) {
    const { node, x, y } = cell;
    const color = cellColor(node);
    const strength = node.type >= 5 ? Math.abs(values[node.id] - 6) / 35 : Math.abs(values[node.id]);
    const radius = (node.type === 0 ? 3.6 : 3.1) + Math.min(strength, 1) * 2;
    ctx.beginPath();
    ctx.arc(x, y, radius + 5 * Math.min(strength, 1), 0, 2 * Math.PI);
    ctx.fillStyle = alpha(color, Math.min(strength, 1) * 0.18);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = alpha(color, 0.66 + Math.min(strength, 1) * 0.34);
    ctx.fill();
    if (node.id === selectedCell) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = "#f7dfab";
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
  }
  positions.set(canvas, cells);

  for (const plane of planes) {
    const end = project(1.62, plane.depth, -1.09);
    const x = Math.min(width - 33, end.x + 9);
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = plane.color;
    ctx.fillText(plane.name, x, end.y);
  }
}

function traceSpec() {
  const selected = $("trace").value;
  if (selected.startsWith("cell-")) {
    const id = Number(selected.slice(5));
    return { ids: [id], rate: graph.nodes[id].type >= 5 };
  }
  const type = Number(selected);
  return { ids: graph.nodes.filter(node => node.type === type).map(node => node.id), rate: type >= 5 };
}

function drawTrace(time) {
  const { context: ctx, width, height } = sizeCanvas($("traceCanvas"));
  const { ids, rate } = traceSpec();
  const series = results.map(result => result.frames.map(frame => ids.reduce((sum, id) => sum + frame[id], 0) / ids.length));
  const values = series.flat();
  const low = rate ? 0 : Math.floor(Math.min(0, ...values) * 5) / 5;
  const high = rate ? Math.max(10, Math.ceil(Math.max(...values) / 10) * 10)
    : Math.max(0.2, Math.ceil(Math.max(...values) * 5) / 5);
  const area = { left: 50, top: 17, right: width - 16, bottom: height - 27 };
  const x = t => area.left + t / Retina.duration * (area.right - area.left);
  const y = value => area.bottom - (value - low) / (high - low) * (area.bottom - area.top);
  ctx.fillStyle = "#71dec21c";
  ctx.fillRect(x(450), area.top, x(1150) - x(450), area.bottom - area.top);
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const level = low + (high - low) * i / 4;
    const yy = y(level);
    stroke(ctx, [{ x: area.left, y: yy }, { x: area.right, y: yy }], "#334156");
    ctx.textAlign = "right";
    ctx.fillStyle = "#aab9cf";
    ctx.fillText(level.toFixed(rate && level % 1 === 0 ? 0 : 1), area.left - 7, yy);
  }
  ctx.textAlign = "left";
  ctx.fillText(rate ? "Hz" : "Δ a.u.", 3, 6);
  ctx.textAlign = "center";
  for (const t of [0, 500, 1000, 1500, 2000]) ctx.fillText(t.toString(), x(t), height - 12);
  ctx.textAlign = "right";
  ctx.fillText("ms", width - 2, height - 12);
  series.forEach((trace, i) => {
    ctx.setLineDash(i ? [5, 3] : []);
    stroke(ctx, trace.map((value, step) => ({ x: x(step * Retina.dt), y: y(value) })), i ? mutantColor : wildColor, 2);
  });
  ctx.setLineDash([]);
  stroke(ctx, [{ x: x(time), y: area.top }, { x: x(time), y: area.bottom }], "#e9edf7a0");
}

function draw() {
  if (results.length !== 2) return;
  const time = Number($("time").value);
  const frame = Math.round(time / Retina.dt);
  retinaCanvases.forEach((canvas, i) => drawRetina(canvas, results[i], frame));
  $("timeValue").value = time + " ms";
  const lightOn = time >= 450 && time < 1150;
  $("lightState").textContent = lightOn ? "Light on" : "Light off";
  $("lightState").classList.toggle("on", lightOn);
  drawTrace(time);
}

function updateMatrix() {
  const effective = $("matrixView").value === "effective";
  $("matrixNote").textContent = effective
    ? "Mean absolute gain per baseline contact. Wild type → simulated mutant."
    : "Active contacts from row to column. Wild type → simulated mutant.";
  const table = document.createElement("table");
  table.setAttribute("aria-label", effective ? "Mean transmission gain by cell type" : "Contact counts by cell type");
  const header = table.createTHead().insertRow();
  for (const label of ["From / to", ...abbreviations]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    header.append(th);
  }
  const body = table.createTBody();
  $("matrixReadout").textContent = "Select a pathway to highlight its contacts.";
  Retina.types.forEach((source, from) => {
    const row = body.insertRow();
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = abbreviations[from];
    row.append(th);
    Retina.types.forEach((target, to) => {
      const cell = row.insertCell();
      const pair = results.map(result => result.contacts.filter(edge =>
        graph.nodes[edge.from].type === from && graph.nodes[edge.to].type === to
      ));
      if (!pair[0].length) {
        cell.className = "empty";
        cell.textContent = "·";
        return;
      }
      const value = contacts => effective
        ? contacts.reduce((sum, edge) => sum + Math.abs(edge.effective), 0) / contacts.length
        : contacts.filter(edge => edge.active).length;
      const display = pair.map(contacts => effective ? value(contacts).toFixed(2) : value(contacts));
      const description = source.label + " → " + target.label + " · " + display.join(" → ");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "matrix-cell";
      button.textContent = display.join("→");
      button.setAttribute("aria-label", description);
      button.style.backgroundColor = alpha("#486f86", Math.min(0.60, 0.16 + value(pair[0]) / (effective ? 1 : 95)));
      const active = pathway?.from === from && pathway?.to === to;
      button.classList.toggle("selected", active);
      button.setAttribute("aria-pressed", String(active));
      if (active) $("matrixReadout").textContent = description;
      button.addEventListener("click", () => {
        pathway = active ? null : { from, to };
        if (pathway && $("connections").value === "none") $("connections").value = "anatomy";
        updateMatrix();
        $("matrix").querySelectorAll("tr")[from + 1].cells[to + 1].firstChild.focus();
        draw();
      });
      cell.append(button);
    });
  });
  $("matrix").replaceChildren(table);
}

function updateResults() {
  const mode = $("mode").value;
  const stress = Number($("stress").value);
  const loss = Number($("loss").value);
  $("stressValue").value = stress + "%";
  $("loss").disabled = mode !== "contacts";
  $("lossValue").value = mode === "contacts" ? loss + "%" : "Off";
  const stimulus = $("stimulus").value;
  results = [
    Retina.simulate(Retina.makeCase(graph, { stimulus })),
    Retina.simulate(Retina.makeCase(graph, { stimulus, mode, stress: stress / 100, contactLoss: loss / 100 }))
  ];
  const stats = results.map(Retina.summarize);
  ["wildCount", "mutantCount"].forEach((id, i) => {
    $(id).textContent = graph.nodes.length + " cells · " + stats[i].total + " contacts";
  });
  const metrics = [
    { label: "ON ganglion peak · 450–1,150 ms", key: "on" },
    { label: "OFF ganglion peak · 450–1,800 ms", key: "off" }
  ];
  $("metrics").replaceChildren(...metrics.map(metric => {
    const card = document.createElement("div");
    card.className = "metric";
    const label = document.createElement("span");
    label.textContent = metric.label;
    const value = document.createElement("strong");
    value.textContent = stats.map(s => s[metric.key].toFixed(1)).join(" → ") + " Hz";
    card.append(label, value);
    return card;
  }));
  updateMatrix();
  draw();
}

function scheduleResults() {
  if (scheduled) return;
  scheduled = requestAnimationFrame(() => {
    scheduled = 0;
    updateResults();
  });
}

function selectCell(id) {
  const select = $("trace");
  const previous = select.querySelector('[value^="cell-"]');
  if (previous) previous.remove();
  const node = graph.nodes[id];
  const name = node.cone >= 0 ? Retina.coneClasses[node.cone].key + " cone" : Retina.types[node.type].label;
  select.add(new Option(name + " #" + (id + 1), "cell-" + id));
  select.value = "cell-" + id;
  selectedCell = id;
  pathway = null;
  updateMatrix();
  draw();
}

for (const canvas of retinaCanvases) {
  let gesture = null;
  canvas.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    gesture = { x: event.clientX, y: event.clientY, yaw: view.yaw, pitch: view.pitch, moved: false };
  });
  canvas.addEventListener("pointermove", event => {
    if (!gesture) return;
    const dx = event.clientX - gesture.x;
    const dy = event.clientY - gesture.y;
    if (Math.hypot(dx, dy) > 4) gesture.moved = true;
    if (!gesture.moved) return;
    view.yaw = Math.max(-0.95, Math.min(0.95, gesture.yaw + dx * 0.006));
    view.pitch = Math.max(0.08, Math.min(0.92, gesture.pitch + dy * 0.006));
    draw();
  });
  canvas.addEventListener("pointerup", event => {
    if (gesture && !gesture.moved) {
      const box = canvas.getBoundingClientRect();
      const px = event.clientX - box.left;
      const py = event.clientY - box.top;
      const hit = (positions.get(canvas) || []).slice().reverse().find(({ x, y }) => Math.hypot(x - px, y - py) < 9);
      if (hit) selectCell(hit.node.id);
    }
    gesture = null;
  });
  canvas.addEventListener("pointercancel", () => { gesture = null; });
}

function tick(now) {
  if (lastDraw === null) lastDraw = now;
  const elapsed = now - lastDraw;
  if (elapsed >= 32) {
    const timeInput = $("time");
    const advanced = Number(timeInput.value) + Math.round(elapsed / Retina.dt) * Retina.dt;
    timeInput.value = advanced % (Retina.duration + Retina.dt);
    lastDraw = now;
    draw();
  }
  animation = requestAnimationFrame(tick);
}

$("play").addEventListener("click", () => {
  if (animation) {
    cancelAnimationFrame(animation);
    animation = 0;
  } else {
    lastDraw = null;
    animation = requestAnimationFrame(tick);
  }
  $("play").textContent = animation ? "Pause" : "Play";
});
$("resetView").addEventListener("click", () => {
  Object.assign(view, initialView);
  draw();
});
$("trace").addEventListener("change", () => {
  const value = $("trace").value;
  selectedCell = value.startsWith("cell-") ? Number(value.slice(5)) : null;
  draw();
});
for (const control of ["stress", "loss", "stimulus", "mode"]) $(control).addEventListener("input", scheduleResults);
for (const control of ["time", "connections"]) $(control).addEventListener("input", draw);
$("matrixView").addEventListener("input", updateMatrix);

Retina.types.forEach((type, i) => $("trace").add(new Option(type.label + " mean", i)));
$("trace").value = "5";
const legend = [...Retina.coneClasses.map(type => ({ ...type, label: type.key + " cone" })), ...Retina.types.slice(1)];
for (const type of legend) {
  const label = document.createElement("span");
  const dot = document.createElement("i");
  dot.className = "tag";
  dot.style.backgroundColor = type.color;
  label.append(dot, type.label);
  $("cellLegend").append(label);
}
new ResizeObserver(draw).observe(document.querySelector("main"));
updateResults();
