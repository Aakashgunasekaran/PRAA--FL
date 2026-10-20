(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const apiBase = (document.querySelector('meta[name="api-base"]')?.content || "").replace(/\/$/, "");
  const ALLOWED = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];
  let maxMb = 10;
  let objectUrl = null;
  let requestId = 0;

  const el = {
    drop: $("drop"), file: $("file"), choose: $("choose"), clear: $("clear"), preview: $("preview"),
    hint: $("hint"), busy: $("busy"), filename: $("filename"), status: $("status"), banner: $("banner"),
    error: $("error"), empty: $("empty"), result: $("result"), name: $("pred-name"), prob: $("pred-prob"),
    code: $("pred-code"), diseaseProb: $("disease-prob"), diseaseStatus: $("disease-status"),
    bars: $("bars"), seen: $("seen"), meta: $("meta"),
  };

  // ---- backend connection -------------------------------------------------
  async function loadHealth() {
    try {
      const res = await fetch(`${apiBase}/api/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const h = await res.json();
      maxMb = h.max_upload_mb || maxMb;
      el.status.dataset.state = h.mode;
      el.status.textContent = h.mode === "trained" ? `Model ready · ${h.description}` : "Demo model, not trained";
      if (h.mode === "demo" || (h.notes && h.notes.length)) {
        el.banner.hidden = false;
        el.banner.textContent = (h.notes && h.notes[0]) ||
          "This is an untrained demo model. Its predictions are not meaningful.";
      }
    } catch (err) {
      el.status.dataset.state = "offline";
      el.status.textContent = "Backend not reachable";
      showError("The backend did not respond. Start it with “python run_app.py” and reload this page.");
    }
  }

  async function classify(file) {
    const id = ++requestId;
    setBusy(true);
    hideError();
    const body = new FormData();
    body.append("file", file, file.name || "image");
    try {
      const res = await fetch(`${apiBase}/api/predict`, { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (id !== requestId) return;                       // a newer image replaced this one
      if (!res.ok) throw new Error(data.detail || `The server returned HTTP ${res.status}.`);
      render(data);
    } catch (err) {
      if (id !== requestId) return;
      showResultsEmpty();
      const offline = err instanceof TypeError;
      showError(offline ? "Could not reach the backend. Check that “python run_app.py” is still running." : err.message);
    } finally {
      if (id === requestId) setBusy(false);
    }
  }

  // ---- rendering ------------------------------------------------------------
  const pct = (p) => (p >= 0.9995 ? "100%" : p < 0.0005 ? "0%" : (p * 100).toFixed(1) + "%");

  function render(d) {
    const sorted = [...d.probabilities].sort((a, b) => b.probability - a.probability);
    const diseaseClass = d.probabilities.find((c) => c.class_id === 8);
    if (!diseaseClass) throw new Error("The server response is missing the adenocarcinoma tissue class.");
    el.name.textContent = d.predicted.name;
    el.prob.textContent = pct(d.predicted.probability);
    el.code.textContent = d.predicted.code;
    el.diseaseProb.textContent = pct(diseaseClass.probability);
    el.diseaseStatus.textContent = d.predicted.class_id === diseaseClass.class_id
      ? "This is the model's top predicted tissue category."
      : "This is not the model's top predicted tissue category.";
    el.bars.replaceChildren(...sorted.map((c, i) => {
      const li = document.createElement("li");
      if (i === 0) li.className = "top";
      li.innerHTML = '<span class="label"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>';
      li.querySelector(".label").textContent = c.name;
      li.querySelector(".pct").textContent = pct(c.probability);
      requestAnimationFrame(() => { li.querySelector(".fill").style.width = (c.probability * 100) + "%"; });
      return li;
    }));
    el.seen.src = d.model_input_png;
    const rule = d.rule === "prototype" ? "nearest global prototype" : "classifier head";
    el.meta.textContent = `${d.input.original_size.join(" × ")} px image · ${rule} · ${d.inference_ms} ms` +
      (d.demo ? " · demo model" : "");
    el.empty.hidden = true;
    el.result.hidden = false;
  }

  function showResultsEmpty() { el.result.hidden = true; el.empty.hidden = false; }
  function showError(msg) { el.error.textContent = msg; el.error.hidden = false; }
  function hideError() { el.error.hidden = true; }
  function setBusy(on) { el.busy.hidden = !on; el.drop.setAttribute("aria-busy", on ? "true" : "false"); }

  // ---- input handling -------------------------------------------------------
  function handleFile(file) {
    if (!file) return;
    if (!ALLOWED.includes(file.type)) {
      showError("That file type is not supported. Use a PNG, JPG, BMP, TIFF or WEBP image.");
      return;
    }
    if (file.size > maxMb * 1024 * 1024) {
      showError(`That image is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${maxMb} MB.`);
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    el.preview.src = objectUrl;
    el.preview.hidden = false;
    el.hint.hidden = true;
    el.drop.classList.add("has-image");
    el.filename.textContent = file.name || "Pasted image";
    el.clear.disabled = false;
    classify(file);
  }

  function clearAll() {
    requestId++;                                         // ignore any request still in flight
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    el.preview.removeAttribute("src");
    el.preview.hidden = true;
    el.hint.hidden = false;
    el.drop.classList.remove("has-image");
    el.filename.textContent = "";
    el.file.value = "";
    el.clear.disabled = true;
    setBusy(false);
    hideError();
    showResultsEmpty();
  }

  el.choose.addEventListener("click", () => el.file.click());
  el.drop.addEventListener("click", () => el.file.click());
  el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); } });
  el.file.addEventListener("change", () => handleFile(el.file.files[0]));
  el.clear.addEventListener("click", clearAll);
  ["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
  el.drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
  window.addEventListener("dragover", (e) => e.preventDefault());   // stop the browser opening a missed drop
  window.addEventListener("drop", (e) => e.preventDefault());
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) handleFile(item.getAsFile());
  });

  loadHealth();
})();
