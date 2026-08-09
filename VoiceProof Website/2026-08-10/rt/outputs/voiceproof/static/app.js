let authenticated = document.body.dataset.authenticated === "true";
let modelReady = document.body.dataset.modelReady === "true";
const input = document.querySelector("#audio-input");
const form = document.querySelector("#analysis-form");
const dropzone = document.querySelector("#dropzone");
const fileRow = document.querySelector("#file-row");
const analyzeButton = document.querySelector("#analyze-button");

const percent = (number) => `${Math.round(number * 100)}%`;
const formatBytes = (bytes) => bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;

function setFile(file) {
  if (!file || file.size > 25 * 1024 * 1024) {
    if (file) alert("Please choose an audio file smaller than 25 MB.");
    return;
  }
  const container = new DataTransfer();
  container.items.add(file);
  input.files = container.files;
  document.querySelector("#file-name").textContent = file.name;
  document.querySelector("#file-size").textContent = formatBytes(file.size);
  fileRow.hidden = false;
  analyzeButton.disabled = !modelReady;
  dropzone.classList.add("has-file");
}

function renderSession(session) {
  authenticated = Boolean(session.user);
  modelReady = Boolean(session.model_ready);
  document.body.dataset.authenticated = authenticated;
  document.body.dataset.modelReady = modelReady;
  document.querySelector("#google-login").hidden = authenticated;
  document.querySelector("#account").hidden = !authenticated;
  document.querySelector("#auth-gate").hidden = authenticated;
  form.hidden = !authenticated;
  document.querySelector("#history-section").hidden = !authenticated;
  document.querySelector("#setup-notice").hidden = modelReady;
  input.disabled = !modelReady;
  if (session.user) {
    document.querySelector("#account-name").textContent = session.user.name || session.user.email;
    const picture = document.querySelector("#account-picture");
    picture.hidden = !session.user.picture;
    if (session.user.picture) picture.src = session.user.picture;
  }
  if (authenticated) loadHistory();
}

async function syncSession() {
  try {
    const response = await fetch("/api/session");
    if (!response.ok) throw new Error("Not served by VoiceProof yet");
    renderSession(await response.json());
  } catch {
    // A direct file preview still renders the completed interface without raw template tags.
    renderSession({ user: null, model_ready: false });
  }
}

if (input) {
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((event) => dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((event) => dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
  document.querySelector("#remove-file").addEventListener("click", () => { input.value = ""; fileRow.hidden = true; analyzeButton.disabled = true; dropzone.classList.remove("has-file"); });
}

function resultMarkup(result) {
  const synthetic = result.verdict === "synthetic";
  const source = result.source
    ? `<div class="attribution"><span class="source-label">LIKELY SOURCE</span><strong>${escapeHtml(result.source)}</strong><span>${percent(result.source_confidence)} attribution confidence</span></div>`
    : synthetic
      ? `<div class="attribution uncertain"><span class="source-label">SOURCE ATTRIBUTION</span><strong>Inconclusive</strong><span>Confidence did not clear the calibrated threshold.</span></div>`
      : `<div class="attribution uncertain"><span class="source-label">SOURCE ATTRIBUTION</span><strong>Not assessed</strong><span>Attribution is only provided for synthetic verdicts.</span></div>`;
  const alternatives = result.alternatives?.length ? `<div class="alternatives"><span>MODEL DISTRIBUTION</span>${result.alternatives.map((item) => `<div><i style="width:${item.probability * 100}%"></i><b>${escapeHtml(item.name)}</b><em>${percent(item.probability)}</em></div>`).join("")}</div>` : "";
  return `<div class="verdict ${synthetic ? "synthetic" : "human"}"><span class="verdict-orb"></span><div><span class="source-label">AUTHENTICITY SIGNAL</span><h3>${synthetic ? "Synthetic audio detected" : "Likely human audio"}</h3><p>${synthetic ? "The model found patterns consistent with generated or converted speech." : "The model did not find enough synthetic-speech signal at the selected threshold."}</p></div><div class="score"><strong>${percent(result.fake_probability)}</strong><span>synthetic signal</span></div></div>${source}${alternatives}<p class="result-notice">${escapeHtml(result.notice)}</p>`;
}

function escapeHtml(value = "") { const holder = document.createElement("div"); holder.textContent = value; return holder.innerHTML; }

function showResult(result) {
  document.querySelector("#result-filename").textContent = result.filename;
  document.querySelector("#result-card").innerHTML = resultMarkup(result);
  document.querySelector("#result-section").hidden = false;
  document.querySelector("#result-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

if (form) form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!input.files[0]) return;
  analyzeButton.disabled = true; analyzeButton.innerHTML = '<span class="spinner"></span> Reading forensic signal…';
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed.");
    showResult(payload); loadHistory();
  } catch (error) { alert(error.message); }
  finally { analyzeButton.disabled = false; analyzeButton.innerHTML = 'Analyze audio <span>→</span>'; }
});

async function loadHistory() {
  const list = document.querySelector("#history-list");
  if (!authenticated || !list) return;
  try {
    const response = await fetch("/api/history");
    const records = await response.json();
    list.innerHTML = records.length ? records.map((record) => `<button class="history-row" data-record="${encodeURIComponent(JSON.stringify(record))}"><span class="history-file">${escapeHtml(record.filename)}</span><span class="history-verdict ${record.verdict}">${record.verdict === "synthetic" ? "Synthetic signal" : "Likely human"}</span><span>${new Date(record.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span><span>→</span></button>`).join("") : '<p class="empty-history">No analyses yet. Your first result will appear here.</p>';
    list.querySelectorAll(".history-row").forEach((row) => row.addEventListener("click", () => showResult(JSON.parse(decodeURIComponent(row.dataset.record)))));
  } catch { list.innerHTML = '<p class="empty-history">Could not load your history.</p>'; }
}
document.querySelector("#close-result")?.addEventListener("click", () => document.querySelector("#result-section").hidden = true);
syncSession();
