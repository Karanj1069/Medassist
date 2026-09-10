// ---- Symptom search/filter ----
const searchBox = document.getElementById("symptomSearch");
const chips = Array.from(document.querySelectorAll(".chip"));

searchBox.addEventListener("input", () => {
  const q = searchBox.value.toLowerCase();
  chips.forEach(chip => {
    const label = chip.textContent.toLowerCase();
    chip.style.display = label.includes(q) ? "inline-flex" : "none";
  });
});

// ---- Track selected symptoms ----
const selectedListEl = document.getElementById("selectedList");
const checkboxes = Array.from(document.querySelectorAll(".symptom-checkbox"));

function getSelectedSymptoms() {
  return checkboxes.filter(cb => cb.checked).map(cb => cb.value);
}

function updateSelectedDisplay() {
  const selected = getSelectedSymptoms();
  selectedListEl.textContent = selected.length
    ? selected.map(s => s.replace(/_/g, " ")).join(", ")
    : "none yet";
}

checkboxes.forEach(cb => cb.addEventListener("change", updateSelectedDisplay));

// ---- Symptom check -> /api/predict ----
const checkBtn = document.getElementById("checkBtn");
const checkError = document.getElementById("checkError");
const resultCard = document.getElementById("resultCard");
const aiSummaryBox = document.getElementById("aiSummaryBox");

let lastPredictedDisease = null;

checkBtn.addEventListener("click", async () => {
  checkError.textContent = "";
  const symptoms = getSelectedSymptoms();
  if (symptoms.length === 0) {
    checkError.textContent = "Please select at least one symptom.";
    return;
  }

  checkBtn.disabled = true;
  checkBtn.textContent = "Checking…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms, with_ai: true }),
    });
    const data = await res.json();

    if (!res.ok) {
      checkError.textContent = data.error || "Something went wrong.";
      return;
    }

    lastPredictedDisease = data.disease;
    renderResult(data);
  } catch (err) {
    checkError.textContent = "Could not reach the server.";
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check my symptoms";
  }
});

function fillList(elId, items) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  if (!items || items.length === 0) {
    el.innerHTML = "<li>None listed</li>";
    return;
  }
  items.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function renderResult(data) {
  document.getElementById("resultDisease").textContent = data.disease;
  document.getElementById("resultDesc").textContent = data.description;

  fillList("resultPrecautions", data.precautions);
  fillList("resultMedications", data.medications);
  fillList("resultDiet", data.diet);
  fillList("resultWorkout", data.workout);

  if (data.ai_summary) {
    document.getElementById("aiSummaryText").textContent = data.ai_summary;
    aiSummaryBox.classList.remove("hidden");
  } else {
    aiSummaryBox.classList.add("hidden");
  }

  renderCandidates(data.candidates);

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderCandidates(candidates) {
  const box = document.getElementById("candidatesBox");
  const list = document.getElementById("candidatesList");
  list.innerHTML = "";

  // Skip the first one - it's already the headline result above.
  const rest = (candidates || []).slice(1);
  if (rest.length === 0) {
    box.classList.add("hidden");
    return;
  }

  rest.forEach(c => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${c.disease}</span> — matched ${c.matched_count}/${c.disease_symptom_count} known symptoms (overlap score ${c.score})`;
    list.appendChild(li);
  });
  box.classList.remove("hidden");
}

// ---- Chat popup ----
const chatToggle = document.getElementById("chatToggle");
const chatPanel = document.getElementById("chatPanel");
const chatClose = document.getElementById("chatClose");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const chatSources = document.getElementById("chatSources");

let chatHistory = []; // {role, content}[]

chatToggle.addEventListener("click", () => {
  chatPanel.classList.toggle("hidden");
});
chatClose.addEventListener("click", () => chatPanel.classList.add("hidden"));

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  chatHistory.push({ role: "user", content: text });
  chatInput.value = "";
  chatSources.textContent = "";

  const thinkingEl = document.createElement("div");
  thinkingEl.className = "msg bot";
  thinkingEl.textContent = "…";
  chatMessages.appendChild(thinkingEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: chatHistory.slice(0, -1), // send prior turns, current one is `message`
        predicted_disease: lastPredictedDisease,
      }),
    });
    const data = await res.json();
    thinkingEl.remove();

    const reply = data.reply || data.error || "Something went wrong.";
    addMessage("bot", reply);
    chatHistory.push({ role: "assistant", content: reply });

    if (data.sources && data.sources.length) {
      chatSources.textContent = "Grounded in: " + data.sources.join(", ");
    }
  } catch (err) {
    thinkingEl.remove();
    addMessage("bot", "Could not reach the server.");
  }
});
