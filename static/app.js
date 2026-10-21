const MAX_LENGTH = 1000;

const form = document.getElementById("check-form");
const textarea = document.getElementById("message");
const count = document.getElementById("count");
const submit = document.getElementById("submit");
const resultBox = document.getElementById("result");

function updateCount() {
  count.textContent = textarea.value.length;
  count.parentElement.classList.toggle("limit", textarea.value.length >= MAX_LENGTH);
}

function showResult(type, icon, html) {
  resultBox.innerHTML = `<div class="result ${type}"><span class="icon">${icon}</span><p>${html}</p></div>`;
}

async function checkMessage(event) {
  event.preventDefault();
  const text = textarea.value.trim();
  if (!text) return;

  submit.disabled = true;
  submit.textContent = "Checking...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();

    if (!response.ok) {
      showResult("error", "⚠️", data.error || "Something went wrong.");
    } else if (data.is_spam) {
      showResult("spam", "🚫", "This looks like <strong>spam</strong>.");
    } else {
      showResult("ham", "✅", "This looks like a <strong>normal message</strong>.");
    }
  } catch {
    showResult("error", "⚠️", "Could not reach the server. Please try again.");
  } finally {
    submit.disabled = false;
    submit.textContent = "Check";
  }
}

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textarea.value = chip.dataset.text;
    updateCount();
    textarea.focus();
  });
});

textarea.addEventListener("input", updateCount);
form.addEventListener("submit", checkMessage);
updateCount();
