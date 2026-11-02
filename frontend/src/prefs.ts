// Per-browser UI preferences. Storage can be unavailable (private mode, blocked
// site data), so every access is guarded and falls back to the default.
const AUTO_ANSWER_KEY = "askito.autoAnswer";

export function getAutoAnswer(): boolean {
  try {
    return window.localStorage.getItem(AUTO_ANSWER_KEY) === "1";
  } catch {
    return false;
  }
}

export function setAutoAnswer(on: boolean): void {
  try {
    window.localStorage.setItem(AUTO_ANSWER_KEY, on ? "1" : "0");
  } catch {
    /* preference not persisted */
  }
}
