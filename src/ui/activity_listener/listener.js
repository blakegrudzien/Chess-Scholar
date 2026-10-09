// Reports the first keystroke in any text box on the page while Python has
// marked the session inactive, so typing a question counts as activity
// before it's submitted. Streamlit's own inputs only reach Python on submit.
//
// Listens on the document rather than on specific widgets, so the chat
// input and the board's "Ask about this position" box are both covered
// without touching either one. Capture phase, so it sees the keystroke
// even if a widget stops it from bubbling.
export default function (component) {
  const { data, setTriggerValue } = component;
  if (!data.armed) return;

  const onKeydown = (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) return;
    // One report per armed period: Python disarms the listener on the
    // rerun this triggers, which remounts the component.
    document.removeEventListener("keydown", onKeydown, true);
    setTriggerValue("keystroke", true);
  };

  document.addEventListener("keydown", onKeydown, true);
  return () => document.removeEventListener("keydown", onKeydown, true);
}
