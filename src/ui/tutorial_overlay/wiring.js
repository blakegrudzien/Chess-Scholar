// Spotlights one real UI element at a time with a short tooltip -- a quick
// guided tour launched from the "How this works" button in
// tutorial_overlay/__init__.py. Fully client-side: Next/Back/Skip never
// touch Streamlit, so stepping through the tour costs no rerun and no
// network round trip.
export default function (component) {
  const { data } = component;
  const { launchId, steps } = data;

  // launchId 0 means the button has never been clicked (see __init__.py's
  // docstring for why this is a counter, not a bool) -- stay dormant.
  if (!launchId) return undefined;

  let stepIndex = 0;
  let target = null;
  let revealed = false;

  const backdrop = document.createElement("div");
  backdrop.className = "tour-backdrop";

  const spotlight = document.createElement("div");
  spotlight.className = "tour-spotlight";

  const tooltip = document.createElement("div");
  tooltip.className = "tour-tooltip";
  tooltip.innerHTML = `
    <div class="tour-step-label"></div>
    <h4 class="tour-title"></h4>
    <p class="tour-text"></p>
    <div class="tour-controls">
      <button class="tour-btn tour-skip" type="button">Skip</button>
      <div class="tour-controls-right">
        <button class="tour-btn tour-back" type="button">Back</button>
        <button class="tour-btn tour-btn-primary tour-next" type="button">Next</button>
      </div>
    </div>
  `;

  const stepLabelEl = tooltip.querySelector(".tour-step-label");
  const titleEl = tooltip.querySelector(".tour-title");
  const textEl = tooltip.querySelector(".tour-text");
  const backBtn = tooltip.querySelector(".tour-back");
  const nextBtn = tooltip.querySelector(".tour-next");
  const skipBtn = tooltip.querySelector(".tour-skip");

  // Appended to document.body, not to `component.parentElement` -- a
  // position: fixed element's containing block silently becomes the
  // nearest ancestor with a transform/filter/contain property if one sits
  // between the mount point and <html>, which there's no cheap way to
  // audit for in Streamlit's own chrome. Anchoring to <body> directly
  // sidesteps that whole class of bug. isolate_styles is False in
  // __init__.py specifically so this file's CSS still reaches these three
  // elements despite them living outside parentElement's own subtree.
  document.body.appendChild(backdrop);
  document.body.appendChild(spotlight);
  document.body.appendChild(tooltip);

  function reveal() {
    if (revealed) return;
    revealed = true;
    backdrop.style.visibility = "visible";
    spotlight.style.visibility = "visible";
    tooltip.style.visibility = "visible";
  }

  function positionTooltip(rect) {
    const margin = 12;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Measured while still off-screen: visibility: hidden (unlike
    // display: none) keeps the element in layout, so this reflects the
    // tooltip's real rendered size for the step's actual text.
    const ttRect = tooltip.getBoundingClientRect();
    const spaceBelow = vh - rect.bottom - margin;
    const spaceAbove = rect.top - margin;
    let top =
      spaceBelow >= ttRect.height || spaceBelow >= spaceAbove
        ? rect.bottom + margin
        : rect.top - ttRect.height - margin;
    top = Math.max(8, Math.min(top, vh - ttRect.height - 8));
    const left = Math.max(8, Math.min(rect.left, vw - ttRect.width - 8));
    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
  }

  function reposition() {
    if (!target || !target.isConnected) return;
    const rect = target.getBoundingClientRect();
    const pad = 6;
    spotlight.style.top = `${rect.top - pad}px`;
    spotlight.style.left = `${rect.left - pad}px`;
    spotlight.style.width = `${rect.width + pad * 2}px`;
    spotlight.style.height = `${rect.height + pad * 2}px`;
    positionTooltip(rect);
    reveal();
  }

  // Finds the nearest step at or past `start` (moving by `direction`, +1 or
  // -1) whose target actually exists in the DOM right now, or -1 if none
  // does. A step's element can be legitimately absent -- "Find related
  // resources" only renders after the first chat exchange -- so a tour
  // opened before that must skip it, not crash or spotlight an empty rect.
  //
  // Direction matters, not just "does this index resolve": Next always
  // searches forward and Back always searches backward, so skipping a
  // missing step never leaves a dead end where Back and Next both land on
  // the same resolved step (confirmed live -- an earlier version of this
  // function always searched forward regardless of which button was
  // pressed, which is exactly what produced that stuck state).
  function findStep(start, direction) {
    let i = start;
    while (i >= 0 && i < steps.length) {
      if (document.querySelector(steps[i].selector)) return i;
      console.warn(`Tutorial: no element for selector "${steps[i].selector}", skipping.`);
      i += direction;
    }
    return -1;
  }

  function showStep(index, direction = 1) {
    const i = findStep(index, direction);
    if (i === -1) {
      // Ran off the end going forward (or "Done" was clicked): nothing left
      // to show. Ran off the start going backward: there's truly nothing
      // before the current step, so just stay put rather than closing --
      // only Next reaching past the last step should end the tour.
      if (direction > 0) close();
      return;
    }
    target = document.querySelector(steps[i].selector);
    stepIndex = i;
    const step = steps[stepIndex];
    stepLabelEl.textContent = `Step ${stepIndex + 1} of ${steps.length}`;
    titleEl.textContent = step.title;
    textEl.textContent = step.text;
    // Not just stepIndex === 0 -- the first *resolvable* step (e.g. Step 2
    // if Step 1's element isn't in the DOM yet) has nowhere valid to go
    // back to either, even though its own index isn't 0.
    backBtn.disabled = findStep(stepIndex - 1, -1) === -1;
    nextBtn.textContent = stepIndex === steps.length - 1 ? "Done" : "Next";
    target.scrollIntoView({ block: "center", behavior: "smooth" });
    // Two rAFs: one to let the browser commit the scroll, one to let
    // layout settle after it, before measuring the target's post-scroll
    // position.
    requestAnimationFrame(() => requestAnimationFrame(reposition));
  }

  function close() {
    backdrop.remove();
    spotlight.remove();
    tooltip.remove();
    window.removeEventListener("resize", reposition);
    window.removeEventListener("scroll", reposition, true);
    window.removeEventListener("keydown", onKeydown);
  }

  function onKeydown(event) {
    if (event.key === "Escape") close();
    else if (event.key === "ArrowRight") nextBtn.click();
    else if (event.key === "ArrowLeft" && !backBtn.disabled) backBtn.click();
  }

  backBtn.addEventListener("click", () => showStep(stepIndex - 1, -1));
  nextBtn.addEventListener("click", () => {
    if (stepIndex === steps.length - 1) close();
    else showStep(stepIndex + 1, 1);
  });
  skipBtn.addEventListener("click", close);
  backdrop.addEventListener("click", close);
  // capture: true -- scroll events don't bubble to window from most
  // scrollable ancestors, so this has to be registered on the capture
  // phase to see them at all.
  window.addEventListener("resize", reposition, { passive: true });
  window.addEventListener("scroll", reposition, { capture: true, passive: true });
  window.addEventListener("keydown", onKeydown);

  showStep(0);

  return close;
}
