// The "How this works" tour: spotlights one page element at a time with a
// tooltip. Runs entirely in the browser, so stepping through it never
// reruns the app.
export default function (component) {
  const { data } = component;
  const { launchId } = data;

  // launchId 0: the button hasn't been clicked yet.
  if (!launchId) return undefined;

  // Phones (the layout's 640px breakpoint) get the steps in page order,
  // their own wording where a step has one, and the tooltip as a bottom
  // sheet, since the board and chat are too tall for a tooltip beside them.
  const isPhone = window.matchMedia("(max-width: 640px)").matches;
  const steps = isPhone ? data.phoneStepOrder.map((i) => data.steps[i]) : data.steps;

  let stepIndex = 0;
  let target = null;
  let revealed = false;

  const backdrop = document.createElement("div");
  backdrop.className = "tour-backdrop";

  const spotlight = document.createElement("div");
  spotlight.className = "tour-spotlight";

  const tooltip = document.createElement("div");
  tooltip.className = isPhone ? "tour-tooltip tour-sheet" : "tour-tooltip";
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

  // Added to document.body: a position: fixed element is positioned against
  // any ancestor with a transform, which Streamlit's own markup might have.
  // isolate_styles=False in __init__.py lets this CSS reach them there.
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
    // Measured while hidden: visibility: hidden still lays the tooltip out,
    // so this is its real size for this step's text.
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
    // The bottom sheet is placed by CSS; only the floating tooltip moves.
    if (!isPhone) positionTooltip(rect);
    reveal();
  }

  // The chat and its input are last on a phone page, with nothing below to
  // scroll into, so they'd stay behind the sheet. A spacer the sheet's
  // height is added while the tour is open.
  const pageEnd = document.querySelector('[data-testid="stMainBlockContainer"]');
  const spacer = document.createElement("div");
  spacer.className = "tour-sheet-spacer";
  if (isPhone && pageEnd) pageEnd.appendChild(spacer);

  // Phones: scroll the target into the space between Streamlit's header and
  // the sheet, centred if it fits. Streamlit scrolls its own main
  // container, not the window.
  function scrollAboveSheet() {
    spacer.style.height = `${tooltip.getBoundingClientRect().height}px`;
    const header = document.querySelector('[data-testid="stHeader"]');
    const areaTop = (header ? header.getBoundingClientRect().bottom : 0) + 8;
    const areaBottom = window.innerHeight - tooltip.getBoundingClientRect().height - 8;
    const rect = target.getBoundingClientRect();
    const spare = areaBottom - areaTop - rect.height;
    const desiredTop = spare > 0 ? areaTop + spare / 2 : areaTop;
    const scroller = target.closest('[data-testid="stMain"]') || document.scrollingElement;
    scroller.scrollBy({ top: rect.top - desiredTop, behavior: "smooth" });
  }

  // The nearest step from `start`, moving by `direction` (+1 or -1), whose
  // element is on the page, or -1. A step can be missing, and searching in
  // the direction of travel means skipping it never leaves Back and Next
  // stuck on the same step.
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
      // Past the last step going forward, the tour ends; going back past
      // the first, it stays put.
      if (direction > 0) close();
      return;
    }
    target = document.querySelector(steps[i].selector);
    stepIndex = i;
    const step = steps[stepIndex];
    stepLabelEl.textContent = `Step ${stepIndex + 1} of ${steps.length}`;
    titleEl.textContent = step.title;
    textEl.textContent = (isPhone && step.phoneText) || step.text;
    // Disabled on the first step that exists, which isn't always index 0.
    backBtn.disabled = findStep(stepIndex - 1, -1) === -1;
    nextBtn.textContent = stepIndex === steps.length - 1 ? "Done" : "Next";
    if (isPhone) scrollAboveSheet();
    else target.scrollIntoView({ block: "center", behavior: "smooth" });
    // Two animation frames: one for the scroll, one for layout to settle,
    // before measuring the target.
    requestAnimationFrame(() => requestAnimationFrame(reposition));
  }

  function close() {
    spacer.remove();
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
  // capture: true, since scroll events from inner containers don't bubble
  // to window.
  window.addEventListener("resize", reposition, { passive: true });
  window.addEventListener("scroll", reposition, { capture: true, passive: true });
  window.addEventListener("keydown", onKeydown);

  showStep(0);

  return close;
}
