// Wires chessboard.js into the position Python sends via `data`, and reports
// drops back to Python as a transient trigger value.
//
// Optimistic UI, no client-side legality check: onDrop always lets the piece
// visually land where it was dropped (no snapback here), reports {from, to}
// via setTriggerValue, and leaves validation entirely to python-chess on the
// next rerun -- see chat.py's _render_board_panel docstring for why. If the
// move turns out illegal, Python leaves the board unchanged, and the next
// render's `data` (the still-unmoved FEN) snaps the piece back visually via
// chessboard.js's own position(fen, useAnimation) diffing.
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const { fen, size, draggable = true } = data;

  // chessboard.js's Chessboard() constructor calls .html(...) on its
  // container to build the board markup, which replaces the container's
  // *entire* innerHTML -- including the <style> tag Streamlit already
  // injected as a sibling inside parentElement for the css= content passed
  // to st.components.v2.component. Giving chessboard.js its own empty
  // child div (instead of parentElement itself) keeps that destructive
  // .html() call scoped to a div with nothing else in it.
  const boardContainer = document.createElement("div");
  // chessboard.js sizes itself to its container's current width (it does
  // not take a size config option) -- without an explicit width it stretch
  // to fill the column, growing taller than the fixed pixel height Python
  // requested for the component's outer frame, clipping the bottom ranks.
  // `size` here is round-tripped from the same value chat.py's
  // _render_board_panel passes as chess_board(..., size=...), so the two
  // stay in sync automatically instead of needing the same number hardcoded
  // in two places.
  boardContainer.style.width = `${size}px`;
  parentElement.appendChild(boardContainer);

  // On touch-first devices a drag starting on the board scrolls the page
  // instead of moving the piece, so those get tap-to-move: tap a piece,
  // then tap its destination. Desktop keeps dragging.
  const tapToMove = draggable && window.matchMedia("(pointer: coarse)").matches;

  const board = window.Chessboard(boardContainer, {
    position: fen,
    draggable: draggable && !tapToMove,
    pieceTheme: (piece) => CHESS_RAG_PIECE_IMAGES[piece],
    onDrop: (source, target) => {
      if (source === target) return;
      setTriggerValue("drop", { from: source, to: target });
    },
  });

  if (!tapToMove) return () => board.destroy();

  // chessboard.js's own highlight for a dragged piece's square, reused so
  // a tapped selection looks the same as a drag in progress.
  const SELECTED_CLASS = "highlight1-32417";
  const sideToMove = fen.split(" ")[1]; // "w" or "b"
  let selected = null;

  const squareElement = (square) => boardContainer.querySelector(`[data-square="${square}"]`);
  const select = (square) => {
    if (selected) squareElement(selected)?.classList.remove(SELECTED_CLASS);
    selected = square;
    if (square) squareElement(square)?.classList.add(SELECTED_CLASS);
  };

  // Tapping the selected piece deselects it, tapping another piece of the
  // side to move switches to it, and any other tap with a piece selected is
  // sent to Python as a move, the same {from, to} a drag produces. Legality
  // stays in python-chess: an unreachable square comes back as an illegal
  // move, and the rerun remounts the board with nothing selected.
  const onTap = (event) => {
    const square = event.target.closest("[data-square]")?.dataset.square;
    if (!square) return;
    const piece = board.position()[square]; // e.g. "wP", or undefined if empty
    if (square === selected) {
      select(null);
    } else if (piece && piece[0] === sideToMove) {
      select(square);
    } else if (selected) {
      const from = selected;
      select(null);
      setTriggerValue("drop", { from, to: square });
    }
  };
  boardContainer.addEventListener("click", onTap);

  return () => {
    boardContainer.removeEventListener("click", onTap);
    board.destroy();
  };
}
