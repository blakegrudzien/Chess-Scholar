// Draws the board with chessboard.js and plays moves instantly in the
// browser, using chess.js (concatenated ahead of this file, see
// __init__.py) to know the rules. python-chess on the server still has the
// final say: every legal move is sent to Python, which checks it, records
// it, and sends its own FEN back.
//
// Moves made here are never redrawn by Python. The component is keyed, so
// Streamlit keeps this element alive across reruns and calls this function
// again whenever `data` changes, without running the cleanup in between.
// Each call compares Python's FEN with what the board already shows and
// redraws only when they differ: a change made on the server side (Reset,
// Undo, replay, a position from the chat) or a move python-chess rejected.

// chessboard.js's own highlight for a dragged piece's square, reused so a
// tapped selection looks the same as a drag in progress.
const SELECTED_CLASS = "highlight1-32417";

// Placement, side to move and castling rights. The en passant field is left
// out because python-chess and chess.js disagree on when to write it, which
// would make identical positions compare as different.
const samePosition = (a, b) => a.split(" ").slice(0, 3).join(" ") === b.split(" ").slice(0, 3).join(" ");

function createView(parentElement, { fen, size }) {
  // chessboard.js's Chessboard() constructor calls .html(...) on its
  // container to build the board markup, which replaces the container's
  // entire innerHTML, including the style tag Streamlit injects into
  // parentElement for this component's CSS. Its own empty child div keeps
  // that call contained. chessboard.js also sizes itself to its container's
  // width, so the width is set explicitly from the size Python passes.
  const container = document.createElement("div");
  container.style.width = `${size}px`;
  parentElement.appendChild(container);

  // On touch-first devices a drag starting on the board scrolls the page
  // instead of moving the piece, so those get tap-to-move: tap a piece, then
  // tap its destination. Desktop keeps dragging.
  const tapToMove = window.matchMedia("(pointer: coarse)").matches;

  const view = { size, movable: false, setTriggerValue: () => {} };
  let game = null;
  // Moves played here that Python hasn't confirmed yet, oldest first. Only
  // the oldest has been sent: Streamlit can fold two quick trigger values
  // into one rerun, so each move waits for the previous one's confirmation.
  const pending = [];
  let selected = null;

  const loadGame = (newFen) => {
    try {
      game = new Chess(newFen);
    } catch (error) {
      // A position python-chess accepts but chess.js won't (an unusual
      // position from the chat, for example). Moves then go straight to
      // Python, which decides alone, as it did before chess.js was added.
      console.warn("chess.js could not load this position; moves will be checked by the server only.", error);
      game = null;
    }
  };
  loadGame(fen);

  const sendOldestPending = () => {
    if (!pending.length) return;
    const { from, to } = pending[0];
    // Deferred a tick: this often runs inside Streamlit's call that
    // delivers Python's confirmation, and a trigger value set during that
    // call never reaches Python.
    setTimeout(() => view.setTriggerValue("drop", { from, to }), 0);
  };

  // Plays a move locally if legal (auto-queen, matching the server) and
  // queues it for Python. Returns false for an illegal move, which is
  // rejected here and never sent.
  const play = (from, to) => {
    if (!game) {
      view.setTriggerValue("drop", { from, to });
      return true;
    }
    try {
      game.move({ from, to, promotion: "q" });
    } catch {
      return false;
    }
    pending.push({ from, to, fenAfter: game.fen() });
    if (pending.length === 1) sendOldestPending();
    return true;
  };

  const board = window.Chessboard(container, {
    position: fen,
    draggable: !tapToMove,
    pieceTheme: (piece) => CHESS_RAG_PIECE_IMAGES[piece],
    onDragStart: (source, piece) => view.movable && (!game || piece[0] === game.turn()),
    onDrop: (source, target) => {
      if (source === target) return "snapback";
      return play(source, target) ? undefined : "snapback";
    },
    // Castling, en passant and promotion change more than the dragged piece;
    // redraw from chess.js's resulting position once the drop settles.
    onSnapEnd: () => {
      if (game) board.position(game.fen(), false);
    },
  });

  const squareElement = (square) => container.querySelector(`[data-square="${square}"]`);
  const select = (square) => {
    if (selected) squareElement(selected)?.classList.remove(SELECTED_CLASS);
    selected = square;
    if (square) squareElement(square)?.classList.add(SELECTED_CLASS);
  };

  // Tapping the selected piece deselects it, tapping another piece of the
  // side to move switches to it, and any other tap with a piece selected is
  // a move: played at once if legal, and the selection cleared either way.
  const onTap = (event) => {
    if (!tapToMove || !view.movable) return;
    const square = event.target.closest("[data-square]")?.dataset.square;
    if (!square) return;
    const piece = board.position()[square]; // e.g. "wP", or undefined if empty
    const sideToMove = game ? game.turn() : null;
    if (square === selected) {
      select(null);
    } else if (piece && (!game || piece[0] === sideToMove)) {
      select(square);
    } else if (selected) {
      const from = selected;
      select(null);
      if (play(from, square) && game) board.position(game.fen());
    }
  };
  container.addEventListener("click", onTap);

  view.sync = (newFen) => {
    if (pending.length && samePosition(newFen, pending[0].fenAfter)) {
      // Python confirmed the oldest move; the board already shows it.
      pending.shift();
      sendOldestPending();
      return;
    }
    if (!pending.length && game && samePosition(newFen, game.fen())) return;
    // Python's position differs from the board's: a server-side change, or
    // a move python-chess rejected. Python is the authority, so redraw.
    pending.length = 0;
    select(null);
    loadGame(newFen);
    board.position(newFen);
  };

  view.destroy = () => {
    container.removeEventListener("click", onTap);
    board.destroy();
    container.remove();
  };
  return view;
}

export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  let view = parentElement.__chessRagBoard;
  const rebuild = !view || view.size !== data.size;
  if (rebuild) {
    view?.destroy();
    view = createView(parentElement, data);
    parentElement.__chessRagBoard = view;
  }
  // Refreshed on every call, before sync() can send the next queued move:
  // a previous call's setTriggerValue is not guaranteed to still reach
  // Python. `movable` lets replay and answer generation turn moving off
  // without rebuilding the board.
  view.setTriggerValue = setTriggerValue;
  view.movable = data.draggable !== false;
  if (!rebuild) view.sync(data.fen);

  return () => {
    view.destroy();
    delete parentElement.__chessRagBoard;
  };
}
