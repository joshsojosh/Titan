import shutil
import time

import chess
import chess.engine
import chess.polyglot

SEARCH_TIME_SECONDS = 10.0
ANALYSIS_TIME_SECONDS = 1.0
MATE_SCORE = 100_000
INFINITY = 1_000_000
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class ChessBot:
    def __init__(self, search_time: float = SEARCH_TIME_SECONDS):
        self.search_time = search_time
        self.deadline = 0.0
        self.transposition_table = {}
        self.killer_moves = {}
        self.history_scores = {}

    def find_best_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("Cannot choose a move in a game-over position.")

        self.deadline = time.monotonic() + self.search_time
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.history_scores.clear()
        best_move = legal_moves[0]
        previous_score = None

        for depth in range(1, 65):
            window = 50
            alpha = -INFINITY if previous_score is None else max(
                -INFINITY, previous_score - window
            )
            beta = INFINITY if previous_score is None else min(
                INFINITY, previous_score + window
            )

            while True:
                try:
                    score = self._search(board, depth, alpha, beta, 0)
                except TimeoutError:
                    return best_move

                if score <= alpha and alpha > -INFINITY:
                    alpha = max(-INFINITY, alpha - window)
                elif score >= beta and beta < INFINITY:
                    beta = min(INFINITY, beta + window)
                else:
                    break
                window *= 2

            entry = self.transposition_table.get(self._position_key(board))
            if entry is not None and entry[3] in legal_moves:
                best_move = entry[3]
            previous_score = score

        return best_move

    def _position_key(self, board: chess.Board):
        return chess.polyglot.zobrist_hash(board), board.halfmove_clock

    def _search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        if time.monotonic() >= self.deadline:
            raise TimeoutError
        if board.is_game_over(claim_draw=True):
            return -MATE_SCORE + ply if board.is_checkmate() else 0
        if depth == 0:
            return self._quiescence(board, alpha, beta, ply, 0)

        key = self._position_key(board)
        original_alpha = alpha
        cached = self.transposition_table.get(key)
        cached_move = cached[3] if cached is not None else None
        if cached is not None and cached[0] >= depth:
            score, _, bound, _ = cached
            if bound == "exact":
                return score
            if bound == "lower":
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        best_score = -INFINITY
        best_move = None
        moves = sorted(
            board.legal_moves,
            key=lambda move: self._move_order_score(board, move, cached_move, ply),
            reverse=True,
        )
        for move_index, move in enumerate(moves):
            board.push(move)
            try:
                if move_index == 0:
                    score = -self._search(
                        board, depth - 1, -beta, -alpha, ply + 1
                    )
                else:
                    score = -self._search(
                        board, depth - 1, -alpha - 1, -alpha, ply + 1
                    )
                    if alpha < score < beta:
                        score = -self._search(
                            board, depth - 1, -beta, -alpha, ply + 1
                        )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.is_capture(move) and move.promotion is None:
                    self._record_quiet_cutoff(board, move, ply, depth)
                break

        bound = "upper" if best_score <= original_alpha else "lower" if best_score >= beta else "exact"
        self.transposition_table[key] = (depth, best_score, bound, best_move)
        return best_score

    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int, qdepth: int
    ) -> int:
        if time.monotonic() >= self.deadline:
            raise TimeoutError
        if board.is_game_over(claim_draw=True):
            return -MATE_SCORE + ply if board.is_checkmate() else 0

        in_check = board.is_check()
        stand_pat = self._evaluate(board)
        if not in_check:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        if qdepth >= 8:
            return stand_pat

        moves = list(board.legal_moves)
        if not in_check:
            moves = [
                move for move in moves
                if board.is_capture(move) or move.promotion is not None
            ]
        moves.sort(
            key=lambda move: self._move_order_score(board, move, None, ply),
            reverse=True,
        )

        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _move_order_score(
        self,
        board: chess.Board,
        move: chess.Move,
        preferred_move: chess.Move | None,
        ply: int,
    ) -> int:
        if move == preferred_move:
            return 2_000_000

        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                victim_value = PIECE_VALUES[chess.PAWN]
            else:
                victim_value = PIECE_VALUES[victim.piece_type] if victim else 0
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
            score += 1_000_000 + 10 * victim_value - attacker_value
        if move.promotion is not None:
            score += 850_000 + PIECE_VALUES[move.promotion]
        if score:
            return score

        killers = self.killer_moves.get(ply, ())
        if move in killers:
            return 700_000 - 100_000 * killers.index(move)

        history_key = (board.turn, move.from_square, move.to_square)
        history_score = self.history_scores.get(history_key, 0)
        if board.gives_check(move):
            return 400_000 + history_score
        return history_score

    def _record_quiet_cutoff(
        self, board: chess.Board, move: chess.Move, ply: int, depth: int
    ) -> None:
        killers = self.killer_moves.setdefault(ply, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]

        history_key = (board.turn, move.from_square, move.to_square)
        self.history_scores[history_key] = min(
            300_000,
            self.history_scores.get(history_key, 0) + depth * depth,
        )

    @staticmethod
    def _evaluate(board: chess.Board) -> int:
        score = 0
        non_pawn_material = sum(
            PIECE_VALUES[piece.piece_type]
            for piece in board.piece_map().values()
            if piece.piece_type not in (chess.PAWN, chess.KING)
        )

        for square, piece in board.piece_map().items():
            file_index = chess.square_file(square)
            rank = chess.square_rank(square)
            relative_rank = rank if piece.color == chess.WHITE else 7 - rank
            center = (
                7 - abs(2 * file_index - 7) + 7 - abs(2 * rank - 7)
            ) / 2
            value = PIECE_VALUES[piece.piece_type]

            if piece.piece_type == chess.PAWN:
                value += 6 * relative_rank + 2 * center
            elif piece.piece_type == chess.KNIGHT:
                value += 7 * center
            elif piece.piece_type == chess.BISHOP:
                value += 4 * center
            elif piece.piece_type == chess.ROOK and relative_rank == 6:
                value += 15
            elif piece.piece_type == chess.QUEEN:
                value += 2 * center
            elif piece.piece_type == chess.KING:
                value += 8 * center if non_pawn_material < 2_600 else -2 * center

            score += value if piece.color == chess.WHITE else -value

        for color in (chess.WHITE, chess.BLACK):
            if len(board.pieces(chess.BISHOP, color)) >= 2:
                score += 30 if color == chess.WHITE else -30
        return score if board.turn == chess.WHITE else -score


def choose_human_color() -> chess.Color:
    while True:
        choice = input("Black or White? ").strip().lower()
        if choice in ("white", "w"):
            return chess.WHITE
        if choice in ("black", "b"):
            return chess.BLACK
        print("Please enter White or Black.")


def print_board(board: chess.Board) -> None:
    for rank in range(7, -1, -1):
        pieces = (
            board.piece_at(chess.square(file_index, rank))
            for file_index in range(8)
        )
        row = " ".join(piece.symbol() if piece else "." for piece in pieces)
        print(f"{rank + 1} {row}")
    print("  A B C D E F G H")


def analyze_position(board: chess.Board) -> str:
    executable = shutil.which("stockfish") or "/usr/games/stockfish"
    with chess.engine.SimpleEngine.popen_uci(executable) as engine:
        info = engine.analyse(
            board, chess.engine.Limit(time=ANALYSIS_TIME_SECONDS)
        )

    principal_variation = info.get("pv")
    if principal_variation:
        best_move = principal_variation[0]
        player = "White" if board.turn == chess.WHITE else "Black"
        best_move_text = (
            f"Best move for {player}: {board.san(best_move)} "
            f"({best_move.uci()})."
        )
    else:
        best_move_text = "Stockfish found no legal move."

    score = info["score"].white()
    if score.is_mate():
        mate_in = score.mate()
        if mate_in is None:
            evaluation = "Stockfish evaluates the position as a forced mate."
        else:
            winner = "White" if mate_in > 0 else "Black"
            moves = abs(mate_in)
            evaluation = f"Stockfish: {winner} can force mate in {moves}."
    else:
        centipawns = score.score()
        if centipawns is None:
            evaluation = "Stockfish could not evaluate this position."
        else:
            pawns = centipawns / 100
            if centipawns > 50:
                verdict = "White is winning"
            elif centipawns < -50:
                verdict = "Black is winning"
            else:
                verdict = "The position is roughly equal"
            evaluation = f"Stockfish: {verdict} ({pawns:+.2f})."

    return f"{evaluation}\n{best_move_text}"


def print_analysis(board: chess.Board) -> None:
    try:
        print(analyze_position(board))
    except FileNotFoundError:
        print("Stockfish was not found. Install it and add it to PATH to analyze positions.")
    except chess.engine.EngineError as error:
        print(f"Stockfish analysis failed: {error}")


def main() -> None:
    human_color = choose_human_color()
    bot_color = not human_color
    board = chess.Board()
    bot = ChessBot()
    print(f"You are {'White' if human_color else 'Black'}; the bot is {'Black' if bot_color else 'White'}.")
    game_result = None

    while not board.is_game_over():
        print_board(board)
        print_analysis(board)

        if board.turn == human_color:
            text = input("Your move (like e2e4, or Analyze): ")
            command = text.strip().lower()
            if command == "analyze":
                print_analysis(board)
                continue
            if command == "resign":
                print("You resigned.")
                game_result = "0-1" if human_color == chess.WHITE else "1-0"
                break

            try:
                move = chess.Move.from_uci(text)
            except ValueError:
                print("Try a move like e2e4.")
                continue

            if move not in board.legal_moves:
                print("That move isn't legal. Try again.")
                continue
        else:
            move = bot.find_best_move(board)
            print("Bot plays:", move)

        board.push(move)

    print_board(board)
    print("Game result:", game_result or board.result())


if __name__ == "__main__":
    main()
