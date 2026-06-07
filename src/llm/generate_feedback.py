"""Coaching feedback generation.

Takes structured game state (from
``src.integration.fusion.game_state.GameStateBuilder``) and produces natural
language coaching feedback.

Three interchangeable backends:

* ``rule``   — explainable, dependency-free heuristics. Default, always works.
* ``openai`` — hosted LLM via the ``openai`` SDK (lazy-imported).
* ``hf``     — local HuggingFace ``transformers`` text-generation (lazy-imported).

Heavy deps are imported lazily inside the backend so this module imports cleanly
without the ``[llm]`` extras installed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .prompt_templates import PromptTemplates


class CoachingFeedbackGenerator:
    """Generate coaching feedback from game state or scene captions."""

    def __init__(self, backend: str = "rule", model: Optional[str] = None):
        """
        Args:
            backend: "rule" | "openai" | "hf".
            model: Backend model id. Defaults: openai→gpt-4o-mini, hf→distilgpt2.
        """
        self.backend = backend
        self.model = model
        self.prompts = PromptTemplates()
        self._client = None  # lazy backend handle

    # -- public API ---------------------------------------------------------

    def from_caption(self, caption: str) -> str:
        """Coaching advice for a single scene caption."""
        prompt = self.prompts.get("clip_interpretation", caption=caption)
        return self._run(prompt, caption=caption)

    def from_game_state(self, state: Dict[str, Any]) -> str:
        """Coaching advice for one structured game-state dict."""
        caption = self._describe(state)
        return self.from_caption(caption)

    def summarize(self, states: List[Dict[str, Any]]) -> str:
        """High-level summary across a sequence of game states."""
        captions = "\n".join(f"- {self._describe(s)}" for s in states)
        prompt = self.prompts.get("game_summary", captions=captions)
        return self._run(prompt, caption=captions)

    # -- backends -----------------------------------------------------------

    def _run(self, prompt: str, caption: str = "") -> str:
        if self.backend == "rule":
            return self._rule_based(caption or prompt)
        if self.backend == "openai":
            return self._openai(prompt)
        if self.backend == "hf":
            return self._hf(prompt)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def _openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - needs [llm] extra
            raise RuntimeError("openai backend needs `pip install openai`") from e
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = self._client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def _hf(self, prompt: str) -> str:
        try:
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover - needs [llm] extra
            raise RuntimeError("hf backend needs `pip install transformers`") from e
        if self._client is None:
            self._client = pipeline("text-generation", model=self.model or "distilgpt2")
        out = self._client(prompt, max_new_tokens=80, num_return_sequences=1)
        generated = out[0]["generated_text"]
        return generated[len(prompt):].strip() or generated.strip()

    # -- rule-based (no deps) ----------------------------------------------

    @staticmethod
    def _rule_based(text: str) -> str:
        """Deterministic, explainable feedback from action keywords.

        Placeholder coach until a real LLM backend is wired — but good enough to
        smoke-test the full pipeline end to end without any model download.
        """
        t = text.lower()
        tips = []
        if "no-ball" in t:
            tips.append("No ball detected — check camera angle / lighting for this segment.")
        if "fast-exchange" in t:
            tips.append("Fast exchange at the net: stay low, paddle up, react with compact blocks rather than big swings.")
        if "rally" in t:
            tips.append("Sustained rally: keep resetting to the kitchen line and aim deep to limit your opponent's angles.")
        if "serve" in t or "reset" in t:
            tips.append("Serve/reset phase: get into ready position early and target a deep, consistent serve.")
        if "dive" in t or "defensive" in t:
            tips.append("Defensive scramble: prioritise a high, soft return to buy time and regain court position.")
        if not tips:
            tips.append("Maintain ready position, split-step on contact, and move to the non-volley line when possible.")
        return " ".join(tips)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _describe(state: Dict[str, Any]) -> str:
        """One-line natural description of a game state for prompting."""
        n = state.get("num_players", 0)
        action = state.get("action", "unknown")
        ball = state.get("ball", {}) or {}
        ball_txt = "ball in play" if ball.get("centroid") else "ball not visible"
        return f"{n} players, action={action}, {ball_txt}"


def generate_feedback(state: Dict[str, Any], backend: str = "rule") -> str:
    """Convenience one-shot: feedback for a single game state."""
    return CoachingFeedbackGenerator(backend=backend).from_game_state(state)
