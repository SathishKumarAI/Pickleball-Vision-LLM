"""Coaching feedback generation.

Takes structured game state (from
``src.integration.fusion.game_state.GameStateBuilder``) and produces natural
language coaching feedback.

Backends (local-first, OSS-first; cloud via a managed provider — not OpenAI):

* ``rule``   — explainable, dependency-free heuristics. **Default**, always works,
  and the timeout/error fallback for the others (P0-5).
* ``hf``     — **canonical real backend**: local HuggingFace ``transformers``
  text-generation (e.g. Mistral/LLaMA). No external API, runs on-box.
* ``cloud``  — managed hosted LLM via ``provider``: ``bedrock`` (AWS, Claude —
  default), ``azure`` (Azure AI/OpenAI), or ``vertex`` (Google, Gemini). Opt-in;
  see docs/BUDGET_PLAN.md for cost. Install the matching extra
  (``[bedrock]`` / ``[azure]`` / ``[vertex]``).

NOTE: the direct OpenAI backend was removed per project decision (use a cloud
provider instead). Its code is retained, commented, below for reference.

Heavy deps (boto3 / azure / google SDKs, transformers) are imported lazily so
this module imports cleanly without any of those extras installed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .prompt_templates import PromptTemplates


class CoachingFeedbackGenerator:
    """Generate coaching feedback from game state or scene captions."""

    def __init__(self, backend: str = "rule", model: Optional[str] = None,
                 provider: str = "bedrock", deadline_s: float = 8.0,
                 fallback: bool = True):
        """
        Args:
            backend: "rule" | "hf" | "cloud".
            model: Backend model id. Defaults: hf→distilgpt2; cloud→provider default.
            provider: cloud provider when backend=="cloud":
                "bedrock" (AWS) | "azure" | "vertex" (Google).
            deadline_s: hard wall-clock for a model call before falling back (P0-5).
            fallback: on timeout/error, return the rule-based coach instead of raising,
                so a slow/failing LLM can never blow the pipeline's latency budget.
        """
        self.backend = backend
        self.model = model
        self.provider = provider
        self.deadline_s = deadline_s
        self.fallback = fallback
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
        if self.backend not in ("cloud", "hf"):
            raise ValueError(f"Unknown backend: {self.backend!r}")

        # P0-5: enforce a wall-clock deadline on the model call; on timeout or any
        # error, degrade to the dependency-free rule coach instead of blowing the
        # latency budget (or failing the whole job).
        import concurrent.futures as cf

        call = self._cloud if self.backend == "cloud" else self._hf
        try:
            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(call, prompt).result(timeout=self.deadline_s)
        except Exception:  # noqa: BLE001 - timeout, API error, cost cap, etc.
            if self.fallback:
                return self._rule_based(caption or prompt)
            raise

    # --- managed cloud backends (AWS / Azure / Google) ---------------------

    def _cloud(self, prompt: str) -> str:
        """Dispatch to the configured managed provider."""
        if self.provider == "bedrock":
            return self._bedrock(prompt)
        if self.provider == "azure":
            return self._azure(prompt)
        if self.provider == "vertex":
            return self._vertex(prompt)
        raise ValueError(f"Unknown cloud provider: {self.provider!r}")

    def _bedrock(self, prompt: str) -> str:
        """AWS Bedrock (default: Claude). Needs the ``[bedrock]`` extra (boto3)."""
        try:
            import json
            import boto3
        except ImportError as e:  # pragma: no cover - needs [bedrock] extra
            raise RuntimeError("bedrock backend needs `pip install boto3`") from e
        if self._client is None:
            self._client = boto3.client("bedrock-runtime",
                                        region_name=os.getenv("AWS_REGION", "us-east-1"))
        model_id = self.model or "anthropic.claude-3-haiku-20240307-v1:0"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = self._client.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"].strip()

    def _azure(self, prompt: str) -> str:
        """Azure AI / Azure OpenAI. Needs the ``[azure]`` extra."""
        try:
            from openai import AzureOpenAI
        except ImportError as e:  # pragma: no cover - needs [azure] extra
            raise RuntimeError("azure backend needs `pip install openai` (Azure SDK)") from e
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            )
        resp = self._client.chat.completions.create(
            model=self.model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def _vertex(self, prompt: str) -> str:
        """Google Vertex AI (Gemini). Needs the ``[vertex]`` extra."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as e:  # pragma: no cover - needs [vertex] extra
            raise RuntimeError("vertex backend needs `pip install google-cloud-aiplatform`") from e
        if self._client is None:
            vertexai.init(project=os.getenv("GCP_PROJECT"),
                          location=os.getenv("GCP_REGION", "us-central1"))
            self._client = GenerativeModel(self.model or "gemini-1.5-flash")
        return self._client.generate_content(prompt).text.strip()

    # --- removed: direct OpenAI backend (use a cloud provider instead) -----
    # Retained, commented, per project decision (see docs/BUDGET_PLAN.md and the
    # `cloud-llm-not-openai` memory). Re-enable only if a direct OpenAI dependency
    # is explicitly wanted again.
    #
    # def _openai(self, prompt: str) -> str:
    #     try:
    #         from openai import OpenAI
    #     except ImportError as e:
    #         raise RuntimeError("openai backend needs `pip install openai`") from e
    #     if self._client is None:
    #         self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #     resp = self._client.chat.completions.create(
    #         model=self.model or "gpt-4o-mini",
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return resp.choices[0].message.content.strip()

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
