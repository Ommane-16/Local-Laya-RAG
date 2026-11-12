"""Laya choice-based reranker. Probabilities are model choice probabilities,
not calibrated relevance probabilities."""
from __future__ import annotations
from config import LAYA_MODEL, LAYA_MAX_CANDIDATES

class LayaReranker:
    def __init__(self, model_id: str = LAYA_MODEL,
                 max_candidates: int = LAYA_MAX_CANDIDATES,
                 device: str = "cpu"):
        self.model_id = model_id
        self.max_candidates = min(32, max(1, int(max_candidates)))
        self.device = device
        self.runner = None

    def load(self):
        if self.runner is None:
            import laya
            self.runner = laya.load(model_id_or_path=self.model_id, device=self.device)
        return self.runner

    @staticmethod
    def _probabilities(decision):
        raw = getattr(decision, "probabilities", None) or {}
        if isinstance(raw, dict):
            for key in ("best_passage", "best_candidate", "choice"):
                value = raw.get(key)
                if isinstance(value, dict):
                    return value
            return raw
        return {}

    def rerank_with_details(self, query: str, candidates: list[dict],
                            top_k: int = 10) -> dict:
        clean = [
            dict(c) for c in (candidates or [])
            if str(c.get("text", "")).strip()
        ][:self.max_candidates]
        if not clean:
            return {"results": [], "decision": None, "status": "no_candidates"}
        if len(clean) == 1:
            clean[0]["laya_selected"] = True
            clean[0]["laya_probability"] = None
            clean[0]["method"] = "hybrid_laya"
            return {"results": clean[:top_k], "decision": None, "status": "single_candidate"}

        runner = self.load()
        labels = [f"Passage {i + 1}" for i in range(len(clean))]
        state = {"query": query}
        for label, candidate in zip(labels, clean):
            state[label] = candidate["text"]
        questions = {
            "best_passage": {
                "type": "choice",
                "instructions": (
                    "Choose the passage that most directly and substantively answers "
                    "the user's query. Prefer definitions and explanatory content over "
                    "table-of-contents entries, indexes, code fragments, or incidental "
                    "keyword mentions. If all are weak, choose the least weak passage."
                ),
                "criteria": {label: None for label in labels},
            }
        }
        import laya
        decision = laya.decide(
            runner, state, questions=questions, return_details=True
        )
        values = getattr(decision, "values", {}) or {}
        selected = values.get("best_passage") if isinstance(values, dict) else None
        probs = self._probabilities(decision)
        normalized = {}
        for i, label in enumerate(labels):
            value = probs.get(label) if isinstance(probs, dict) else None
            if isinstance(value, (int, float)):
                normalized[label] = float(value)
            else:
                normalized[label] = None
        if selected not in labels:
            available = [(p, label) for label, p in normalized.items() if p is not None]
            selected = max(available)[1] if available else labels[0]
        ordered_labels = [selected] + [
            label for label in sorted(
                labels,
                key=lambda x: normalized[x] if normalized[x] is not None else -1,
                reverse=True
            ) if label != selected
        ]
        by_label = dict(zip(labels, clean))
        results = []
        for label in ordered_labels:
            item = by_label[label]
            item["laya_selected"] = label == selected
            item["laya_probability"] = normalized[label]
            item["method"] = "hybrid_laya"
            results.append(item)
        return {"results": results[:top_k], "decision": decision, "status": "ok"}

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 10) -> list[dict]:
        return self.rerank_with_details(query, candidates, top_k)["results"]
