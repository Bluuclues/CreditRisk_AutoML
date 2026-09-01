"""
ai_assistant.py
Natural-Language Analytics Assistant for the KBA Credit Risk AutoML platform.

Turns free-text questions into a structured answer + auto-generated dashboards
(Plotly charts, tables and KPI cards). It works fully offline out of the box, and
optionally uses a local **Ollama** LLM (http://localhost:11434) to interpret free-form
questions and write a natural-language narrative.

Architecture (accuracy first):
  1. Interpret the question  -> intent + parameters  (Ollama if available, else regex)
  2. Execute the analysis    -> charts/tables/KPIs computed deterministically (never hallucinated)
  3. Narrate the answer      -> Ollama summarises the *computed facts* (else a templated answer)
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import os
import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Columns that are identifiers/dates and therefore never treated as segments.
_ID_LIKE = {"borrower_id", "loan_no", "loan_date", "due_date", "payoff_date", "session_id"}
_TARGET = "default_flag"


class OllamaClient:
    """Minimal dependency-free client for a local Ollama server (no `requests` needed)."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = (model or os.getenv("OLLAMA_MODEL") or "").strip()
        self.timeout = timeout

    def _get(self, path: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self, timeout: int = 3) -> List[str]:
        try:
            data = self._get("/api/tags", timeout=timeout)
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def available(self) -> bool:
        return bool(self.list_models(timeout=3))

    def resolve_model(self) -> str:
        """Returns the configured model, or auto-detects the first sensible local model."""
        if self.model:
            return self.model
        models = self.list_models(timeout=3)
        if not models:
            return "llama3.2"
        for pref in ("llama3", "qwen", "mistral", "gemma", "phi", "deepseek", "command", "llama2", "codellama"):
            for m in models:
                if pref in m.lower():
                    return m
        return models[0]

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        data = self._post("/api/chat", payload)
        return (data.get("message") or {}).get("content", "")


class NLQueryEngine:
    """Natural-language -> analysis + dashboard engine (offline, with optional Ollama LLM)."""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm
        self._params: Dict[str, Any] = {}

        # Intent name -> handler method.
        self._HANDLERS = {
            "help": self._handle_help,
            "borrower": self._handle_borrower,
            "top_risky": self._handle_top_risky,
            "feature_importance": self._handle_feature_importance,
            "correlation": self._handle_correlation,
            "distribution": self._handle_distribution,
            "risk_tiers": self._handle_risk_tiers,
            "segment": self._handle_segment,
            "compare": self._handle_compare,
            "overview": self._handle_portfolio_overview,
        }

        # Regex fallback router: (intent, pattern) checked in order.
        self._RULE_HANDLERS = [
            ("help", r"\b(help|what can you|commands|examples|how do i|how to)\b"),
            ("borrower", r"\b(cust-\d+|borrower[_ ]?id|who is|details? for)\b"),
            ("top_risky", r"\b(top|riskiest|most risky|highest|worst|danger)\b"),
            ("feature_importance", r"\b(feature importance|driv(e|es|ers?)|important feature|what matters|shap|most important|key factors?)\b"),
            ("correlation", r"\b(correl|relationship between|related to)"),
            ("distribution", r"\b(distribution|histogram|spread of|how is .* distributed)\b"),
            ("risk_tiers", r"\b(risk tier|risk breakdown|breakdown of risk|how many (high|medium|low) risk|tier)\b"),
            ("segment", r"\b(by |across |per |for each |breakdown|segment)\b"),
            ("compare", r"\b(compare|versus|vs\.?|difference between)\b"),
        ]

    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Interprets a free-text question and returns an answer + dashboards.

        `context` may contain:
            df: pd.DataFrame             - the analytical feature store / scored portfolio
            probs: np.ndarray            - predicted default probabilities (optional)
            leaderboard: pd.DataFrame    - AutoML comparative leaderboard (optional)
            feature_importance: pd.DataFrame - global SHAP importance (optional)
            champion_name: str           - winning model name (optional)
        """
        query = (query or "").strip()
        if not query:
            return self._help_result("Type a question to get started.")

        df = context.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._help_result("Please upload and ingest loan data first.")

        try:
            frame, has_probs = self._prepare_frame(df, context.get("probs"))

            self._params = {}
            intent, params = self._interpret(query, frame)
            self._params = params or {}

            handler = self._HANDLERS.get(intent, self._handle_portfolio_overview)
            result = handler(query, frame, context, has_probs)
            result["intent"] = intent

            result = self._maybe_narrate(query, result, context)
            return result
        except Exception as e:  # never crash the app on a bad query
            return self._fallback_result(str(e))

    # ------------------------------------------------------------------ #
    # Interpretation (Ollama first, regex fallback)
    # ------------------------------------------------------------------ #
    def _interpret(self, query: str, frame: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        if self.llm is not None:
            try:
                interpreted = self._llm_interpret(query, frame)
                if interpreted:
                    return interpreted
            except Exception:
                pass
        return self._rule_interpret(query)

    def _rule_interpret(self, query: str) -> Tuple[str, Dict[str, Any]]:
        lowered = query.lower()
        for name, pattern in self._RULE_HANDLERS:
            if re.search(pattern, lowered):
                return name, {}
        return "overview", {}

    def _llm_interpret(self, query: str, frame: pd.DataFrame) -> Optional[Tuple[str, Dict[str, Any]]]:
        seg_cols = self._segment_columns(frame)
        num_cols = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in _ID_LIKE][:30]

        system = (
            "You are the intent router for a credit-risk analytics dashboard. "
            "Map the user's question to exactly ONE intent and extract its parameters. "
            "Respond with valid JSON only, matching this exact schema:\n"
            '{"intent": "overview|segment|compare|top_risky|feature_importance|correlation|distribution|risk_tiers|borrower|help", '
            '"segment": "column or null", "metric": "default_rate|mean_pd|mean_amount|total_amount|count|null", '
            '"top_n": integer or null, "column": "column or null", "borrower_id": "id or null"}\n\n'
            f"Available segment columns: {seg_cols}\n"
            f"Available numeric columns: {num_cols}\n\n"
            "Metric guidance: 'default rate/defaulted' -> default_rate; 'risk/probability of default/pd' -> mean_pd; "
            "'total amount/exposure/volume' -> total_amount; 'average amount' -> mean_amount; 'how many/count' -> count. "
            "Choose segment/column values ONLY from the lists above. Do not add commentary."
        )
        raw = self.llm.chat(system, query, json_mode=True)
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None

        intent = str(data.get("intent", "")).strip().lower()
        if intent not in self._HANDLERS:
            return None

        params: Dict[str, Any] = {}
        seg = data.get("segment")
        if seg and seg in frame.columns:
            params["segment"] = seg
        metric = data.get("metric")
        if metric in {"default_rate", "mean_pd", "mean_amount", "total_amount", "count"}:
            params["metric"] = metric
        top_n = data.get("top_n")
        if isinstance(top_n, (int, float)):
            params["top_n"] = int(min(max(1, int(top_n)), 50))
        col = data.get("column")
        if col and col in frame.columns:
            params["column"] = col
        bid = data.get("borrower_id")
        if bid:
            params["borrower_id"] = str(bid).strip().upper()
        return intent, params

    # ------------------------------------------------------------------ #
    # Narrative (Ollama summarises computed facts)
    # ------------------------------------------------------------------ #
    def _maybe_narrate(self, query: str, result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm is None:
            return result
        try:
            facts = self._facts_for_llm(query, result)
            system = (
                "You are a senior credit-risk analyst assistant. Write a concise, accurate natural-language answer "
                "to the user's question using ONLY the computed facts provided below. Do not invent numbers. "
                "Format with Markdown (bold key figures, short bullet points where helpful). Keep it under 120 words."
            )
            narrative = self.llm.chat(system, facts)
            if narrative and narrative.strip():
                result["answer"] = narrative.strip()
                result["answer_source"] = "ollama"
        except Exception:
            pass
        return result

    @staticmethod
    def _facts_for_llm(query: str, result: Dict[str, Any]) -> str:
        parts = [f"User question: {query}"]
        if result.get("answer"):
            parts.append(f"Deterministic answer: {result.get('answer')}")
        kpis = result.get("kpis") or []
        if kpis:
            parts.append("Key figures: " + "; ".join(f"{k.get('label')}={k.get('value')}" for k in kpis))
        for i, tbl in enumerate(result.get("tables") or []):
            try:
                parts.append(f"Table {i + 1} preview:\n{tbl.head(6).to_csv(index=False)}")
            except Exception:
                pass
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Frame preparation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare_frame(df: pd.DataFrame, probs: Optional[np.ndarray]) -> Tuple[pd.DataFrame, bool]:
        frame = df.copy()
        has_probs = probs is not None and len(probs) == len(frame)
        if has_probs:
            probs = np.asarray(probs, dtype=float)
            frame["predicted_pd"] = probs
            frame["predicted_pd_pct"] = probs * 100.0
            frame["risk_tier"] = np.where(
                probs >= 0.60, "High Risk",
                np.where(probs >= 0.30, "Medium Risk", "Low Risk")
            )
        return frame, has_probs

    @staticmethod
    def _segment_columns(frame: pd.DataFrame) -> List[str]:
        """Candidate categorical/segmentation columns, excluding IDs and the target."""
        cols = []
        for c in frame.columns:
            if c in _ID_LIKE or c == _TARGET:
                continue
            if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_categorical_dtype(frame[c]) or isinstance(frame[c].dtype, pd.StringDtype):
                cols.append(c)
                continue
            if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique() <= 20:
                cols.append(c)
        return cols

    def _detect_segment(self, query: str, frame: pd.DataFrame) -> Optional[str]:
        if self._params.get("segment") and self._params["segment"] in frame.columns:
            return self._params["segment"]

        lowered = query.lower()
        seg_candidates = self._segment_columns(frame)

        # Direct column-name mentions take priority (case-insensitive).
        for c in seg_candidates:
            if c.lower().replace("_", " ") in lowered or c.lower() in lowered:
                return c

        alias_map = {
            "county": "county_code",
            "country": "country_code",
            "jurisdiction": "country_code",
            "type": "borrower_type",
            "segment": "borrower_type",
            "sector": "borrower_type",
            "tier": "risk_tier",
            "risk": "risk_tier",
            "tenure": "tenure_days",
            "term": "tenure_days",
            "amount": "amount",
            "principal": "amount",
        }
        for alias, col in alias_map.items():
            if alias in lowered and col in frame.columns:
                return col

        # Fallback: first object segment column.
        for c in seg_candidates:
            if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_categorical_dtype(frame[c]):
                return c
        return seg_candidates[0] if seg_candidates else None

    def _detect_metric(self, query: str) -> str:
        if self._params.get("metric") in {"default_rate", "mean_pd", "mean_amount", "total_amount", "count"}:
            return self._params["metric"]

        lowered = query.lower()
        if re.search(r"default rate|defaulted|delinquen|default rate|bad rate", lowered):
            return "default_rate"
        if re.search(r"risk|probability|pd\b|probab", lowered):
            return "mean_pd"
        if re.search(r"avg|average|mean", lowered) and re.search(r"amount|principal|exposure", lowered):
            return "mean_amount"
        if re.search(r"amount|principal|exposure|volume|value|sum", lowered):
            return "total_amount"
        if re.search(r"count|how many|number of|total", lowered):
            return "count"
        return "default_rate"

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    def _handle_help(self, query, frame, context, has_probs) -> Dict[str, Any]:
        return self._help_result(None)

    def _help_result(self, note: Optional[str]) -> Dict[str, Any]:
        intro = f"🤖 **KBA AI Analyst** — {note}" if note else "🤖 **KBA AI Analyst** — ask me anything about your portfolio."
        suggestions = [
            "📊 'Show me a portfolio overview'",
            "🎯 'What is the default rate by county?'",
            "🚨 'Who are the top 10 riskiest borrowers?'",
            "🧭 'What drives default risk?' (feature importance)",
            "📈 'Show the distribution of loan amount'",
            "🔎 'What correlates with default?'",
            "⚖️ 'Compare Retail vs SME default rates'",
        ]
        return {
            "answer": f"{intro}\n\nTry one of these:\n\n" + "\n".join(suggestions),
            "figures": [],
            "tables": [],
            "kpis": [],
        }

    def _handle_portfolio_overview(self, query, frame, context, has_probs) -> Dict[str, Any]:
        kpis = []
        total = len(frame)

        hist_rate = None
        if _TARGET in frame.columns:
            hist_rate = float(frame[_TARGET].mean()) * 100.0
            kpis.append({"label": "Portfolio Size", "value": f"{total:,}"})
            kpis.append({"label": "Historical Default Rate", "value": f"{hist_rate:.1f}%"})
        else:
            kpis.append({"label": "Portfolio Size", "value": f"{total:,}"})

        mean_pd = None
        if has_probs:
            mean_pd = float(frame["predicted_pd"].mean()) * 100.0
            high = int((frame["risk_tier"] == "High Risk").sum())
            med = int((frame["risk_tier"] == "Medium Risk").sum())
            low = int((frame["risk_tier"] == "Low Risk").sum())
            kpis.append({"label": "Predicted Mean PD", "value": f"{mean_pd:.1f}%"})
            kpis.append({"label": "High-Risk Borrowers", "value": f"{high:,}"})
            kpis.append({"label": "Risk Mix", "value": f"{high}/{med}/{low} (H/M/L)"})

        figures = []
        if has_probs:
            tier_counts = frame["risk_tier"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
            colors = ["#dc2626", "#d97706", "#16a34a"]
            fig = go.Figure(go.Bar(
                x=tier_counts.index.tolist(),
                y=tier_counts.values.tolist(),
                marker_color=colors,
                text=tier_counts.values.tolist(),
                textposition="auto",
            ))
            fig.update_layout(title="Risk-Tier Distribution", xaxis_title="Risk Tier", yaxis_title="Borrowers", template="plotly_white", height=380)
            figures.append(fig)
        elif _TARGET in frame.columns:
            fig = px.pie(
                names=["Performing (0)", "Defaulted (1)"],
                values=[int((frame[_TARGET] == 0).sum()), int((frame[_TARGET] == 1).sum())],
                title="Historical Default Composition",
                color_discrete_sequence=["#10b981", "#ef4444"],
            )
            figures.append(fig)

        lines = [f"Your portfolio holds **{total:,} records**."]
        if hist_rate is not None:
            lines.append(f"The historical default rate is **{hist_rate:.1f}%**.")
        if mean_pd is not None:
            lines.append(f"The model predicts an average Probability of Default (PD) of **{mean_pd:.1f}%**.")
        if context.get("champion_name"):
            lines.append(f"Champion model: **{context.get('champion_name')}**.")

        return {"answer": "\n\n".join(lines), "figures": figures, "tables": [], "kpis": kpis}

    def _handle_risk_tiers(self, query, frame, context, has_probs) -> Dict[str, Any]:
        if not has_probs:
            return {"answer": "Risk tiers are only available after the AutoML pipeline has scored the portfolio. Run the model first, then ask me again.", "figures": [], "tables": [], "kpis": []}

        tier_counts = frame["risk_tier"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        total = max(int(tier_counts.sum()), 1)
        colors = ["#dc2626", "#d97706", "#16a34a"]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=tier_counts.index.tolist(), y=tier_counts.values.tolist(), marker_color=colors, text=tier_counts.values.tolist(), textposition="auto"))
        fig.update_layout(title="Borrowers by Risk Tier", xaxis_title="Risk Tier", yaxis_title="Borrowers", template="plotly_white", height=380)

        tbl = pd.DataFrame({
            "Risk Tier": tier_counts.index.tolist(),
            "Borrowers": tier_counts.values.tolist(),
            "Share (%)": [round(v / total * 100, 1) for v in tier_counts.values.tolist()],
        })

        kpis = [
            {"label": "High Risk", "value": f"{int(tier_counts['High Risk']):,}"},
            {"label": "Medium Risk", "value": f"{int(tier_counts['Medium Risk']):,}"},
            {"label": "Low Risk", "value": f"{int(tier_counts['Low Risk']):,}"},
        ]
        answer = f"The portfolio breaks down into **{int(tier_counts['High Risk'])} high**, **{int(tier_counts['Medium Risk'])} medium**, and **{int(tier_counts['Low Risk'])} low** risk borrowers."
        return {"answer": answer, "figures": [fig], "tables": [tbl], "kpis": kpis}

    def _handle_top_risky(self, query, frame, context, has_probs) -> Dict[str, Any]:
        top_n = self._extract_top_n(query)

        if not has_probs:
            return {"answer": "I need the model's Probability of Default (PD) to rank the riskiest borrowers. Run the AutoML pipeline first, then ask again.", "figures": [], "tables": [], "kpis": []}

        ranked = frame.sort_values("predicted_pd", ascending=False).head(top_n)
        id_col = "borrower_id" if "borrower_id" in ranked.columns else None
        rows = []
        for _, r in ranked.iterrows():
            row = {}
            if id_col:
                row["Borrower ID"] = r[id_col]
            row["Predicted PD (%)"] = round(float(r["predicted_pd"]) * 100, 1)
            if "amount" in ranked.columns:
                row["Principal (KES)"] = f"{float(r['amount']):,.0f}"
            if _TARGET in ranked.columns:
                row["Actual Default"] = int(r[_TARGET])
            rows.append(row)

        tbl = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(
            x=[round(float(r["predicted_pd"]) * 100, 1) for _, r in ranked.iterrows()],
            y=[r[id_col] if id_col else f"#{i+1}" for i, (_, r) in enumerate(ranked.iterrows())],
            orientation="h",
            marker_color="#dc2626",
            text=[f"{round(float(r['predicted_pd'])*100,1)}%" for _, r in ranked.iterrows()],
            textposition="auto",
        ))
        fig.update_layout(title=f"Top {top_n} Riskiest Borrowers by Predicted PD", xaxis_title="Predicted Probability of Default (%)", yaxis_title="Borrower", template="plotly_white", height=420)

        answer = f"Here are the **{top_n} riskiest borrowers**, ranked by the model's predicted Probability of Default."
        return {"answer": answer, "figures": [fig], "tables": [tbl], "kpis": []}

    def _handle_feature_importance(self, query, frame, context, has_probs) -> Dict[str, Any]:
        imp = context.get("feature_importance")
        if imp is None or not isinstance(imp, pd.DataFrame) or imp.empty:
            explainer = context.get("explainer")
            if explainer is not None and hasattr(explainer, "get_global_feature_importance_df"):
                try:
                    imp = explainer.get_global_feature_importance_df(top_n=25)
                except Exception:
                    imp = None
        if imp is None or not isinstance(imp, pd.DataFrame) or imp.empty:
            return {"answer": "Feature importance is available after the AutoML pipeline has fitted the SHAP explainer. Run the model first.", "figures": [], "tables": [], "kpis": []}

        top = imp.head(15)
        fig = go.Figure(go.Bar(
            x=top.iloc[::-1]["Mean Absolute SHAP"],
            y=top.iloc[::-1]["Feature Name"],
            orientation="h",
            marker_color="#2563eb",
            text=[f"{pct:.1f}%" for pct in top.iloc[::-1]["Impact Percentage (%)"]] if "Impact Percentage (%)" in top.columns else None,
            textposition="auto",
        ))
        fig.update_layout(title="Top Risk Drivers (Mean |SHAP|)", xaxis_title="Mean |SHAP Value|", template="plotly_white", height=460)

        top_driver = str(top.iloc[0]["Feature Name"]) if len(top) else "N/A"
        answer = f"The single biggest driver of default risk is **{top_driver}** (see chart for the full ranking)."
        return {"answer": answer, "figures": [fig], "tables": [imp.head(25)], "kpis": []}

    def _handle_correlation(self, query, frame, context, has_probs) -> Dict[str, Any]:
        _exclude = set(_ID_LIKE) | {"session_id", "predicted_pd", "predicted_pd_pct"}
        numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in _exclude]
        if len(numeric) < 2:
            return {"answer": "Not enough numeric features to compute correlations.", "figures": [], "tables": [], "kpis": []}

        corr = frame[numeric].corr()
        clean = {c: c.replace("feat_", "").replace("_", " ").title() for c in numeric}
        corr_clean = corr.rename(index=clean, columns=clean)

        fig = go.Figure(go.Heatmap(
            z=corr_clean.values,
            x=corr_clean.columns.tolist(),
            y=corr_clean.columns.tolist(),
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=np.round(corr_clean.values, 2), texttemplate="%{text}",
            colorbar=dict(title="Pearson r"),
        ))
        fig.update_layout(title="Feature Correlation Matrix", xaxis=dict(tickangle=-45), template="plotly_white", height=520)

        # Highlight correlations with the target if present.
        target_corr = None
        if _TARGET in corr.columns:
            target_corr = corr[_TARGET].drop(_TARGET).abs().sort_values(ascending=False).head(5)
        answer = "Here is the correlation matrix of numeric features."
        if target_corr is not None and len(target_corr):
            top_feat = clean.get(target_corr.index[0], target_corr.index[0])
            answer += f" The feature most correlated with default is **{top_feat}** (|r| = {target_corr.iloc[0]:.2f})."
        return {"answer": answer, "figures": [fig], "tables": [corr_clean.round(3)], "kpis": []}

    def _handle_distribution(self, query, frame, context, has_probs) -> Dict[str, Any]:
        col = self._detect_column(query, frame)
        if col is None:
            col = "amount" if "amount" in frame.columns else frame.select_dtypes(include=[np.number]).columns[0]

        series = frame[col].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            vc = series.value_counts().head(15)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={"x": col, "y": "Count"}, title=f"Distribution of {col}")
        else:
            fig = px.histogram(frame, x=col, nbins=30, color=_TARGET if _TARGET in frame.columns else None,
                               barmode="overlay", title=f"Distribution of {col}",
                               color_discrete_sequence=["#10b981", "#ef4444"])
            fig.update_layout(template="plotly_white", height=400)

        stats = {
            "Count": int(series.shape[0]),
            "Mean": round(float(series.mean()), 2) if pd.api.types.is_numeric_dtype(series) else "-",
            "Median": round(float(series.median()), 2) if pd.api.types.is_numeric_dtype(series) else "-",
            "Min": round(float(series.min()), 2) if pd.api.types.is_numeric_dtype(series) else "-",
            "Max": round(float(series.max()), 2) if pd.api.types.is_numeric_dtype(series) else "-",
        }
        answer = f"Distribution of **{col}**: mean {stats['Mean']}, median {stats['Median']}, range [{stats['Min']}, {stats['Max']}] across {stats['Count']:,} records."
        return {"answer": answer, "figures": [fig], "tables": [], "kpis": []}

    def _handle_segment(self, query, frame, context, has_probs) -> Dict[str, Any]:
        seg_col = self._detect_segment(query, frame)
        if seg_col is None:
            return self._fallback_result("I couldn't identify a segmentation column for that question.")

        metric = self._detect_metric(query)
        grouped = self._aggregate_by(frame, seg_col, metric)

        title = f"{metric.replace('_', ' ').title()} by {seg_col.replace('_', ' ').title()}"
        fig = go.Figure(go.Bar(
            x=grouped["segment"].astype(str).tolist(),
            y=grouped["value"].astype(float).tolist(),
            text=[self._fmt(v) for v in grouped["value"].astype(float).tolist()],
            textposition="auto",
            marker_color="#3b82f6",
        ))
        fig.update_layout(title=title, xaxis_title=seg_col.replace("_", " ").title(), yaxis_title=metric.replace("_", " ").title(), template="plotly_white", height=420)

        tbl = grouped.rename(columns={"segment": seg_col.replace("_", " ").title(), "value": metric.replace("_", " ").title()})
        top_seg = str(grouped.iloc[0]["segment"])
        top_val = self._fmt(float(grouped.iloc[0]["value"]))
        answer = f"Across **{seg_col.replace('_', ' ').title()}**, the highest value is **{top_seg}** ({top_val}). Full breakdown in the chart and table."
        return {"answer": answer, "figures": [fig], "tables": [tbl], "kpis": []}

    def _handle_compare(self, query, frame, context, has_probs) -> Dict[str, Any]:
        seg_col = self._detect_segment(query, frame)
        if seg_col is None:
            return self._fallback_result("Tell me which segments to compare (e.g. 'compare Retail vs SME').")
        return self._handle_segment(query, frame, context, has_probs)

    def _handle_borrower(self, query, frame, context, has_probs) -> Dict[str, Any]:
        if "borrower_id" not in frame.columns:
            return {"answer": "The dataset does not contain a `borrower_id` column.", "figures": [], "tables": [], "kpis": []}

        bid = self._params.get("borrower_id")
        if not bid:
            m = re.search(r"(cust-\d+)", query, re.IGNORECASE)
            if not m:
                return {"answer": "Please give me a Borrower ID, e.g. 'show me CUST-10492'.", "figures": [], "tables": [], "kpis": []}
            bid = m.group(1).upper()

        row = frame[frame["borrower_id"].astype(str).str.upper() == str(bid).upper()]
        if row.empty:
            return {"answer": f"No borrower found with ID **{bid}**.", "figures": [], "tables": [], "kpis": []}

        r = row.iloc[0]
        kpis = []
        if has_probs:
            pd_pct = float(r["predicted_pd"]) * 100
            kpis.append({"label": "Predicted PD", "value": f"{pd_pct:.1f}%"})
            kpis.append({"label": "Risk Tier", "value": str(r["risk_tier"])})
        if "amount" in r:
            kpis.append({"label": "Principal (KES)", "value": f"{float(r['amount']):,.0f}"})
        if _TARGET in r:
            kpis.append({"label": "Actual Default", "value": "Yes" if int(r[_TARGET]) == 1 else "No"})

        detail_cols = [c for c in frame.columns if c not in _ID_LIKE and c != "session_id"]
        tbl = pd.DataFrame({"Feature": detail_cols, "Value": [str(r[c]) for c in detail_cols]})
        answer = f"Profile for **{bid}**:"
        return {"answer": answer, "figures": [], "tables": [tbl], "kpis": kpis}

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate_by(frame: pd.DataFrame, seg_col: str, metric: str) -> pd.DataFrame:
        g = frame.groupby(seg_col, dropna=False)
        if metric == "count":
            s = g.size()
        elif metric == "total_amount":
            s = g["amount"].sum() if "amount" in frame.columns else g.size()
        elif metric == "mean_amount":
            s = g["amount"].mean() if "amount" in frame.columns else g.size()
        elif metric == "mean_pd":
            s = g["predicted_pd"].mean() if "predicted_pd" in frame.columns else pd.Series(dtype=float)
        else:  # default_rate
            s = g[_TARGET].mean() * 100.0 if _TARGET in frame.columns else pd.Series(dtype=float)

        out = pd.DataFrame({"segment": s.index.astype(str), "value": s.values})
        out = out.sort_values("value", ascending=False).reset_index(drop=True)
        return out

    def _detect_column(self, query: str, frame: pd.DataFrame) -> Optional[str]:
        if self._params.get("column") and self._params["column"] in frame.columns:
            return self._params["column"]
        lowered = query.lower()
        for c in frame.columns:
            if c.lower().replace("_", " ") in lowered or c.lower() in lowered:
                return c
        return None

    def _extract_top_n(self, query: str) -> int:
        tn = self._params.get("top_n")
        if isinstance(tn, int) and 1 <= tn <= 50:
            return tn
        m = re.search(r"\btop\s+(\d{1,3})\b", query, re.IGNORECASE)
        if m:
            return max(1, min(int(m.group(1)), 50))
        return 10

    @staticmethod
    def _fmt(v: float) -> str:
        if pd.isna(v):
            return "N/A"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:,.1f}"
        return f"{v:,.2f}"

    def _fallback_result(self, reason: str = "") -> Dict[str, Any]:
        note = f"Sorry, I couldn't interpret that query. ({reason})" if reason else "Sorry, I couldn't interpret that query."
        return {
            "answer": f"{note}\n\nHere's a portfolio overview instead:",
            "figures": [],
            "tables": [],
            "kpis": [],
        }
