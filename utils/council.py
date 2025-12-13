from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import streamlit as st

from .llm_handler import LLMHandler


# ---------- Helpers ----------

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from a possibly noisy LLM response.
    Looks for the first '{' and last '}' and parses that substring.
    """
    try:
        text = text.strip()
        # Fast path – already pure JSON
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        # Fallback: find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])

    except Exception as e:
        st.error(f"JSON parse error: {e}")

    # If everything fails, return a safe default
    return {
        "resume_jd_match_score": 0,
        "resume_overall_score": 0,
        "strengths": [],
        "gaps": [],
        "specific_rewrite_suggestions": [],
        "short_summary": "Could not parse model response as JSON."
    }


def _build_evaluator_prompt(resume_text: str, jd_text: str) -> str:
    return f"""
You are a strict but constructive career coach and ATS expert helping a student tailor their resume to a specific job.

Evaluate the match between the following RESUME and JOB DESCRIPTION (JD).

JOB DESCRIPTION (JD):
---------------------
{jd_text}

RESUME:
-------
{resume_text}

1. Score how well this resume matches the JD on a 0–100 scale.
2. Score the overall quality of the resume itself on a 0–100 scale.
3. List 3–5 key strengths that are relevant to this JD.
4. List 3–5 gaps or weaknesses relative to this JD.
5. Suggest 3–7 VERY SPECIFIC edits or rewrites to improve the resume for this JD (mention section/line if possible).
6. Give a 2–3 sentence summary that a student can easily understand.

Respond ONLY in valid JSON, using this schema and numeric ranges exactly:

{{
  "resume_jd_match_score": 0,
  "resume_overall_score": 0,
  "strengths": [],
  "gaps": [],
  "specific_rewrite_suggestions": [],
  "short_summary": ""
}}
""".strip()


def _build_peer_review_prompt(
    resume_text: str,
    jd_text: str,
    own_opinion: Dict[str, Any],
    peer_opinions: List[Dict[str, Any]]
) -> str:
    return f"""
You are one of multiple AI experts evaluating a resume vs a job description.

JOB DESCRIPTION (JD):
---------------------
{jd_text}

RESUME:
-------
{resume_text}

YOUR ORIGINAL OPINION (JSON):
-----------------------------
{json.dumps(own_opinion, indent=2)}

ANONYMISED PEER OPINIONS (JSON LIST):
-------------------------------------
{json.dumps(peer_opinions, indent=2)}

Tasks:
1. Briefly state what you AGREE with in the peer opinions.
2. Briefly state what you DISAGREE with or think is over/under-stated.
3. Update your scores and suggestions if you think you were too harsh/lenient or missed an important gap.

Return ONLY a JSON object in this schema (do not include your commentary, only the final JSON):

{{
  "resume_jd_match_score": 0,
  "resume_overall_score": 0,
  "strengths": [],
  "gaps": [],
  "specific_rewrite_suggestions": [],
  "short_summary": ""
}}
""".strip()


def _build_judge_prompt(
    resume_text: str,
    jd_text: str,
    opinions: List[Dict[str, Any]]
) -> str:
    return f"""
You are an impartial judge combining the opinions of multiple AI experts
to help a student improve their resume for a specific job.

JOB DESCRIPTION (JD):
---------------------
{jd_text}

RESUME:
-------
{resume_text}

EXPERT OPINIONS (ANONYMISED):
-----------------------------
{json.dumps(opinions, indent=2)}

Based on all of this, you must produce ONE final verdict.

1. Give a FINAL MATCH SCORE (0–100) based on all opinions.
2. Give a FINAL RESUME QUALITY SCORE (0–100).
3. Combine strengths into 3–7 KEY STRENGTHS (deduplicated).
4. Combine gaps into 3–7 KEY GAPS (deduplicated).
5. Synthesize the TOP 5 CONCRETE EDITS the student should make TODAY.
6. Explain the verdict in 4–6 sentences in simple, encouraging language.

Respond ONLY in JSON with this schema:

{{
  "final_match_score": 0,
  "final_resume_score": 0,
  "key_strengths": [],
  "key_gaps": [],
  "top_5_edits_to_make_today": [],
  "explanation": ""
}}
""".strip()


def _anonymize_opinions(opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip model name and re-label as Opinion_1, Opinion_2, ..."""
    anon = []
    for i, op in enumerate(opinions):
        anon.append({
            "id": f"Opinion_{i+1}",
            "resume_jd_match_score": op.get("resume_jd_match_score", 0),
            "resume_overall_score": op.get("resume_overall_score", 0),
            "strengths": op.get("strengths", []),
            "gaps": op.get("gaps", []),
            "specific_rewrite_suggestions": op.get("specific_rewrite_suggestions", []),
            "short_summary": op.get("short_summary", "")
        })
    return anon


# ---------- Core classes ----------

@dataclass
class CouncilMember:
    """One LLM in the council (e.g., ChatGPT, Gemini, DeepSeek)."""
    name: str         # Display name
    model_name: str   # Key used in Config.LLM_MODELS and LLMHandler

    def evaluate(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        handler = LLMHandler(self.model_name)
        prompt = _build_evaluator_prompt(resume_text, jd_text)
        raw = handler.generate_raw(prompt)
        parsed = _extract_json(raw)
        parsed["model"] = self.name
        return parsed

    def revise_with_peers(
        self,
        resume_text: str,
        jd_text: str,
        own_opinion: Dict[str, Any],
        peer_opinions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        handler = LLMHandler(self.model_name)
        prompt = _build_peer_review_prompt(resume_text, jd_text, own_opinion, peer_opinions)
        raw = handler.generate_raw(prompt)
        parsed = _extract_json(raw)
        parsed["model"] = self.name
        return parsed


class CouncilJudge:
    """Final judge model that combines all member opinions."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def decide(
        self,
        resume_text: str,
        jd_text: str,
        opinions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        handler = LLMHandler(self.model_name)
        prompt = _build_judge_prompt(resume_text, jd_text, opinions)
        raw = handler.generate_raw(prompt)

        # Judge has a different schema
        try:
            text = raw.strip()
            if not (text.startswith("{") and text.endswith("}")):
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    text = text[start : end + 1]
            return json.loads(text)
        except Exception as e:
            st.error(f"JSON parse error in judge: {e}")
            return {
                "final_match_score": 0,
                "final_resume_score": 0,
                "key_strengths": [],
                "key_gaps": [],
                "top_5_edits_to_make_today": [],
                "explanation": "Could not parse judge response as JSON."
            }


@dataclass
class CouncilConfig:
    members: List[CouncilMember]
    judge_model_name: str


class CouncilOrchestrator:
    """
    Runs the end-to-end council:
    1) Each member evaluates independently.
    2) Optional peer review round.
    3) Judge combines opinions into a final verdict.
    """

    def __init__(self, config: CouncilConfig):
        self.members = config.members
        self.judge = CouncilJudge(config.judge_model_name)

    def run(
        self,
        resume_text: str,
        jd_text: str,
        peer_review: bool = True
    ) -> Dict[str, Any]:
        # Round 1 – independent evaluations
        base_opinions: List[Dict[str, Any]] = []
        for m in self.members:
            with st.spinner(f"{m.name} is evaluating..."):
                op = m.evaluate(resume_text, jd_text)
            base_opinions.append(op)

        if peer_review:
            # Prepare anonymised opinions
            anon = _anonymize_opinions(base_opinions)
            revised: List[Dict[str, Any]] = []
            for i, m in enumerate(self.members):
                # Own opinion with full detail
                own = base_opinions[i]
                # Peers = all anon opinions except the same index
                peers = [o for j, o in enumerate(anon) if j != i]
                with st.spinner(f"{m.name} is revising opinion after peer review..."):
                    rev = m.revise_with_peers(resume_text, jd_text, own, peers)
                revised.append(rev)
            final_opinions = revised
        else:
            final_opinions = base_opinions

        # Judge phase
        with st.spinner("Judge is synthesizing final verdict..."):
            verdict = self.judge.decide(resume_text, jd_text, final_opinions)

        return {
            "member_opinions": final_opinions,
            "verdict": verdict
        }
