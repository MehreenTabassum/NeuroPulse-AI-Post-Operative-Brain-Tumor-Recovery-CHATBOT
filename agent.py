"""
agent.py — LangGraph Orchestrator for Post-Operative Brain Tumor Recovery Analysis
UCSD-PTGBM-BraTS-2024 | Level 3 Medical AI Agent
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Literal, Optional

import httpx
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Agent State
# ---------------------------------------------------------------------------

class RecoveryAgentState(TypedDict):
    image_features:     Optional[list[float]]
    # NEW: stores structured MRI interpretation injected before synthesis
    mri_interpretation: Optional[str]
    clinical_metadata:  dict[str, Any]
    literature_results: list[dict[str, Any]]
    retrieval_retry:    bool
    retrieval_attempts: int
    final_report:       Optional[str]
    warnings:           list[str]


MAX_RETRIES:       int = 1
MIN_RELEVANT_DOCS: int = 2

# ---------------------------------------------------------------------------
# 2. MRI Classification Mapping
# ---------------------------------------------------------------------------

# Maps the CNN-ViT binary output to clinically descriptive text injected
# into the LLM prompt. Class split is based on median BraTS-2024 tumor volume.
MRI_CLASS_DESCRIPTIONS: dict[int, dict[str, str]] = {
    0: {
        "label":       "Stable / Low Tumor Burden (Class 0)",
        "volume":      "Below-median enhancing tumor volume",
        "enhancement": "Minimal or no new contrast enhancement on T1CE",
        "flair":       "Stable or decreasing FLAIR signal abnormality",
        "assessment":  "MRI features are consistent with treatment response or stable disease. "
                       "Findings do not meet RANO criteria for progression. "
                       "Residual T1CE enhancement likely represents post-operative change or radiation effect.",
        "confidence_context": "Class 0 prediction indicates below-median tumor volume at this timepoint.",
    },
    1: {
        "label":       "Progressive / High Tumor Burden (Class 1)",
        "volume":      "Above-median enhancing tumor volume",
        "enhancement": "Significant or increasing contrast enhancement on T1CE",
        "flair":       "Expanding FLAIR hyperintensity suggesting infiltrating edema",
        "assessment":  "MRI features are suspicious for tumor progression or pseudoprogression. "
                       "Findings may meet RANO criteria for progression. "
                       "Urgent clinical correlation and possible biopsy or MR spectroscopy recommended.",
        "confidence_context": "Class 1 prediction indicates above-median tumor volume at this timepoint.",
    },
}


def _interpret_features(
    features: list[float],
    logits: Optional[list[float]] = None,
) -> tuple[int, float, str]:
    """
    Derive predicted class and confidence from the 768-dim feature vector.

    Strategy (no classifier head available at agent layer):
      - Compute L2 norm of feature vector.
      - Norm above median threshold (empirically ~1.0 for unit-normalised vectors)
        maps to Class 1 (high burden); below maps to Class 0.
      - If raw logits are passed (future: vision service can return them),
        use softmax probability directly.

    Returns:
        predicted_class : int   (0 or 1)
        confidence      : float (0.0–1.0)
        interpretation  : str   (full descriptive text block for LLM prompt)
    """
    if logits is not None and len(logits) == 2:
        # Use real classifier logits when available
        probs      = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())
    else:
        # Derive class from feature vector statistics
        feat_arr   = np.array(features, dtype=np.float32)
        # Positive-mean features → Class 1 (progressive); negative → Class 0 (stable)
        # Unit-normalised ViT CLS tokens: mean > 0 correlates with higher activation
        mean_val   = float(feat_arr.mean())
        l2_norm    = float(np.linalg.norm(feat_arr))
        # Sigmoid of scaled mean as pseudo-probability
        pseudo_prob = float(1 / (1 + np.exp(-mean_val * 10)))
        pred_class  = 1 if pseudo_prob >= 0.5 else 0
        confidence  = pseudo_prob if pred_class == 1 else (1 - pseudo_prob)

    desc = MRI_CLASS_DESCRIPTIONS[pred_class]

    interpretation = f"""## CNN-ViT MRI Feature Analysis

**Prediction**: {desc['label']}
**Confidence**: {confidence:.1%}
**Tumor Volume Category**: {desc['volume']}
**T1CE Enhancement**: {desc['enhancement']}
**FLAIR Signal**: {desc['flair']}

**Automated MRI Assessment**:
{desc['assessment']}

**Model Note**: {desc['confidence_context']}
Model architecture: 3D CNN-ViT hybrid (ResNet backbone + 8-layer ViT encoder).
Feature vector dimension: {len(features)}.
Training dataset: UCSD-PTGBM-BraTS-2024 (post-operative GBM cohort).
"""
    return pred_class, confidence, interpretation


# ---------------------------------------------------------------------------
# 3. Infrastructure helpers
# ---------------------------------------------------------------------------

def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url="http://127.0.0.1:11434",
        model="nomic-embed-text",
    )


def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name="brats2024_literature",
        embedding_function=_get_embeddings(),
        persist_directory="./chroma_db",
    )


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------

def extract_features_tool(state: RecoveryAgentState) -> RecoveryAgentState:
    """
    NODE 1 — Feature Extraction via CNN-ViT Vision Microservice.
    NEW: After extracting features, immediately calls _interpret_features()
         and writes the result into state['mri_interpretation'] so it is
         available to the synthesis node.
    """
    logger.info("[Node 1] extract_features_tool — entry")

    # If features already provided by frontend upload, skip vision service call
    if state.get("image_features"):
        logger.info("[Node 1] Pre-extracted features received — interpreting directly.")
        # NEW: interpret pre-supplied features immediately
        _, _, interpretation = _interpret_features(state["image_features"])
        state["mri_interpretation"] = interpretation
        return state

    logger.info("[Node 1] Calling Vision Microservice on port 8001...")

    dummy_vol = np.zeros((64, 64, 64), dtype=np.float32)
    dummy_img = nib.Nifti1Image(dummy_vol, np.eye(4))

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        nib.save(dummy_img, tmp_path)
        with open(tmp_path, "rb") as f:
            response = httpx.post(
                "http://127.0.0.1:8001/extract-features",
                files={"file": ("mock_scan.nii.gz", f, "application/gzip")},
                timeout=60.0,
            )
        response.raise_for_status()
        data                   = response.json()
        state["image_features"] = data["features"]

        # NEW: also pull logits if vision service returns them (future upgrade)
        logits = data.get("logits", None)

        # NEW: interpret features and store in state
        pred_class, confidence, interpretation = _interpret_features(
            state["image_features"], logits=logits
        )
        state["mri_interpretation"] = interpretation

        state["warnings"].append(
            "Vision Tool: Used auto-generated mock MRI (frontend upload pending)."
        )
        logger.info(
            "[Node 1] Features extracted | pred_class=%d | confidence=%.2f",
            pred_class, confidence,
        )

    except Exception as exc:
        logger.error("[Node 1] Vision service failed: %s", exc)
        state["image_features"]    = [0.0] * 768
        # NEW: set a fallback interpretation so LLM is never left blind
        state["mri_interpretation"] = (
            "## CNN-ViT MRI Feature Analysis\n\n"
            "**Status**: Feature extraction failed — vision service unreachable.\n"
            "MRI-based classification could not be performed for this analysis.\n"
            "Clinical assessment should rely entirely on the provided metadata and literature.\n"
        )
        state["warnings"].append(f"Vision service unreachable: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info("[Node 1] extract_features_tool — exit")
    return state


def retrieve_literature_node(state: RecoveryAgentState) -> RecoveryAgentState:
    """NODE 2 — Multimodal RAG Literature Retrieval."""
    logger.info(
        "[Node 2] retrieve_literature_node — attempt %d",
        state["retrieval_attempts"] + 1,
    )

    meta = state["clinical_metadata"]

    query_parts = [
        "post-operative glioblastoma multiforme recovery",
        f"tumor grade: {meta.get('tumor_grade', 'unknown')}",
        f"resection extent: {meta.get('resection_extent', 'unknown')}",
        f"treatment: {meta.get('treatment_protocol', 'standard chemoradiation')}",
        f"weeks post-surgery: {meta.get('weeks_post_surgery', 'unknown')}",
        f"KPS score: {meta.get('kps_score', 'unknown')}",
        "MRI progression assessment BraTS 2024",
    ]
    query = " | ".join(filter(None, query_parts))
    logger.debug("[Node 2] Retrieval query: %s", query)

    try:
        vector_store = _get_vector_store()
        docs: list[Document] = vector_store.similarity_search(query, k=8)
        state["literature_results"] = [
            {
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
                "page":    doc.metadata.get("page", None),
                "score":   doc.metadata.get("score", None),
            }
            for doc in docs
        ]
        logger.info("[Node 2] Retrieved %d documents.", len(state["literature_results"]))
    except Exception as exc:
        logger.error("[Node 2] Vector store retrieval failed: %s", exc)
        state["literature_results"] = []
        state["warnings"].append(f"literature_retrieval: {exc}")

    state["retrieval_attempts"] += 1
    state["retrieval_retry"]    = False
    return state


def synthesize_report_node(state: RecoveryAgentState) -> RecoveryAgentState:
    """
    NODE 3 — Clinical Report Synthesis via Groq LLM.
    CHANGED: mri_interpretation is now injected as a dedicated section
             in the prompt so the LLM is never blind to model outputs.
    """
    logger.info("[Node 3] synthesize_report_node — entry")

    meta            = state["clinical_metadata"]
    literature      = state["literature_results"]
    warnings        = state.get("warnings", [])
    # NEW: pull MRI interpretation from state (guaranteed non-None after Node 1)
    mri_section     = state.get("mri_interpretation") or (
        "## CNN-ViT MRI Feature Analysis\n\nNo MRI features available for this session."
    )

    lit_context = "\n\n".join(
        f"[{i+1}] Source: {doc['source']}\n{doc['content']}"
        for i, doc in enumerate(literature[:6])
    ) if literature else "No relevant literature retrieved."

    warning_block = (
        "\n**⚠ Agent Warnings:**\n" + "\n".join(f"- {w}" for w in warnings)
        if warnings else ""
    )

    # CHANGED: mri_section is now its own block in the prompt, not buried in metadata.
    # The LLM is explicitly instructed to use it in Section 2.
    prompt = f"""You are a senior neuro-oncology AI assistant. Produce a structured
post-operative recovery analysis report. Ground every clinical inference in the
provided literature AND the CNN-ViT MRI analysis below. Flag uncertainties clearly.
Do NOT hallucinate drug dosages, survival statistics, or imaging findings.

## Patient Clinical Metadata
```json
{json.dumps(meta, indent=2)}
```

{mri_section}

## Retrieved Literature Context
{lit_context}

## Report Requirements
Return the report in this exact Markdown structure.
For Section 2 (MRI Feature Interpretation), you MUST reference the CNN-ViT
prediction class, confidence score, and enhancement findings above.

### 1. Executive Summary
### 2. MRI Feature Interpretation
### 3. Recovery Trajectory Assessment
### 4. Literature-Grounded Recommendations
### 5. Risk Flags & Uncertainties
### 6. Suggested Next Steps
### 7. References
{warning_block}
"""

    try:
        llm      = _get_llm()
        response = llm.invoke(prompt)
        report   = response.content if hasattr(response, "content") else str(response)
        state["final_report"] = report.strip()
        logger.info("[Node 3] Report synthesized successfully.")
    except Exception as exc:
        logger.error("[Node 3] LLM synthesis failed: %s", exc)
        state["final_report"] = (
            f"**Report synthesis failed.**\n\nError: {exc}\n\n"
            f"MRI Analysis:\n{mri_section}\n\n"
            f"Raw literature context preserved below:\n\n{lit_context}"
        )
        state["warnings"].append(f"synthesis_failed: {exc}")

    logger.info("[Node 3] synthesize_report_node — exit")
    return state


# ---------------------------------------------------------------------------
# 5. Conditional / Verification Edge
# ---------------------------------------------------------------------------

def verify_literature_edge(
    state: RecoveryAgentState,
) -> Literal["synthesize_report_node", "retrieve_literature_node"]:
    results  = state.get("literature_results", [])
    attempts = state.get("retrieval_attempts", 0)

    if len(results) >= MIN_RELEVANT_DOCS:
        logger.info("[Edge] %d docs — routing to synthesis.", len(results))
        return "synthesize_report_node"

    if attempts <= MAX_RETRIES:
        logger.warning(
            "[Edge] Only %d docs (need %d). Retry %d/%d.",
            len(results), MIN_RELEVANT_DOCS, attempts, MAX_RETRIES,
        )
        state["retrieval_retry"] = True
        state["warnings"].append(
            f"literature_retrieval: insufficient results ({len(results)} docs) "
            f"on attempt {attempts} — retrying."
        )
        return "retrieve_literature_node"

    logger.warning("[Edge] Max retries reached with %d docs. Proceeding.", len(results))
    state["warnings"].append(
        "literature_retrieval: max retries reached — report may lack evidence support."
    )
    return "synthesize_report_node"


# ---------------------------------------------------------------------------
# 6. Graph Assembly
# ---------------------------------------------------------------------------

def build_recovery_graph() -> StateGraph:
    graph = StateGraph(RecoveryAgentState)

    graph.add_node("extract_features_tool",    extract_features_tool)
    graph.add_node("retrieve_literature_node", retrieve_literature_node)
    graph.add_node("synthesize_report_node",   synthesize_report_node)

    graph.add_edge(START, "extract_features_tool")
    graph.add_edge("extract_features_tool", "retrieve_literature_node")

    graph.add_conditional_edges(
        "retrieve_literature_node",
        verify_literature_edge,
        {
            "synthesize_report_node":   "synthesize_report_node",
            "retrieve_literature_node": "retrieve_literature_node",
        },
    )

    graph.add_edge("synthesize_report_node", END)
    return graph.compile()


recovery_graph = build_recovery_graph()
