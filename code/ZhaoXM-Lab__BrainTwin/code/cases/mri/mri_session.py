"""
MRI feature Q&A session — structured JSON (volume/surface) input.
Ported from MRI_parsing/data/ADNI_experiment.ipynb pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Iterator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401

from LLM_utils import CustomOpenAIProxyChat  # noqa: E402
from multiagent_system.config import Config  # noqa: E402

VOLUME_FEWSHOT = """
--------------------------------
OUTPUT FORMAT
--------------------------------

Return the selected features in JSON:

{{
  "selected_module": "...",
  "selected_level": "...",
  "selected_regions": [],
  "selected_metrics": [],
  "selected_features": [],
  "reason": "..."
}}

--------------------------------
FEW-SHOT EXAMPLES
--------------------------------

Example 1

Question:

Does the subject show structural patterns associated with memory decline?

Selected features:

{{
 "selected_module": "volume",
 "selected_level": "regional",
 "selected_regions": [
   "Left_Hippocampus",
   "Right_Hippocampus"
 ],
 "selected_metrics": [
   "Volume"
 ],
 "selected_features": [
   "volume.regional.Left_Hippocampus.Volume",
   "volume.regional.Right_Hippocampus.Volume"
 ],
 "reason": "Hippocampal volume is strongly associated with memory-related neurodegeneration."
}}

Example 2

Question:

Is there ventricular enlargement?

Selected features:

{{
 "selected_module": "volume",
 "selected_level": "regional",
 "selected_regions": [
   "Left_Lateral_Ventricle",
   "Right_Lateral_Ventricle",
   "3rd_Ventricle"
 ],
 "selected_metrics": [
   "Volume"
 ],
 "selected_features": [
   "volume.regional.Left_Lateral_Ventricle.Volume",
   "volume.regional.Right_Lateral_Ventricle.Volume",
   "volume.regional.3rd_Ventricle.Volume"
 ],
 "reason": "Ventricular enlargement is reflected by increased ventricle volumes."
}}

Example 3

Question:

What is the volume of the left hippocampus?

Selected features:

{{
 "selected_module": "volume",
 "selected_level": "regional",
 "selected_regions": [
   "Left_Hippocampus"
 ],
 "selected_metrics": [
   "Volume"
 ],
 "selected_features": [
   "volume.regional.Left_Hippocampus.Volume"
 ],
 "reason": "The question asks for hippocampal volume; select the left hippocampus volume feature."
}}

--------------------------------
IMPORTANT CONSTRAINTS
--------------------------------

Never invent feature names.

Only use structures listed in the schema.

Ensure every selected feature path follows the schema format.

Before returning the result, verify: The feature path exists in the schema.
"""

SURFACE_FEWSHOT = """
--------------------------------
OUTPUT FORMAT
--------------------------------

Return the selected features in JSON:

{{
  "selected_module": "...",
  "selected_level": "...",
  "selected_regions": [],
  "selected_metrics": [],
  "selected_features": [],
  "reason": "..."
}}

--------------------------------
FEW-SHOT EXAMPLES
--------------------------------

Example 1

Question:

Are there signs of cortical thinning?

Selected features:

{{
 "selected_module": "surface",
 "selected_level": "regional",
 "selected_regions": [
   "entorhinal",
   "precuneus",
   "superiorfrontal"
 ],
 "selected_metrics": [
   "ThickAvg"
 ],
 "selected_features": [
   "surface.regional.entorhinal.left_hemisphere.ThickAvg",
   "surface.regional.entorhinal.right_hemisphere.ThickAvg",
   "surface.regional.precuneus.left_hemisphere.ThickAvg",
   "surface.regional.precuneus.right_hemisphere.ThickAvg"
 ],
 "reason": "Average cortical thickness is the primary indicator of cortical thinning."
}}

Example 2

Question:

What is the average thickness of the entorhinal cortex?

Selected features:

{{
 "selected_module": "surface",
 "selected_level": "regional",
 "selected_regions": [
   "entorhinal"
 ],
 "selected_metrics": [
   "ThickAvg"
 ],
 "selected_features": [
   "surface.regional.entorhinal.left_hemisphere.ThickAvg",
   "surface.regional.entorhinal.right_hemisphere.ThickAvg"
 ],
 "reason": "Entorhinal cortical thickness is requested; include both hemispheres."
}}

--------------------------------
IMPORTANT CONSTRAINTS
--------------------------------

Never invent feature names.

Only use structures listed in the schema.

Ensure every selected feature path follows the schema format.

Before returning the result, verify: The feature path exists in the schema.
"""

REASONING_SYSTEM_CLINICAL = """
You are a neuroimaging reasoning assistant.

You are given MRI-derived features with numerical values, pre-selected for the question.

Rules:
- Use ONLY the provided features and values. Do NOT invent regions or measurements.
- Base conclusions explicitly on the given evidence.
- If evidence is insufficient, state so clearly.
"""

REASONING_SYSTEM_METRIC = """
You are a neuroimaging assistant answering factual questions about MRI measurements for one subject.

Rules:
- Report the exact values from the provided evidence.
- Include units: volume in mm³, cortical thickness in mm, surface area in mm².
- Name each brain region and metric clearly.
- Answer the question directly; do NOT discuss disease or abnormalities unless asked.
- If the requested metric is missing from the evidence, say so clearly.
"""

REASONING_SYSTEM_KNOWLEDGE = """
You are a neuroscience assistant answering questions about brain anatomy and function.

Rules:
- Answer the user's substantive question using established neuroscience knowledge.
- MRI feature values, when provided, describe structure only — they do not encode functional role.
- It is fine to explain function/anatomy from general knowledge even when MRI data lacks that information.
- If structural values for the region are in the evidence, add a brief "In this sample" note with cited values.
- Do NOT refuse the question just because function is not in the MRI JSON.
- Do NOT discuss abnormalities, disease risk, or clinical interpretation unless the question explicitly asks for it.
"""

REASONING_USER_CLINICAL = """
Question:
{question}

Available evidence:
{evidences}

Task:
1. Identify which brain structures show abnormalities or notable patterns relevant to the question.
2. Explain how the provided features support or do not support the question.
3. State your final conclusion clearly.
4. Explicitly cite feature names and values when reasoning.

{reasoning_format_instructions}
"""

REASONING_USER_METRIC = """
Question:
{question}

Available evidence:
{evidences}

Task:
1. Answer the question by reporting the requested measurements from the evidence.
2. Cite each feature path and its numeric value with the correct unit.
3. If multiple related metrics are present, organize them clearly (e.g. left/right, volume vs thickness).
4. Keep the answer concise and factual.

{reasoning_format_instructions}
"""

REASONING_USER_KNOWLEDGE = """
Question:
{question}

Available evidence (structural MRI values for this subject, if any):
{evidences}

Task:
1. Answer the user's question directly (function, anatomy, role, location, etc.).
2. Use general neuroscience knowledge where needed.
3. If relevant structural values are available, add them in a short "In this sample" section.
4. Do not claim the MRI data contains functional information it does not contain.
5. Do not add clinical interpretation or abnormality assessment unless asked.

{reasoning_format_instructions}
"""

INTENT_SYSTEM = """
Classify the user's question about brain MRI data into exactly one intent:

- metric: asks for a specific measurement, size, volume, thickness, area, or numeric value
- clinical: asks about disease, abnormality, atrophy, decline, patterns, risk, or yes/no diagnostic inference
- knowledge: asks about function, role, anatomy, location, or what a region does
- comparison: asks to compare regions or hemispheres

Return JSON only.
"""

INTENT_USER = """Question:
{question}

Classify the intent."""


class FeatureSelection(BaseModel):
    selected_module: str = Field(description="volume or surface")
    selected_level: str = Field(description="global or regional")
    selected_regions: list[str] = Field(default_factory=list)
    selected_metrics: list[str] = Field(default_factory=list)
    selected_features: list[str] = Field(default_factory=list)
    reason: str = ""


class ReasoningResult(BaseModel):
    answer: str = Field(description="Direct answer to the user's question")
    reasoning: str = Field(default="", description="Supporting explanation")
    cited_features: list[str] = Field(default_factory=list)
    abnormalities: list[str] = Field(default_factory=list)
    conclusion: str = Field(default="", description="Final conclusion for clinical questions")
    prediction: str = Field(default="", description="yes or no for clinical questions")


class QuestionIntent(BaseModel):
    intent: str = Field(
        description="One of: metric, clinical, knowledge, comparison"
    )
    reason: str = ""


def validate_structured_sample(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Feature JSON must be an object.")
    if "volume" not in data or "surface" not in data:
        raise ValueError(
            "Expected structured features with top-level 'volume' and 'surface' keys "
            "(see MRI_parsing/data/ADNI_data/sample_structural.json)."
        )
    for module in ("volume", "surface"):
        if not isinstance(data[module], dict):
            raise ValueError(f"'{module}' must be an object.")
    return data


def sample_summary(data: dict[str, Any]) -> dict[str, Any]:
    volume = data.get("volume", {})
    surface = data.get("surface", {})
    return {
        "volume_global": len(volume.get("global", {})),
        "volume_regional": len(volume.get("regional", {})),
        "surface_global": len(surface.get("global", {})),
        "surface_regional": len(surface.get("regional", {})),
    }


def construct_evidences(selected_features: list[str], sample_info: dict[str, Any]) -> str:
    proofs = "For the given question, the following evidences are provided:\n"
    for feature in selected_features:
        try:
            info: Any = sample_info
            for step in feature.split("."):
                info = info[step]
            if isinstance(info, dict):
                parts = []
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        unit = ""
                        if key == "Volume":
                            unit = " mm³"
                        elif key in {"ThickAvg", "ThickStd", "MeanCurv", "GausCurv"}:
                            unit = " mm"
                        elif key == "SurfArea":
                            unit = " mm²"
                        elif key == "GrayVol":
                            unit = " mm³"
                        parts.append(f"{key}={value}{unit}")
                if parts:
                    proofs += f"{feature}: {', '.join(parts)}\n"
                else:
                    proofs += f"{feature}: {info}\n"
            else:
                unit = ""
                if feature.endswith(".Volume"):
                    unit = " mm³"
                elif feature.endswith(".ThickAvg"):
                    unit = " mm"
                elif feature.endswith(".SurfArea"):
                    unit = " mm²"
                proofs += f"{feature}: {info}{unit}\n"
        except (KeyError, TypeError):
            continue
    return proofs


def _build_volume_schema(regional_keys: list[str]) -> str:
    return f"""
You are an assistant designed to analyze structural MRI data stored in a hierarchical JSON format.

Your task has TWO stages:
1. Understand the MRI feature schema.
2. Select relevant MRI features to answer a user question.

You must NOT invent features that do not exist in the schema.

--------------------------------
STRUCTURAL MRI DATA SCHEMA
--------------------------------

================================
1. VOLUME MODULE
================================

Volume features are derived from voxel-based segmentation of anatomical structures.

Structure:

volume
 ├── global
 └── regional

--------------------------------
volume.global
--------------------------------

Whole-brain volumetric summary measures.

Example features:

- BrainSegVol
- BrainSegVolNotVent
- CortexVol
- SupraTentorialVol
- eTIV
- TotalGrayVol
- CerebralWhiteMatterVol

Units:

- volume → mm³
- ratios → unitless

--------------------------------
volume.regional
--------------------------------

Segmentation-based anatomical structures.

AVAILABLE STRUCTURES: {", ".join(regional_keys)}

Each structure contains metrics: NVoxels, normMean, normStdDev, normMin, normMax, normRange, Volume

Example feature paths:

volume.regional.Left_Hippocampus.Volume
volume.regional.Right_Amygdala.Volume
"""


def _build_surface_schema(regional_keys: list[str]) -> str:
    return f"""

================================
2. SURFACE MODULE
================================

Surface features describe cortical morphology based on reconstructed cortical surfaces.

Structure:

surface
 ├── global
 └── regional

--------------------------------
surface.regional
--------------------------------

Cortical regions defined by a cortical parcellation atlas.

AVAILABLE STRUCTURES: {", ".join(regional_keys)}

Feature path format:

surface.regional.<region>.<hemisphere>.<metric>

Each region contains left_hemisphere and right_hemisphere.

Metrics: NumVert, SurfArea, GrayVol, ThickAvg, ThickStd, MeanCurv, GausCurv, FoldInd, CurvInd

Example feature paths:

surface.regional.entorhinal.left_hemisphere.ThickAvg
surface.regional.precuneus.right_hemisphere.SurfArea
"""


class MRIParsingSession:
    def __init__(self, sample_info: dict[str, Any], config: Config | None = None):
        self.sample_info = validate_structured_sample(sample_info)
        self.config = config or Config()
        self.llm = CustomOpenAIProxyChat(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model_id,
            streaming=False,
        )
        self._init_chains()

    def _init_chains(self) -> None:
        volume_regions = list(self.sample_info["volume"].get("regional", {}).keys())
        surface_regions = list(self.sample_info["surface"].get("regional", {}).keys())

        volume_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _build_volume_schema(volume_regions) + VOLUME_FEWSHOT),
                (
                    "human",
                    """Here is the user question:
{question}
Please select the most relevant MRI features and return JSON only.
""",
                ),
            ]
        )
        surface_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _build_surface_schema(surface_regions) + SURFACE_FEWSHOT),
                (
                    "human",
                    """Here is the user question:
{question}
Please select the most relevant MRI features and return JSON only.
""",
                ),
            ]
        )

        parser = PydanticOutputParser(pydantic_object=FeatureSelection)
        self.volume_chain = volume_prompt | self.llm | parser
        self.surface_chain = surface_prompt | self.llm | parser

        intent_parser = PydanticOutputParser(pydantic_object=QuestionIntent)
        intent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTENT_SYSTEM),
                ("human", INTENT_USER),
            ]
        )
        self.intent_chain = intent_prompt | self.llm | intent_parser

        reasoning_parser = PydanticOutputParser(pydantic_object=ReasoningResult)
        self.reasoning_format_instructions = reasoning_parser.get_format_instructions()
        self.reasoning_chains = {
            "metric": self._build_reasoning_chain(
                REASONING_SYSTEM_METRIC,
                REASONING_USER_METRIC,
                reasoning_parser,
            ),
            "knowledge": self._build_reasoning_chain(
                REASONING_SYSTEM_KNOWLEDGE,
                REASONING_USER_KNOWLEDGE,
                reasoning_parser,
            ),
            "comparison": self._build_reasoning_chain(
                REASONING_SYSTEM_METRIC,
                REASONING_USER_METRIC,
                reasoning_parser,
            ),
            "clinical": self._build_reasoning_chain(
                REASONING_SYSTEM_CLINICAL,
                REASONING_USER_CLINICAL,
                reasoning_parser,
            ),
        }

    def _build_reasoning_chain(
        self,
        system_prompt: str,
        user_prompt: str,
        parser: PydanticOutputParser,
    ):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        return prompt | self.llm | parser

    def _classify_intent(self, question: str) -> str:
        try:
            result = self.intent_chain.invoke({"question": question})
            intent = (result.intent or "clinical").strip().lower()
            if intent not in self.reasoning_chains:
                return "clinical"
            return intent
        except Exception:
            return "clinical"

    def ask(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        volume_result = self.volume_chain.invoke({"question": question})
        selected_features = list(volume_result.selected_features)

        surface_result = self.surface_chain.invoke({"question": question})
        selected_features.extend(surface_result.selected_features)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_features: list[str] = []
        for feat in selected_features:
            if feat not in seen:
                seen.add(feat)
                unique_features.append(feat)

        return self._attach_visualization(
            self._finalize_answer(question, unique_features, volume_result.reason, surface_result.reason)
        )

    def _finalize_answer(
        self,
        question: str,
        unique_features: list[str],
        volume_reason: str,
        surface_reason: str,
        intent: str | None = None,
    ) -> dict[str, Any]:
        intent = intent or self._classify_intent(question)
        evidences = construct_evidences(unique_features, self.sample_info)
        reasoning = self.reasoning_chains[intent].invoke(
            {
                "question": question,
                "evidences": evidences,
                "reasoning_format_instructions": self.reasoning_format_instructions,
            }
        )

        answer_parts: list[str] = []
        primary = (reasoning.answer or reasoning.conclusion or "").strip()
        if primary:
            answer_parts.append(primary)
        if reasoning.reasoning.strip() and reasoning.reasoning.strip() != primary:
            answer_parts.append(reasoning.reasoning.strip())

        return {
            "answer": "\n\n".join(part for part in answer_parts if part),
            "intent": intent,
            "features": unique_features,
            "evidences": evidences.strip(),
            "abnormalities": reasoning.abnormalities,
            "conclusion": reasoning.conclusion or reasoning.answer,
            "reasoning": reasoning.reasoning,
            "cited_features": reasoning.cited_features,
            "volume_selection_reason": volume_reason,
            "surface_selection_reason": surface_reason,
            "visualization": {"regions": [], "images": []},
        }

    def _attach_visualization(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from mri_visualization import build_visualization

            payload["visualization"] = build_visualization(
                payload["features"],
                self.sample_info,
            )
        except Exception:
            payload["visualization"] = {
                "regions": [],
                "cortical_regions": [],
                "subcortical_regions": [],
                "images": [],
                "unmapped_features": [],
            }
        return payload

    def iter_ask_events(self, question: str) -> Iterator[dict[str, Any]]:
        yield {"type": "status", "text": "Selecting relevant volume features"}
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        volume_result = self.volume_chain.invoke({"question": question})
        selected_features = list(volume_result.selected_features)
        if volume_result.selected_features:
            preview = ", ".join(volume_result.selected_features[:4])
            suffix = (
                f" (+{len(volume_result.selected_features) - 4} more)"
                if len(volume_result.selected_features) > 4
                else ""
            )
            yield {"type": "status", "text": f"Volume features: {preview}{suffix}"}

        yield {"type": "status", "text": "Selecting relevant surface features"}
        surface_result = self.surface_chain.invoke({"question": question})
        selected_features.extend(surface_result.selected_features)

        seen: set[str] = set()
        unique_features: list[str] = []
        for feat in selected_features:
            if feat not in seen:
                seen.add(feat)
                unique_features.append(feat)

        if surface_result.selected_features:
            preview = ", ".join(surface_result.selected_features[:4])
            suffix = (
                f" (+{len(surface_result.selected_features) - 4} more)"
                if len(surface_result.selected_features) > 4
                else ""
            )
            yield {"type": "status", "text": f"Surface features: {preview}{suffix}"}

        yield {"type": "status", "text": "Understanding question type"}
        intent = self._classify_intent(question)
        intent_labels = {
            "metric": "measurement lookup",
            "clinical": "clinical inference",
            "knowledge": "anatomy / function",
            "comparison": "region comparison",
        }
        yield {
            "type": "status",
            "text": f"Question type: {intent_labels.get(intent, intent)}",
        }

        yield {"type": "status", "text": "Reasoning over selected evidence"}

        payload = self._finalize_answer(
            question,
            unique_features,
            volume_result.reason,
            surface_result.reason,
            intent=intent,
        )

        if unique_features:
            yield {"type": "status", "text": "Rendering highlighted brain regions"}

        payload = self._attach_visualization(payload)

        yield {
            "type": "done",
            **payload,
        }


def load_sample_from_json(text: str) -> dict[str, Any]:
    data = json.loads(text)
    return validate_structured_sample(data)
