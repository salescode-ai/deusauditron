import re
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher

from deusauditron.app_logging.logger import logger
from deusauditron.llm_abstraction.llm_helper import LLMInvoker
from deusauditron.schemas.shared_models.models import Message, MessageRole
from deusauditron.schemas.autogen.references.voice_evaluation_result_schema import VoiceComparisonResult
from deusauditron.schemas.autogen.references.voice_evaluation_result_schema import VoiceEvaluationDetails, VoiceEvalCategory


class BaseVoiceEvaluationStrategy:
    """Base class for voice evaluation strategies"""
    
    def __init__(self, api_keys: Dict[str, str], evaluation_model: Optional[str] = None):
        self.api_keys = api_keys
        self.evaluation_model = evaluation_model

    async def evaluate(self, generated_text: str, actual_text: str) -> VoiceComparisonResult:
        """Evaluate generated text against actual text"""
        raise NotImplementedError
    
    def _get_evaluation_model(self) -> str:
        """Get the evaluation model from payload, config, or fallback"""
        from deusauditron.config import get_config
        config = get_config()
        
        # Priority: payload > config > fallback
        if self.evaluation_model:
            return self.evaluation_model
        
        return config.voice_evaluation.evaluation_model

    def _normalize_numbers(self, text: str) -> str:
        """Normalize common number words/phrases to digits to reduce numeric mismatches."""
        if not text:
            return ""
        lowered = text.lower()
        
        # Phrase-first replacements (longer phrases first) - comprehensive list
        mapping = {
            # Hundreds
            "nine hundred and ninety nine": "999",
            "nine hundred ninety nine": "999",
            "seven hundred and fifty": "750",
            "seven hundred fifty": "750",
            "five hundred": "500",
            "two hundred and fifty": "250",
            "two hundred fifty": "250",
            "one hundred and twenty five": "125",
            "one hundred twenty five": "125",
            "one hundred": "100",
            # Common commerce numbers
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "sixty": "60",
            "seventy": "70",
            "eighty": "80",
            "ninety": "90",
        }
        for phrase in sorted(mapping.keys(), key=len, reverse=True):
            lowered = re.sub(rf"\b{re.escape(phrase)}\b", mapping[phrase], lowered)
        
        # Singles 0-19
        singles = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19",
        }
        for w, d in singles.items():
            lowered = re.sub(rf"\b{w}\b", d, lowered)
        
        # Normalize percent: "five percent" -> "5 percent", "5%" -> "5 percent"
        lowered = re.sub(r"\b(\d+)\s*%", r"\1 percent", lowered)
        lowered = re.sub(r"\bpercent\b", "percent", lowered)
        
        # Normalize unit formats (ml, liter, etc.)
        lowered = re.sub(r"\b(\d+)\s*[- ]?ml\b", r"\1 ml", lowered, flags=re.IGNORECASE)
        lowered = re.sub(r"\b(\d+)\s*[- ]?liter(s)?\b", r"\1 liter", lowered, flags=re.IGNORECASE)
        
        return lowered

    def _clean_transcript_for_stt(self, text: str) -> str:
        """Aggressively clean transcript to keep ONLY text and numbers for fair comparison."""
        if not text:
            return ""
        
        cleaned = text
        
        # Step 1: Remove JSON-like artifacts and special structures
        # Remove dict/JSON patterns like {'type': 'text', 'text': '...'}
        cleaned = re.sub(r"\{[^\}]*\}", " ", cleaned)
        
        # Step 2: Remove angle-bracket placeholders like < ... >, <...>
        cleaned = re.sub(r"<[^>]*>", " ", cleaned)
        
        # Step 3: Remove parenthetical annotations e.g., (coughs), (phone ringing)
        cleaned = re.sub(r"\([^\)]*\)", " ", cleaned)
        
        # Step 4: Remove bracketed annotations e.g., [noise]
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
        
        # Step 5: Remove common filler words
        cleaned = re.sub(r"\b(uh|uhh|um|umm|uh-huh|hmm|er|ah|mm|mhm)\b", " ", cleaned, flags=re.IGNORECASE)
        
        # Step 6: Normalize numbers BEFORE removing punctuation
        cleaned = self._normalize_numbers(cleaned)
        
        # Step 7: Convert to lowercase
        cleaned = cleaned.lower()
        
        # Step 8: AGGRESSIVE CLEANUP - Remove ALL punctuation and special characters
        # Keep ONLY: letters, digits, and spaces
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        
        # Step 9: Collapse multiple spaces into single space
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        return cleaned.strip()

    def _split_into_lines(self, text: str) -> List[str]:
        """Simple sentence/line splitter for alignment display."""
        if not text:
            return []
        # Insert a space after sentence punctuation if missing (e.g., "fanta?and" -> "fanta? and")
        prepped = re.sub(r"([\.!?])(?!\s)", r"\\1 ", text.strip())
        prepped = re.sub(r"\s+", " ", prepped).strip()
        # Split on sentence enders and preserve order
        parts = re.split(r"(?<=[\.!?])\s+", prepped)
        # Fallback if no sentence enders
        if len(parts) == 1:
            parts = [p.strip() for p in text.split("\n") if p.strip()]
        # Strip leading leftover punctuation (commas/semicolons/colons)
        normalized = []
        for p in parts:
            line = re.sub(r"^[\s,;:]+", "", p).strip()
            # Remove leading discourse markers like "yeah,", "okay,", "well," at line start
            line = re.sub(r"^(yeah|ok|okay|well|so|um|uh|hmm)\s*,?\s+", "", line, flags=re.IGNORECASE)
            # Drop trivial lines (single punctuation or very short non-alnum content)
            if line and re.search(r"[a-zA-Z0-9]", line) and len(line) > 2 and line != ".":
                normalized.append(line)
        return normalized

    def _extract_json_array(self, text: str) -> List[Dict[str, str]]:
        import json
        try:
            return json.loads(text)
        except Exception:
            # Try to find first [...] block
            import re
            m = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return []
            return []

class STTEvaluationStrategy(BaseVoiceEvaluationStrategy):
    """Speech-to-Text evaluation strategy"""
    
    async def evaluate(self, generated_text: str, actual_text: str) -> VoiceComparisonResult:
        """LLM-only STT evaluation: clean, call LLM with response_schema, return simple array of mistakes."""
        try:
            logger.info("Starting STT evaluation (LLM via response_schema)")
            from deusauditron.config import get_config
            config = get_config()
            model_name = self._get_evaluation_model()
            temperature = config.voice_evaluation.temperature
            logger.info(f"Using model: {model_name} with temperature: {temperature}")

            # Clean both sides
            generated_clean = self._clean_transcript_for_stt(generated_text)
            actual_clean = self._clean_transcript_for_stt(actual_text)

            # Strict response schema (object with items array to satisfy structured-output providers)
            response_schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expected_output": {"type": "string"},
                                "actual_output_by_our_stt": {"type": "string"}
                            },
                            "required": ["expected_output", "actual_output_by_our_stt"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            }

            prompt = (
                "Compare these two texts and identify SPECIFIC word-level or phrase-level differences.\n"
                "For each difference, extract ONLY the mismatched portion (not the entire text).\n"
                "Ignore punctuation, capitalization, and minor formatting differences.\n"
                "Focus on actual content differences like:\n"
                "- Word substitutions (e.g., 'thums up' vs 'thumbs up')\n"
                "- Missing words or phrases\n"
                "- Extra words or phrases\n"
                "- Significant content changes\n\n"
                "Return a JSON array with each individual difference as a separate item.\n"
                "Schema: { items: [ { expected_output: \"<specific phrase from expected>\", actual_output_by_our_stt: \"<corresponding phrase from actual>\" } ] }\n\n"
                f"Expected text:\n{generated_clean}\n\n"
                f"Actual text (from our STT):\n{actual_clean}\n\n"
                "IMPORTANT: Each item should contain ONLY the specific mismatched portion, not the entire text."
            )
            messages = [Message(content=prompt, role=MessageRole.USER)]

            llm_response = await LLMInvoker.get_instance().invoke_non_streaming(
                llm_request=messages,
                model_name=model_name,
                temperature=temperature,
                reasoning=False,
                response_schema=response_schema,
            )

            issues: List[Dict[str, str]] = []
            if not llm_response.error and llm_response.content:
                # Accept either a raw JSON array or an object with items[]
                try:
                    import json as _json
                    parsed = _json.loads(llm_response.content)
                    if isinstance(parsed, list):
                        issues = parsed
                    elif isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                        issues = parsed.get("items", [])
                except Exception:
                    # Fallback to array extraction if provider wrapped content oddly
                    issues = self._extract_json_array(llm_response.content)

            if not isinstance(issues, list):
                issues = []

            # Minimal scoring derived from issue count
            # Accuracy: 100 - (issues / max(1, reference sentences)) * 100
            num_ref_lines = max(1, len(self._split_into_lines(actual_clean)))
            num_issues = len(issues)
            accuracy_score = max(0.0, 100.0 - (num_issues / num_ref_lines) * 100.0)

            token_mismatches: List[Dict[str, str]] = []
            for it in issues:
                if isinstance(it, dict):
                    exp = it.get("expected_output")
                    got = it.get("actual_output_by_our_stt")
                    if exp is not None and got is not None:
                        token_mismatches.append({"expected": str(exp), "got": str(got)})

            return VoiceComparisonResult(
                accuracy_score=accuracy_score,
                token_mismatches=token_mismatches,
            )

        except Exception as e:
            logger.error(f"STT evaluation failed: {e}")
            return VoiceComparisonResult(
                accuracy_score=0.0,
                token_mismatches=[],
            )



class TTSEvaluationStrategy(BaseVoiceEvaluationStrategy):
    """Text-to-Speech evaluation strategy"""
    
    async def evaluate(self, generated_text: str, actual_text: str) -> VoiceComparisonResult:
        """LLM-only TTS evaluation mirroring STT: clean, call LLM with response_schema, return token mismatches + accuracy."""
        try:
            logger.info("Starting TTS evaluation (LLM via response_schema)")
            from deusauditron.config import get_config
            config = get_config()
            model_name = self._get_evaluation_model()
            temperature = config.voice_evaluation.temperature
            logger.info(f"Using model: {model_name} with temperature: {temperature}")

            # Clean both sides (NOTE: cleaned_actual_text is the source of truth)
            generated_clean = self._clean_transcript_for_stt(generated_text)
            actual_clean = self._clean_transcript_for_stt(actual_text)

            # Strict response schema, identical structure to STT
            response_schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expected_output": {"type": "string"},
                                "actual_output_by_our_tts": {"type": "string"}
                            },
                            "required": ["expected_output", "actual_output_by_our_tts"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            }

            prompt = (
                "Compare these two texts and identify SPECIFIC word-level or phrase-level differences.\n"
                "For each difference, extract ONLY the mismatched portion (not the entire text).\n"
                "Ignore punctuation, capitalization, and minor formatting differences.\n"
                "Focus on actual content differences like:\n"
                "- Word substitutions (e.g., 'coca cola' vs 'coca-cola')\n"
                "- Missing words or phrases\n"
                "- Extra words or phrases\n"
                "- Significant content changes\n\n"
                "Return a JSON array with each individual difference as a separate item.\n"
                "Schema: { items: [ { expected_output: \"<specific phrase from expected>\", actual_output_by_our_tts: \"<corresponding phrase from actual>\" } ] }\n\n"
                f"Expected text:\n{actual_clean}\n\n"
                f"Actual text (from our TTS):\n{generated_clean}\n\n"
                "IMPORTANT: Each item should contain ONLY the specific mismatched portion, not the entire text."
            )
            messages = [Message(content=prompt, role=MessageRole.USER)]

            llm_response = await LLMInvoker.get_instance().invoke_non_streaming(
                llm_request=messages,
                model_name=model_name,
                temperature=temperature,
                reasoning=False,
                response_schema=response_schema,
            )

            issues: List[Dict[str, str]] = []
            if not llm_response.error and llm_response.content:
                try:
                    import json as _json
                    parsed = _json.loads(llm_response.content)
                    if isinstance(parsed, list):
                        issues = parsed
                    elif isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                        issues = parsed.get("items", [])
                except Exception:
                    # Fallback to array extraction
                    issues = self._extract_json_array(llm_response.content)

            if not isinstance(issues, list):
                issues = []

            # Minimal scoring derived from issue count (use actual_clean as reference)
            num_ref_lines = max(1, len(self._split_into_lines(actual_clean)))
            num_issues = len(issues)
            accuracy_score = max(0.0, 100.0 - (num_issues / num_ref_lines) * 100.0)

            token_mismatches: List[Dict[str, str]] = []
            for it in issues:
                if isinstance(it, dict):
                    exp = it.get("expected_output")
                    got = it.get("actual_output_by_our_tts") or it.get("actual_output_by_our_stt")
                    if exp is not None and got is not None:
                        token_mismatches.append({"expected": str(exp), "got": str(got)})

            return VoiceComparisonResult(
                accuracy_score=accuracy_score,
                token_mismatches=token_mismatches,
            )

        except Exception as e:
            logger.error(f"TTS evaluation failed: {e}")
            return VoiceComparisonResult(
                accuracy_score=0.0,
                token_mismatches=[],
            )
