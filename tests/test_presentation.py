"""Tests for presentation draft generation and PowerPoint export."""

from __future__ import annotations

from io import BytesIO
import unittest
import zipfile

import pandas as pd

from src.presentation import (
    build_presentation_deck_draft,
    normalize_presentation_input,
    presentation_deck_to_dict,
    presentation_deck_to_pptx_bytes,
    validate_presentation_deck,
)


class PresentationDraftTests(unittest.TestCase):
    def test_normalize_presentation_input_rejects_empty_source(self) -> None:
        with self.assertRaises(ValueError):
            normalize_presentation_input({"sourceType": "topic", "topic": "", "slideCount": 8})

    def test_deterministic_deck_generation_returns_requested_slide_count(self) -> None:
        result = build_presentation_deck_draft(
            {
                "sourceType": "topic",
                "topic": "PJM battery revenue strategy",
                "purpose": "executive_summary",
                "audience": "executives",
                "style": "consulting_style",
                "slideCount": 8,
                "includeSpeakerNotes": True,
                "includeVisualSuggestions": True,
            },
            zone_df=_zone_data(),
            monthly_df=_monthly_data(),
            prefer_ai=False,
        )

        self.assertEqual(result.generation_mode, "Deterministic local draft")
        self.assertEqual(len(result.deck.slides), 8)
        self.assertEqual(result.deck.slides[0].type, "title")
        self.assertEqual(result.deck.slides[-1].type, "closing")
        self.assertIn("Zone-Level Battery Revenue", result.deck.deckTitle)
        self.assertTrue(result.deck.slides[1].speakerNotes)

    def test_validate_presentation_deck_repairs_ids_and_slide_numbers(self) -> None:
        deck = validate_presentation_deck(
            {
                "deckTitle": "Draft",
                "audience": "Executives",
                "purpose": "Executive summary",
                "style": "Consulting-style",
                "slides": [
                    {"id": "same", "slideNumber": 9, "type": "title", "title": "One"},
                    {"id": "same", "slideNumber": 9, "type": "insight", "title": "Two"},
                    {"id": "", "slideNumber": 9, "type": "closing", "title": "Three"},
                ],
            }
        )

        self.assertEqual([slide.slideNumber for slide in deck.slides], [1, 2, 3])
        self.assertEqual(len({slide.id for slide in deck.slides}), 3)
        self.assertEqual(deck.audience, "executives")
        self.assertEqual(deck.purpose, "executive_summary")
        self.assertEqual(deck.style, "consulting_style")

    def test_presentation_deck_to_dict_is_json_ready(self) -> None:
        result = build_presentation_deck_draft(
            {
                "sourceType": "source_material",
                "sourceText": "Use Flexworks outputs to explain zonal revenue variation.",
                "slideCount": 5,
            },
            prefer_ai=False,
        )

        payload = presentation_deck_to_dict(result.deck)

        self.assertIn("deckTitle", payload)
        self.assertIn("slides", payload)
        self.assertEqual(len(payload["slides"]), 5)

    def test_pptx_export_returns_powerpoint_zip(self) -> None:
        result = build_presentation_deck_draft(
            {
                "sourceType": "topic",
                "topic": "Flexworks market intelligence",
                "slideCount": 5,
                "includeSpeakerNotes": False,
                "includeVisualSuggestions": False,
            },
            zone_df=_zone_data(),
            prefer_ai=False,
        )

        payload = presentation_deck_to_pptx_bytes(result.deck)

        self.assertTrue(payload.startswith(b"PK"))
        with zipfile.ZipFile(BytesIO(payload), mode="r") as archive:
            names = archive.namelist()
            self.assertIn("ppt/presentation.xml", names)
            self.assertIn("ppt/slides/slide1.xml", names)


def _zone_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Zone": ["BGE", "DPL", "PECO", "DOM", "JCPL"],
            "Revenue_per_kW": [149.61, 147.68, 141.2, 148.43, 132.1],
        }
    )


def _monthly_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Zone": ["BGE", "DPL", "BGE"],
            "Month": ["2024-01", "2024-02", "2024-03"],
            "Revenue": [100, 60, 140],
        }
    )


if __name__ == "__main__":
    unittest.main()
