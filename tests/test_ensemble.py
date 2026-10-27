"""Offline checks for the ensemble wiring. No network, no LLM keys: models are fakes."""

import asyncio
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("METACULUS_TOKEN", "offline-test")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from forecasting_tools import BinaryPrediction, BinaryQuestion, GeneralLlm  # noqa: E402


class FakeLlm(GeneralLlm):
    def __init__(self, model: str, reply: str):
        super().__init__(model=model)
        self.reply = reply
        self.calls = 0

    async def invoke(self, prompt, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)  # yield so prediction slots really interleave
        return self.reply


async def fake_structure_output(text, output_type, *args, **kwargs):
    pct = float(re.search(r"Probability:\s*([\d.]+)%", text).group(1))
    return BinaryPrediction(prediction_in_decimal=pct / 100)


def question(n: int = 1) -> BinaryQuestion:
    return BinaryQuestion(question_text=f"Will test event {n} happen?", page_url=f"https://example.test/q{n}")


def make_bot(forecasters, research_reports=1):
    helper = FakeLlm("fake/helper", "Research notes. Probability: 50%")
    return main.EnsembleBot(
        research_reports_per_question=research_reports,
        predictions_per_research_report=max(1, len(forecasters)),
        publish_reports_to_metaculus=False,
        llms={"default": helper, "researcher": helper, "summarizer": helper, "parser": helper},
        forecasters=forecasters,
    )


@mock.patch.object(main, "structure_output", fake_structure_output)
class EnsembleTest(unittest.TestCase):
    def test_each_model_forecasts_once_and_median_wins(self):
        models = [FakeLlm(f"fake/m{i}", f"Reasoning {i}. Probability: {p}%") for i, p in enumerate([10, 30, 80])]
        report = asyncio.run(make_bot(models).forecast_question(question()))
        self.assertEqual([m.calls for m in models], [1, 1, 1])
        self.assertAlmostEqual(report.prediction, 0.30)
        for m in models:
            self.assertIn(f"*Model: {m.model}*", report.explanation)

    def test_slots_stay_per_question_when_questions_run_concurrently(self):
        models = [FakeLlm(f"fake/m{i}", f"Probability: {20 + 10 * i}%") for i in range(3)]
        bot = make_bot(models, research_reports=2)
        reports = asyncio.run(bot.forecast_questions([question(1), question(2)]))
        self.assertEqual(len(reports), 2)
        self.assertEqual([m.calls for m in models], [4, 4, 4])  # 2 questions x 2 reports, one call each

    def test_without_forecasters_falls_back_to_default_llm(self):
        bot = make_bot([])
        report = asyncio.run(bot.forecast_question(question()))
        self.assertAlmostEqual(report.prediction, 0.50)
        self.assertIn("*Model: fake/helper*", report.explanation)

    def test_question_cap_skips_forecasted_then_truncates(self):
        model = FakeLlm("fake/m0", "Probability: 40%")
        bot = make_bot([model])
        bot.skip_previously_forecasted_questions = True
        bot.max_questions_per_run = 2
        done = question(1)
        done.already_forecasted = True
        reports = asyncio.run(bot.forecast_questions([done, question(2), question(3), question(4)]))
        self.assertEqual([r.question.page_url for r in reports], ["https://example.test/q2", "https://example.test/q3"])


BLANK_ENV = {"OPENROUTER_API_KEY": "", "BOT_TIER": "", "BOT_FORECASTERS": "", "BOT_RESEARCHER": "", "BOT_PARSER": ""}


class BuildEnsembleTest(unittest.TestCase):
    def build(self, **env):
        with mock.patch.dict(os.environ, {**BLANK_ENV, **env}):
            return main.build_ensemble()

    def test_openrouter_key_defaults_to_free_tier(self):
        ens = self.build(OPENROUTER_API_KEY="sk-or-test")
        self.assertTrue(ens.free_tier)
        self.assertEqual([f.model for f in ens.forecasters], main.OPENROUTER_FREE_FORECASTERS)
        self.assertTrue(all(f.model.endswith(":free") for f in ens.forecasters))
        self.assertEqual(ens.llms["researcher"], "no_research")
        self.assertEqual(ens.llms["parser"].model, main.OPENROUTER_FREE_PARSER)

    def test_paid_tier_uses_full_ensemble(self):
        ens = self.build(OPENROUTER_API_KEY="sk-or-test", BOT_TIER="paid")
        self.assertFalse(ens.free_tier)
        self.assertEqual([f.model for f in ens.forecasters], main.OPENROUTER_FORECASTERS)
        self.assertEqual(ens.llms["researcher"].model, main.OPENROUTER_RESEARCHER)
        self.assertEqual(ens.llms["parser"].model, main.OPENROUTER_PARSER)
        self.assertIs(ens.llms["default"], ens.forecasters[0])

    def test_env_overrides_and_asknews_passthrough(self):
        ens = self.build(
            OPENROUTER_API_KEY="REPLACE_ME",
            BOT_FORECASTERS="metaculus/a, metaculus/b",
            BOT_RESEARCHER="asknews/news-summaries",
        )
        self.assertFalse(ens.free_tier)
        self.assertEqual([f.model for f in ens.forecasters], ["metaculus/a", "metaculus/b"])
        self.assertEqual(ens.llms["researcher"], "asknews/news-summaries")
        self.assertNotIn("parser", ens.llms)

    def test_no_keys_means_library_defaults(self):
        self.assertEqual(self.build(), main.Ensemble())


if __name__ == "__main__":
    unittest.main()
