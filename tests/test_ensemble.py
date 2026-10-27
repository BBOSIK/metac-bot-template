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


class BuildEnsembleTest(unittest.TestCase):
    def test_openrouter_key_enables_default_ensemble(self):
        env = {"OPENROUTER_API_KEY": "sk-or-test", "BOT_FORECASTERS": "", "BOT_RESEARCHER": "", "BOT_PARSER": ""}
        with mock.patch.dict(os.environ, env):
            forecasters, llms = main.build_ensemble()
        self.assertEqual([f.model for f in forecasters], main.OPENROUTER_FORECASTERS)
        self.assertEqual(llms["researcher"].model, main.OPENROUTER_RESEARCHER)
        self.assertEqual(llms["parser"].model, main.OPENROUTER_PARSER)
        self.assertIs(llms["default"], forecasters[0])

    def test_env_overrides_and_asknews_passthrough(self):
        env = {
            "OPENROUTER_API_KEY": "REPLACE_ME",
            "BOT_FORECASTERS": "metaculus/a, metaculus/b",
            "BOT_RESEARCHER": "asknews/news-summaries",
            "BOT_PARSER": "",
        }
        with mock.patch.dict(os.environ, env):
            forecasters, llms = main.build_ensemble()
        self.assertEqual([f.model for f in forecasters], ["metaculus/a", "metaculus/b"])
        self.assertEqual(llms["researcher"], "asknews/news-summaries")
        self.assertNotIn("parser", llms)

    def test_no_keys_means_library_defaults(self):
        env = {"OPENROUTER_API_KEY": "", "BOT_FORECASTERS": "", "BOT_RESEARCHER": "", "BOT_PARSER": ""}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(main.build_ensemble(), ([], {}))


if __name__ == "__main__":
    unittest.main()
