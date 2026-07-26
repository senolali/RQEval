"""Regression tests for the paper-aligned evaluation protocol."""

import pathlib
import unittest

from llm_datasets.synthetic_dataset import SyntheticDataset
from metrics.accuracy import AccuracyMetric
from metrics.aggregation import AggregationStrategy
from metrics.consistency import ConsistencyMetric
from metrics.robustness import RobustnessMetric
from models.base_model import BaseModel


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CanonicalMatchingTests(unittest.TestCase):
    def setUp(self):
        self.metric = AccuracyMetric()

    def test_short_choice_does_not_match_inside_word(self):
        self.assertFalse(self.metric._is_correct("This is drastic.", "D"))

    def test_short_number_does_not_match_inside_larger_number(self):
        self.assertFalse(self.metric._is_correct("The answer is 25.", "5"))

    def test_structurally_anchored_choice_matches(self):
        self.assertTrue(self.metric._is_correct("The correct option is D.", "D"))

    def test_gsm8k_final_answer_matches(self):
        self.assertTrue(self.metric._is_correct("Work... #### 30", "30"))


class FixedDenominatorTests(unittest.TestCase):
    def test_failed_consistency_run_stays_in_k(self):
        metric = ConsistencyMetric({"consistency_runs": 3})
        # One agreeing pair out of all three K=3 pairs.
        self.assertAlmostEqual(metric.compute_instance(["yes", "yes"], "yes"), 1 / 3)

    def test_failed_perturbation_stays_in_p(self):
        metric = RobustnessMetric({"robustness_perturbations": 3})
        score = metric.compute(["yes"], [["yes", ""]], ["yes"])
        self.assertAlmostEqual(score, 1 / 3)

    def test_incorrect_originals_excluded_from_denominator(self):
        # Paper formula (Sec. 3.2): RS = (1/|C|) * sum_{i in C} (...).
        # Item 0 is originally correct and perfectly robust (3/3);
        # item 1 is originally incorrect, so it must be excluded from
        # BOTH the numerator and the denominator -- not merely zeroed
        # out while still counting toward N, which would silently
        # deflate RS to 0.5 instead of the correct 1.0.
        metric = RobustnessMetric({"robustness_perturbations": 3})
        score = metric.compute(
            ["yes", "no"],
            [["yes", "yes", "yes"], ["yes", "yes", "yes"]],
            ["yes", "yes"],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_zero_correctness_yields_zero_not_error(self):
        metric = RobustnessMetric({"robustness_perturbations": 3})
        score = metric.compute(["no"], [["yes", "yes", "yes"]], ["yes"])
        self.assertEqual(score, 0.0)


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_text = (ROOT / "config" / "config.yaml").read_text(
            encoding="utf-8"
        )

    def test_gemini_budget(self):
        gemini_block = self.config_text.split(
            '- name: "Gemini-2.5-Flash"', 1
        )[1].split('- name:', 1)[0]
        self.assertIn("max_tokens: 1024", gemini_block)

    def test_paper_weights(self):
        safety = self.config_text.split("safety_priority:", 1)[1].split(
            "accuracy_priority:", 1
        )[0]
        accuracy = self.config_text.split("accuracy_priority:", 1)[1].split(
            "efficiency_priority:", 1
        )[0]
        self.assertIn("consistency:      0.05", safety)
        self.assertIn("logical_coherence: 0.25", safety)
        self.assertIn("correctness:      0.50", accuracy)
        self.assertIn("logical_coherence: 0.15", accuracy)

    def test_default_aggregation_matches_config(self):
        defaults = AggregationStrategy().get_strategy_weights("safety_priority")
        self.assertEqual(defaults["logical_coherence"], 0.25)


class SyntheticDatasetTests(unittest.TestCase):
    def test_robustness_probes_are_unique_at_paper_scale(self):
        dataset = SyntheticDataset(
            num_reasoning=0,
            num_adversarial=0,
            num_robustness=75,
            seed=42,
        )
        dataset.load()
        questions = [item["question"] for item in dataset.get_all()]
        self.assertEqual(len(questions), 75)
        self.assertEqual(len(set(questions)), 75)
        self.assertTrue(all(len(item["perturbations"]) == 3 for item in dataset.get_all()))


class TokenUsageTests(unittest.TestCase):
    class DummyModel(BaseModel):
        def generate(self, prompt: str, **kwargs) -> str:
            return prompt

        def generate_with_trace(self, prompt: str, **kwargs):
            return {}

    def test_provider_reported_output_tokens_take_precedence(self):
        model = self.DummyModel("dummy", {}, deterministic=True)
        model._set_reported_output_tokens(17)
        self.assertEqual(model._take_output_token_count("one two"), 17)

    def test_cached_response_preserves_reported_usage(self):
        model = self.DummyModel("dummy", {}, deterministic=True)
        key = model._maybe_cache_key("prompt")
        model._set_reported_output_tokens(9)
        model._set_cached(key, "response")
        self.assertEqual(model._get_cached(key), "response")
        self.assertEqual(model._take_output_token_count("response"), 9)


if __name__ == "__main__":
    unittest.main()
