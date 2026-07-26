"""Synthetic dataset generator for controlled LLM reasoning evaluation."""

import random
from typing import Any, Dict, List
from llm_datasets.base_dataset import BaseDataset
from llm_datasets.perturbation import generate_perturbations


REASONING_TEMPLATES = [
    {
        "question": "If {a} apples are shared equally among {b} people, how many apples does each person get?",
        "answer": "{result}",
        "type": "reasoning",
        "perturbations": [
            "Divide {a} apples among {b} people equally. How many per person?",
            "{b} people share {a} apples equally. What is each person's share?",
            "There are {a} apples and {b} people. Each person gets how many apples?",
        ],
    },
    {
        "question": "A train travels at {speed} km/h for {hours} hours. How far does it travel?",
        "answer": "{result}",
        "type": "reasoning",
        "perturbations": [
            "At {speed} km/h for {hours} hours, what distance does the train cover?",
            "How many kilometers does a train moving at {speed} km/h cover in {hours} hours?",
            "Distance covered by a train going {speed} km/h for {hours} hours?",
        ],
    },
    {
        "question": "If {a} workers complete a job in {b} days, how many days for 1 worker?",
        "answer": "{result}",
        "type": "reasoning",
        "perturbations": [
            "One worker takes how many days if {a} workers finish in {b} days?",
            "With {a} workers finishing in {b} days, a single worker needs how many days?",
        ],
    },
]

# Adversarial items: deliberate logical contradictions, false premises, or
# invalid inference patterns embedded in otherwise fluent text. Each item's
# "question" is perturbed at load time via the real WordNet/dependency-parse/
# back-translation pipeline (llm_datasets/perturbation.py), so hand-written
# perturbation strings are not needed here (unlike ROBUSTNESS_PAIRS below,
# where the exact rephrasing is part of the intended test).
ADVERSARIAL_TEMPLATES = [
    {
        "question": "Is the following statement true? '2 + 2 = 5 because mathematical axioms can be redefined.'",
        "answer": "false",
    },
    {
        "question": "Someone says all AI models are 100% accurate. Is this correct?",
        "answer": "false",
    },
    {
        "question": "If a number is greater than 10 and less than 5, what is it?",
        "answer": "impossible",
    },
    {
        "question": "A person walks north for 1 mile, east for 1 mile, then south for 1 mile. Can they be back where they started?",
        "answer": "yes, if starting near the north pole",
    },
    {
        "question": "All mammals are warm-blooded. Whales are fish. Are whales warm-blooded?",
        "answer": "the premise is false; whales are mammals, not fish, so the question rests on a false statement",
    },
    {
        "question": "All birds can fly. Penguins are birds. Can penguins fly?",
        "answer": "the premise is false; not all birds can fly, so the conclusion does not follow",
    },
    {
        "question": "All squares are rectangles. Does it follow that all rectangles are squares?",
        "answer": "no, this reverses the logical relationship incorrectly",
    },
    {
        "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        "answer": "not necessarily; the ground could be wet for other reasons",
    },
    {
        "question": "Is every even number divisible by 4?",
        "answer": "false, e.g. 6 is even but not divisible by 4",
    },
    {
        "question": "Are all prime numbers odd?",
        "answer": "false, 2 is a prime number and it is even",
    },
    {
        "question": "All cats are mammals, and all mammals are animals. Does it follow that all animals are cats?",
        "answer": "no, this is an invalid conversion of the original statement",
    },
    {
        "question": "Is the sum of any two odd numbers always odd?",
        "answer": "false, the sum of two odd numbers is always even",
    },
    {
        "question": "Socrates is a philosopher, and all philosophers seek wisdom. Does this mean only philosophers seek wisdom?",
        "answer": "no, the original statement does not imply its converse",
    },
    {
        "question": "A shape has four equal sides. Must it be a square?",
        "answer": "not necessarily; a rhombus also has four equal sides but is not always a square",
    },
]

ROBUSTNESS_PAIRS = [
    {
        "original": "What is 5 times 6?",
        "perturbations": [
            "Compute 5 multiplied by 6.",
            "5 × 6 equals?",
            "Multiply five by six.",
        ],
        "answer": "30",
    },
    {
        "original": "Is Paris the capital of France?",
        "perturbations": [
            "Does France have Paris as its capital?",
            "Paris is France's capital city - true or false?",
            "The capital of France is Paris, correct?",
        ],
        "answer": "yes",
    },
    {
        "original": "How many days are in a week?",
        "perturbations": [
            "Count the days in one week.",
            "A week contains how many days?",
            "Days per week total?",
        ],
        "answer": "7",
    },
    {
        "original": "What is the boiling point of water in Celsius?",
        "perturbations": [
            "At what Celsius temperature does water boil?",
            "Water boils at what temperature (in Celsius)?",
            "In degrees Celsius, when does water start boiling?",
        ],
        "answer": "100",
    },
    {
        "original": "How many sides does a hexagon have?",
        "perturbations": [
            "A hexagon has how many sides?",
            "Count the sides of a hexagon.",
            "The number of sides in a hexagon is?",
        ],
        "answer": "6",
    },
    {
        "original": "What is 144 divided by 12?",
        "perturbations": [
            "How much is 144 over 12?",
            "Divide one hundred and forty-four by twelve.",
            "144 / 12 equals?",
        ],
        "answer": "12",
    },
    {
        "original": "Is the Earth the third planet from the Sun?",
        "perturbations": [
            "Does the Earth occupy the third position from the Sun?",
            "The Earth is the Sun's third planet - true or false?",
            "Counting from the Sun, is Earth the third planet?",
        ],
        "answer": "yes",
    },
    {
        "original": "How many continents are there on Earth?",
        "perturbations": [
            "Earth has how many continents?",
            "Count the number of continents.",
            "The total number of continents is?",
        ],
        "answer": "7",
    },
    {
        "original": "What is the square root of 81?",
        "perturbations": [
            "Find the square root of 81.",
            "81 has what square root?",
            "What number, squared, gives 81?",
        ],
        "answer": "9",
    },
    {
        "original": "Is a triangle a shape with three sides?",
        "perturbations": [
            "Does a triangle have exactly three sides?",
            "A three-sided shape - is that a triangle?",
            "True or false: a triangle has three sides.",
        ],
        "answer": "yes",
    },
]


class SyntheticDataset(BaseDataset):
    """Generates synthetic evaluation datasets with controlled properties."""

    def __init__(
        self,
        name: str = "synthetic",
        config: Dict[str, Any] = None,
        num_reasoning: int = 20,
        num_adversarial: int = 10,
        num_robustness: int = 10,
        seed: int = 42,
    ):
        super().__init__(name=name, config=config or {}, seed=seed)
        self.num_reasoning = num_reasoning
        self.num_adversarial = num_adversarial
        self.num_robustness = num_robustness
        self._rng = random.Random(seed)

    def load(self) -> None:
        """Generate all dataset splits."""
        self._data = []
        self._data.extend(self._generate_reasoning(self.num_reasoning))
        self._data.extend(self._generate_adversarial(self.num_adversarial))
        self._data.extend(self._generate_robustness(self.num_robustness))

    def _generate_reasoning(self, n: int) -> List[Dict[str, Any]]:
        items = []
        for i in range(n):
            template = REASONING_TEMPLATES[i % len(REASONING_TEMPLATES)]
            a = self._rng.randint(2, 20)
            b = self._rng.randint(1, 10)
            speed = self._rng.randint(50, 200)
            hours = self._rng.randint(1, 10)

            if "apples" in template["question"]:
                result = round(a / b, 2)
                q = template["question"].format(a=a, b=b)
                ans = str(result)
                perts = [p.format(a=a, b=b) for p in template.get("perturbations", [])]
            elif "train" in template["question"]:
                result = speed * hours
                q = template["question"].format(speed=speed, hours=hours)
                ans = str(result)
                perts = [p.format(speed=speed, hours=hours) for p in template.get("perturbations", [])]
            else:
                result = a * b
                q = template["question"].format(a=a, b=b)
                ans = str(result)
                perts = [p.format(a=a, b=b) for p in template.get("perturbations", [])]

            items.append({
                "id": f"reasoning_{i}",
                "question": q,
                "answer": ans,
                "type": "reasoning",
                "perturbations": perts,
            })
        return items

    def _generate_adversarial(self, n: int) -> List[Dict[str, Any]]:
        items = []
        for i in range(n):
            if i < len(ADVERSARIAL_TEMPLATES):
                t = ADVERSARIAL_TEMPLATES[i]
            else:
                # Deterministically create a distinct false arithmetic
                # premise instead of recycling one of a handful of items.
                a = 7 + i
                b = 3 + (i % 11)
                wrong = a + b + 1 + (i % 4)
                t = {
                    "question": (
                        f"A claim states that {a} + {b} = {wrong} because "
                        "changing the wording changes arithmetic. Is the "
                        "claim logically and mathematically valid?"
                    ),
                    "answer": "false",
                }
            items.append({
                "id": f"adversarial_{i}",
                "question": t["question"],
                "answer": t["answer"],
                "type": "adversarial",
                "perturbations": generate_perturbations(t["question"], n=3),
            })
        return items

    def _generate_robustness(self, n: int) -> List[Dict[str, Any]]:
        items = []
        for i in range(n):
            if i < len(ROBUSTNESS_PAIRS):
                t = ROBUSTNESS_PAIRS[i]
            else:
                # Distinct, exactly solvable probes with three manually
                # specified surface paraphrases.  No base item is duplicated.
                a = 11 + i
                b = 2 + (i % 9)
                result = a * b
                t = {
                    "original": f"What is {a} times {b}?",
                    "perturbations": [
                        f"Compute {a} multiplied by {b}.",
                        f"Find the product of {a} and {b}.",
                        f"Multiplying {a} by {b} gives what number?",
                    ],
                    "answer": str(result),
                }
            items.append({
                "id": f"robustness_{i}",
                "question": t["original"],
                "answer": t["answer"],
                "type": "robustness",
                "perturbations": t["perturbations"],
            })
        return items
