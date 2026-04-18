# Evaluating Gemma 4 26B Coding Efficacy

Model: `google/gemma-4-26b-a4b` running on LMStudio at `macstudio.local:1234`

## Quick Sanity Checks

Before running a full benchmark, try a few manual probes to get a feel for the model's strengths and blind spots.

```bash
# Basic function generation
curl -s http://macstudio.local:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-26b-a4b",
    "messages": [{"role": "user", "content": "Write a Python function that takes a list of intervals [[start, end], ...] and merges overlapping intervals. Include type hints and a docstring."}],
    "temperature": 0.0,
    "max_tokens": 1024
  }' | python3 -m json.tool

# Bug detection
curl -s http://macstudio.local:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-26b-a4b",
    "messages": [{"role": "user", "content": "Find the bug:\n\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1"}],
    "temperature": 0.0,
    "max_tokens": 512
  }' | python3 -m json.tool

# Multi-file reasoning
curl -s http://macstudio.local:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-26b-a4b",
    "messages": [{"role": "user", "content": "Given this SQLAlchemy model:\n\nclass User(Base):\n    __tablename__ = \"users\"\n    id = Column(Integer, primary_key=True)\n    email = Column(String, unique=True, nullable=False)\n    posts = relationship(\"Post\", back_populates=\"author\")\n\nclass Post(Base):\n    __tablename__ = \"posts\"\n    id = Column(Integer, primary_key=True)\n    title = Column(String, nullable=False)\n    author_id = Column(Integer, ForeignKey(\"users.id\"), nullable=False)\n    author = relationship(\"User\", back_populates=\"posts\")\n\nWrite a FastAPI endpoint that returns the top 10 users by post count with their 3 most recent post titles. Use a single efficient query."}],
    "temperature": 0.0,
    "max_tokens": 1024
  }' | python3 -m json.tool
```

## HumanEval Benchmark (164 problems)

The standard coding eval. Tests function-level code generation with unit test verification.

### Setup

```bash
# On the machine that will run the eval (not necessarily macstudio)
pip install human-eval openai

# Or in a fresh venv
python3 -m venv ~/.eval-env
source ~/.eval-env/bin/activate
pip install human-eval openai
```

### Generate completions

```python
#!/usr/bin/env python3
"""Run HumanEval against a local LMStudio model."""

import json
from pathlib import Path
from openai import OpenAI
from human_eval.data import read_problems

MODEL = "google/gemma-4-26b-a4b"
BASE_URL = "http://macstudio.local:1234/v1"
OUTPUT = Path("humaneval_gemma4_samples.jsonl")
NUM_SAMPLES = 1  # increase for pass@k with k>1

client = OpenAI(base_url=BASE_URL, api_key="not-needed")
problems = read_problems()

with open(OUTPUT, "w") as f:
    for i, (task_id, problem) in enumerate(problems.items()):
        print(f"[{i+1}/{len(problems)}] {task_id}")

        for _ in range(NUM_SAMPLES):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Complete the following Python function. Return ONLY the function body, no explanation."},
                    {"role": "user", "content": problem["prompt"]},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            completion = response.choices[0].message.content

            # Strip markdown fences if present
            if "```" in completion:
                lines = completion.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                completion = "\n".join(lines)

            f.write(json.dumps({
                "task_id": task_id,
                "completion": completion,
            }) + "\n")

print(f"Done. Samples written to {OUTPUT}")
```

### Score

```bash
evaluate_functional_correctness humaneval_gemma4_samples.jsonl
# Prints pass@1 score (e.g., 0.72 = 72%)
```

### Reference scores (approximate)

| Model | Size | pass@1 |
|-------|------|--------|
| GPT-4o | - | ~90% |
| Claude Sonnet 4 | - | ~88% |
| Gemma 2 27B | 27B | ~55% |
| Qwen 2.5 Coder 32B | 32B | ~72% |
| Gemma 4 27B (reported) | 27B | ~70% |

## MBPP (Mostly Basic Python Problems)

Broader eval with 974 problems, tests more everyday coding patterns.

```bash
pip install datasets
```

```python
#!/usr/bin/env python3
"""Run MBPP sanitized subset against a local model."""

import json
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

MODEL = "google/gemma-4-26b-a4b"
BASE_URL = "http://macstudio.local:1234/v1"
OUTPUT = Path("mbpp_gemma4_results.jsonl")

client = OpenAI(base_url=BASE_URL, api_key="not-needed")
dataset = load_dataset("mbpp", "sanitized", split="test")

correct = 0
total = 0

with open(OUTPUT, "w") as f:
    for i, item in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] {item['text'][:80]}...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Write a Python function to solve this problem. Return ONLY the code, no explanation."},
                {"role": "user", "content": item["text"]},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        completion = response.choices[0].message.content

        # Strip markdown fences
        if "```" in completion:
            lines = completion.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            completion = "\n".join(lines)

        # Run test cases
        passed = True
        for test in item["test_list"]:
            try:
                exec(completion + "\n" + test, {})
            except Exception:
                passed = False
                break

        if passed:
            correct += 1
        total += 1

        f.write(json.dumps({
            "task_id": item["task_id"],
            "passed": passed,
            "completion": completion,
        }) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  Running accuracy: {correct}/{total} = {correct/total:.1%}")

print(f"\nFinal: {correct}/{total} = {correct/total:.1%}")
```

## Dorothy-Specific Eval

Test things that matter for this project — can the model review code, spot bugs in synthesis output, and reason about the codebase patterns we actually use?

```python
#!/usr/bin/env python3
"""Dorothy-specific coding eval for reviewer model candidates."""

from openai import OpenAI

MODEL = "google/gemma-4-26b-a4b"
BASE_URL = "http://macstudio.local:1234/v1"
client = OpenAI(base_url=BASE_URL, api_key="not-needed")

TASKS = [
    {
        "name": "JSON extraction from messy LLM output",
        "prompt": '''Extract valid JSON from this LLM response. Return ONLY the JSON.

Sure! Here's the analysis:

```json
{
  "scores": {"factuality": 8, "neutrality": 7, "completeness": 5, "structure": 8},
  "improvements": ["Added missing context about the timeline", "Removed editorializing in paragraph 3"],
  "headline": "Senate Passes Infrastructure Bill After Marathon Debate",
  "article": "The U.S. Senate passed a $1.2 trillion infrastructure bill early Saturday..."
}
```

Let me know if you need anything else!''',
        "check": lambda r: '"factuality"' in r and '"improvements"' in r,
    },
    {
        "name": "OpenSearch query construction",
        "prompt": '''Write an OpenSearch query (as Python dict) that finds articles:
- Published in the last 24 hours (field: pub_date)
- In the "politics" column (field: column)  
- That do NOT have an embedding (field: embedding should not exist)
- Sorted by pub_date descending
- Limited to 50 results''',
        "check": lambda r: "must_not" in r and "exists" in r and "pub_date" in r,
    },
    {
        "name": "Bias detection in synthesis",
        "prompt": '''Review this synthesized news paragraph for neutrality issues. List each instance of biased language and suggest a neutral replacement.

"The radical new policy, which critics have rightfully slammed as dangerous, would gut environmental protections that have safeguarded communities for decades. Supporters desperately claim it will boost the struggling economy, but experts warn the consequences could be devastating."''',
        "check": lambda r: any(w in r.lower() for w in ["radical", "rightfully", "gut", "desperately", "devastating"]),
    },
    {
        "name": "Python bug detection",
        "prompt": '''Find all bugs in this function:

def bulk_update_embeddings(updates, index_name):
    actions = [
        {"_op_type": "update", "_index": index_name, "_id": aid, "doc": {"embedding": emb}}
        for aid, emb in updates
    ]
    success, errors = helpers.bulk(client, actions, raise_on_error=False, stats_only=True)
    error_count = len(errors) if isinstance(errors, list) else 0
    return (success, error_count)''',
        "check": lambda r: "stats_only" in r,  # stats_only=True means errors is an int, not a list
    },
]

print(f"Running {len(TASKS)} Dorothy-specific tasks against {MODEL}\n")

for task in TASKS:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": task["prompt"]}],
        temperature=0.0,
        max_tokens=1024,
    )
    result = response.choices[0].message.content
    passed = task["check"](result)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {task['name']}")
    if not passed:
        print(f"  Response preview: {result[:200]}...")
    print()
```

## Running the evals

```bash
# From any machine with network access to macstudio.local:1234
source ~/.eval-env/bin/activate

# Quick sanity (2 min)
python3 eval_dorothy.py

# HumanEval (30-60 min depending on speed)
python3 eval_humaneval.py
evaluate_functional_correctness humaneval_gemma4_samples.jsonl

# MBPP (60-90 min)
python3 eval_mbpp.py
```

## Comparing models

To test a different model, change `MODEL` in the scripts and re-run. Compare results side by side. If you want to test a model not yet loaded in LMStudio, load it there first and use its exact model identifier.
