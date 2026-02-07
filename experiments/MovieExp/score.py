import asyncio
import json
from pathlib import Path
import sys

import aiohttp

sys.path.append(".")

from BRIDGE.utils.configure import Configure
from BRIDGE.utils.configure import Field as F

REL_PROMPT = """## Relevance Evaluation Task

You are given the following inputs:

## Task
{task}

## Answer
{response}

---

### Step 1: Understand the Task Requirements
Carefully analyze the **Task** and identify:
- The core objective the Answer is expected to fulfill  
- Any **explicit constraints** (format, scope, style, rules)  
- Any **implicit expectations** based on the User Profile and Chat History  

### Step 2: Evaluate Answer Relevance and Compliance
Evaluate how well the **Answer** satisfies the Task, considering:
- **Relevance**: Does the Answer directly address the Task, or is it partially / mostly off-topic?  
- **Completeness**: Does it cover the key requirements of the Task?  
- **Instruction Adherence**: Does it follow all stated constraints and rules?  
- **Usefulness**: Would this Answer be helpful to the user in accomplishing the Task?  

### Scoring Rules
Assign a **relevance score between 0 and 10**, following these strict constraints:
- Use **exactly three decimal places**  
- The **last decimal digit must be non-zero**  
- Do **not** output integers or tidy decimals  
  - ❌ `2`, `5`, `10`, `2.500`, `7.250`, `8.750`  
  - ✅ `7.183`, `3.947`  

### Scoring Guidelines (Strict)

Use the full 0–10 range. Avoid clustering scores around the middle.

#### 0–2 (Very Low Relevance)
- Answer is irrelevant, misleading, or ignores the Task  
- Major instruction violations  
- Provides little to no usable value  

#### 3–4 (Low Relevance)
- Touches on the Task but misses most key requirements  
- Significant omissions or misunderstandings  
- Limited usefulness  

#### 5–6 (Moderate Relevance)
- Addresses the main Task objective  
- Some constraints are followed, others partially ignored  
- Useful but incomplete or imperfect  

#### 7–8 (High Relevance)
- Clearly and correctly addresses the Task  
- Follows most instructions and constraints  
- Minor issues that do not significantly reduce usefulness  

#### 9–10 (Exceptional Relevance)
- Fully satisfies the Task’s objective  
- Strictly follows all instructions and constraints  
- Highly useful, precise, and well-aligned with user intent  

### Output Format (Strict)
Your final output **must consist of exactly one line** and **nothing else**.

Relevance Score: X.XXX
"""

PERSON_PROMPT = """## Personalization Evaluation Task

You are given the following inputs:

## User Profile
{profile}

## Task
{task}

## Answer
{response}

---

### Step 1: Analyze User Style
Analyze the **user’s style and tone** based on the **User Profile** and **Chat History**, including (but not limited to):
- Level of formality
- Language preference
- Directness vs. verbosity
- Emotional or conversational tone

### Step 2: Evaluate Personalization Quality
Evaluate **how well the Answer is personalized** to the user, considering:
- Whether it reflects any **user-specific traits, preferences, or context**
- Whether it matches the **tone and style** of the user’s prior messages
- Whether it addresses the user’s **explicit or implicit needs**

### Scoring Rules
Assign a **personalization score between 0 and 10**, following these strict constraints:
- Use **exactly three decimal places**
- The **last decimal digit must be non-zero**
- Do **not** output integers or tidy decimals  
  - ❌ `2`, `5`, `10`, `7.250`  
  - ✅ `7.183`, `3.947`

### Scoring Guidelines (Strict and Differentiated)

Use the full 0–10 range. Avoid clustering scores around 5.

#### 0–2 (Very Low Personalization)
- No reference to the user’s background, interests, goals, or communication style
- The answer could be sent to **any user** without modification
- Entirely generic or content-focused

#### 3–4 (Minimal Personalization)
- Superficial or template-based references to the user (e.g., job title or education only)
- Mentions user traits but does **not meaningfully connect** them to the content
- Tone and wording remain generic

#### 5–6 (Moderate Personalization)
- Clearly references **at least one** relevant user trait
- Shows a logical connection between the content and the user’s background or interests
- Style may still feel neutral or slightly generic

#### 7–8 (Strong Personalization)
- Integrates **multiple** user-specific traits (background, interests, goals, or habits)
- Connections feel **intentional and relevant**, not just mentioned
- Tone and wording noticeably align with the user’s communication style

#### 9–10 (Exceptional Personalization)
- Feels explicitly written **for this user and no one else**
- Seamlessly weaves user traits into the explanation
- Strong alignment with the user’s tone, style, and implied motivations
- No signs of generic templates or boilerplate phrasing

### Penalties
- Deduct points if the answer:
  - Uses generic phrases such as “given your background” without elaboration
  - Repeats user attributes without adding insight
  - Sounds like a standard recommendation or summary with light personalization

### Output Format (Strict)
Your final output **must consist of exactly one line** and **nothing else**.
Relevance Score: X.XXX
"""


class ScoreConfig(Configure):
    results_path = F(str, default="outputs/cogen_pareto/latest/results.json", help="Path to results.json")
    output_path = F(str, default="outputs/cogen_pareto/latest/scored.json", help="Path to write scored.json")
    api_base = F(str, required=True, help="LLM judge base URL (ends with /)")
    api_key = F(str, required=True, help="LLM judge API key")
    api_model = F(str, default="gpt-4o", help="LLM judge model")


def parse_score(text: str):
    if not text:
        return None
    digits = [ch for ch in text if ch.isdigit()]
    if not digits:
        return None
    try:
        return int(digits[0])
    except Exception:
        return None


async def judge_batch(session, base_url, api_key, model, prompts):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    results = []
    for prompt in prompts:
        try:
            print(f"[judge] prompt=\n\n====\n\n{prompt}\n\n====...")
            async with session.post(
                url=f"{base_url}chat/completions",
                json={
                    "model": model,
                    "max_tokens": 30,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=headers,
            ) as response:
                text = await response.text()
                print(f"[judge] status={response.status} body={text}")
                if response.status == 200:
                    try:
                        data = json.loads(text)
                        content = data["choices"][0]["message"]["content"]
                        results.append(content)
                    except Exception:
                        results.append("")
                else:
                    results.append("")
        except Exception:
            results.append("")
    return results


async def score_outputs(config: ScoreConfig, result):
    rel_prompts = []
    person_prompts = []
    for item in result["outputs"]:
        rel_prompts.append(
            REL_PROMPT.format(
                task=item["task"],
                profile=item["profile"],
                response=item["output"],
            )
        )
        person_prompts.append(
            PERSON_PROMPT.format(
                task=item["task"],
                profile=item["profile"],
                response=item["output"],
            )
        )
    async with aiohttp.ClientSession() as session:
        rel_raw = await judge_batch(session, config.api_base, config.api_key, config.api_model, rel_prompts)
        person_raw = await judge_batch(session, config.api_base, config.api_key, config.api_model, person_prompts)
    rel_scores = [parse_score(x) for x in rel_raw]
    person_scores = [parse_score(x) for x in person_raw]

    rel_vals = [s for s in rel_scores if s is not None]
    person_vals = [s for s in person_scores if s is not None]
    rel_avg = sum(rel_vals) / max(1, len(rel_vals))
    person_avg = sum(person_vals) / max(1, len(person_vals))
    score_avg = (rel_avg + person_avg) / 2.0

    result["rel_avg"] = rel_avg
    result["person_avg"] = person_avg
    result["score_avg"] = score_avg
    result["rel_scores"] = rel_scores
    result["person_scores"] = person_scores
    result["rel_raw"] = rel_raw
    result["person_raw"] = person_raw
    return result


def main():
    config = ScoreConfig()
    config.parse_sys_args()

    with open(config.results_path, "r") as f:
        results = json.load(f)

    async def run_all():
        scored = []
        for result in results:
            scored.append(await score_outputs(config, result))
        return scored

    scored_results = asyncio.run(run_all())

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scored_results, f, indent=2)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
