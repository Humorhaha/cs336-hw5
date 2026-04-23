import json
import os
from vllm import LLM, SamplingParams
from typing import Callable, List
from pathlib import Path
from drgrpo_grader import r1_zero_reward_fn
import matplotlib.pyplot as plt



def eval_llm(vllm_model: LLM, reward_fn: Callable[[str, str], dict[str, float]], prompts: List[str], eval_sampling_params: SamplingParams) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)



    answers: List[str] = []
    examples: List[dict[str, str]] = []

    with open(DATASET_PATH / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            examples.append(data)
            answers.append(data["answer"])

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    rewards: List[dict[str, float]] = []

    generation_results: List[str] = []
    for ix, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generation_results.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        rewards.append(reward_fn(generated_text, answers[ix]))

    output_file = Path(__file__).parent / "eval.jsonl"
    with open(output_file, "w") as f:
        for example,  generation, reward in zip(examples, generation_results, rewards):
            result = {
                "example": example,
                "generation": generation,
                "reward": reward
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    

def analyze_eval_results(eval_file: Path | str) -> None:
    """
    Analyze evaluation results.
    """
    if not isinstance(eval_file, Path):
        eval_file = Path(eval_file)

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            reward = data["reward"]
            







if __name__ == "__main__":
    llm = LLM(model=str(Path(__file__).parent / "models" / "Qwen2.5-Math-1.5b"))
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    eval_sampling_params.include_stop_str_in_output = True
    PROMPT_PATH = Path(__file__).parent / "prompts" / "r1_zero.prompt"
    DATASET_PATH = Path(__file__).parent.parent / "data" /  "gsm8k"

    prompts = []
    with open(PROMPT_PATH, "r", encoding="utf-8") as f, open(DATASET_PATH / "test.jsonl", "r", encoding="utf-8") as g:
        prompt = f.read()
        for line in g:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))


    eval_llm(llm, r1_zero_reward_fn, prompts, eval_sampling_params)
    print("Done")




