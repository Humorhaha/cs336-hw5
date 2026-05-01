from vllm.model_executor import set_random_seed as set_vllm_random_seed
from vllm import SamplingParams, LLM
from transformers import PreTrainedModel
from unittest.mock import patch
import torch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start inference process
    Here we hold a GPU instance for inference, separate from the policy.
    """
    set_vllm_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profile_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=False)

    with world_size_patch, profile_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Load the policy that is being updated into the eval llm
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def eval_policy_with_vllm(
    val_examples,
    policy: PreTrainedModel,
    llm: LLM,
    sampling_params: SamplingParams,
    max_samples: int=None,
):
    """
    Use val examples to evaluate the updating policy with vllm
    """
    policy.eval()
    load_policy_into_vllm_instance(policy, llm)

    if max_samples is not None:
        eval_examples = val_examples[:max_samples]
    else:
        eval_examples = val_examples
    
    if hasattr(llm, "reset_prefix_cache"):
        llm.reset_prefix_cache()
    
    prompts = [ex["prompt"] for ex in eval_examples]
    ground_truths = [ex["expected_answer"] for ex in eval_examples]

    outputs = llm.generate(prompts, sampling_params)
    generations = [out.outputs[0].text for out in outputs]

    rewards = [
        r1_zero_reward_fn(generation, ground_truth)
        for generation, ground_truth in zip(generations, ground_truths)
    ]

    metrics = {
        "accuracy": sum(r["reward"] for r in rewards) / len(rewards),
        "format_accuracy": sum(r["format_reward"] for r in rewards) / len(rewards),
        "answer_accuracy": sum(r["answer_reward"] for r in rewards) / len(rewards),
    }

    samples = [
    {
        "prompt": prompt,
        "generation": generation,
        "ground_truth": ground_truth,
        "reward": reward
    }
    for prompt, ground_truth, generation, reward in zip(prompts, ground_truths, generations, rewards)
    ]


    policy.train()
    return metrics, samples
