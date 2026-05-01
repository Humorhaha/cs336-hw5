from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import random

from vllm import SamplingParams, LLM

from cs336_alignment.utils import *
from cs336_alignment.vllm import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.vllm import eval_policy_with_vllm

import json
from pathlib import Path

PROMPT_PATH = Path("sft-cs336-assign5-datasets/sft-reason/r1_zero.prompt")

def train_one_sft_run(
    train_examples,
    val_examples,
    model_id: str,
    output_dir: str,
    lr: float,
    seed: int,
    microbatch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_samples: int,
    eval_every_steps: int,
    run_name: str
   ):
    """
    1. load the tokenizer
    2. load the pretrained policy(pretrained model)
    3. init the optimizer
    4. init the vllm instance
    5. eval_sample
    6. load the data
    """
    prompt_template = PROMPT_PATH.read_text()
    train_examples = format_reasoning_examples(train_examples, prompt_template)
    val_examples = format_reasoning_examples(val_examples, prompt_template)

    random.seed(seed)
    torch.manual_seed(seed)
    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.bfloat16, attn_implementation="flash_attention_2").to(policy_device)

    policy.train()

    optimizer = AdamW(policy.parameters(), lr=lr)

    llm = init_vllm(
        model_id=model_id,
        device=vllm_device,
        seed=seed,
        gpu_memory_utilization=0.85,
    )

    eval_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    dataloader = DataLoader(
        train_examples,
        batch_size=microbatch_size,
        shuffle=True,
        collate_fn=make_sft_collate_fn(tokenizer)
    )

    metrics_logs = []
    optimizer_step = 0
    global_train_step = 0

    optimizer.zero_grad()
    # the first raw llm Qwen run
    # pass raw str and no need for tokenization
    eval_metrics, eval_samples = eval_policy_with_vllm(
        val_examples,
        policy,
        llm,
        eval_params,
        max_samples
    )
    eval_metrics["train_step"] = optimizer_step
    metrics_logs.append(eval_metrics)

    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(policy_device)
            label_ids = batch["labels"].to(policy_device)
            response_mask = batch["response_mask"].to(policy_device)

            logits = policy(input_ids).logits
            log_probs = F.log_softmax(logits, dim=-1)
            policy_log_probs = log_probs.gather(
                dim=-1,
                index=label_ids.unsqueeze(-1),
            ).squeeze(-1)
            #按照response的实际长度进行归一化
            normalize_constant = max(response_mask.sum().item(), 1.0)
            loss, train_meta = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps,
                normalize_constant=normalize_constant
            )
            global_train_step += 1

            if global_train_step % gradient_accumulation_steps == 0:
                # 梯度裁剪
                clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % eval_every_steps == 0:
                    eval_metrics, eval_samples = eval_policy_with_vllm(
                        val_examples,
                        policy,
                        llm,
                        eval_params,
                        max_samples
                    )
                    log_row = {
                        "run_name": run_name,
                        "epoch": epoch,
                        "train_step": optimizer_step,
                        "train/loss": float(loss.detach().cpu()),
                        **{k: float(v.detach().cpu()) if torch.is_tensor(v) else v for k, v in train_meta.items()},
                        **eval_metrics,
                    }

                    metrics_logs.append(log_row)

    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(save_dir / "metrics.json", "w") as f:
        for row in metrics_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics_logs



def make_sft_collate_fn(tokenizer):
    def collate_fn(batch):
        """
        数据批量处理
        """
        prompt_strs = [ex["prompt"] for ex in batch]
        response_strs = [ex["response"] for ex in batch]
        prompt_ids = tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
        return prompt_ids
    return collate_fn


def format_reasoning_examples(examples, prompt_template: str):
    if examples and "prompt" in examples[0]:
        return examples

    formatted = []
    for ex in examples:
        item = {
            "prompt": prompt_template.format(question=ex["problem"]),
            "expected_answer": ex["expected_answer"],
        }
        if "reasoning_trace" in ex:
            item["response"] = ex["reasoning_trace"]
        formatted.append(item)
    return formatted


def main():
    model_id = "cs336_alignment/models/Qwen2.5-Math-1.5b"
    sft_path = "sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b.jsonl"
    val_path = "sft-cs336-assign5-datasets/sft-reason/val.jsonl"
    output_dir = "cs336_alignment/output"

    def load_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [json.loads(line) for line in lines]
    
    train_examples = load_jsonl(sft_path)
    val_examples = load_jsonl(val_path)

    seed = 42
    random.seed(seed)
    random.shuffle(train_examples)

    dataset_sizes = [128, 256, 512, 1024, "full"]

    for size in dataset_sizes:
        if size == "full":
            train_subset = train_examples
            run_name = "sft_full"
        else:
            train_subset = train_examples[:size]
            run_name = f"sft_{size}"

        train_one_sft_run(
            run_name=run_name,
            train_examples=train_subset,
            val_examples=val_examples,
            model_id=model_id,
            output_dir=output_dir,
            num_epochs=1,
            microbatch_size=4,
            gradient_accumulation_steps=16,
            lr=1e-3,
            eval_every_steps=50,
            max_eval_examples=None,   # 正式跑用 full validation
            seed=seed,
        )


if __name__ == "__main__":
    main()
