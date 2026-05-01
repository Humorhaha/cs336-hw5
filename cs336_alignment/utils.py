from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch
import torch.nn.functional as F
from vllm import LLM
def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase) -> dict[str, torch.Tensor]:
    """
    Tokenize prompt string and output strings. 

    Construct a response mask to mask out the prompt tokens and padding tokens.
    """
    assert len(prompt_strs) == len(output_strs),  "Prompt and output strings must have the same length"
    batch_size = len(prompt_strs)
    question_ids = [tokenizer(prompt)["input_ids"] for prompt in prompt_strs]
    output_ids = [tokenizer(output)["input_ids"] for output in output_strs]

    max_len = 0
    for question, output in zip(question_ids, output_ids):
        max_len = max(max_len, len(question) + len(output))
    
    full_tokens = torch.empty((batch_size, max_len), dtype=torch.int32)
    response_mask = torch.empty((batch_size, max_len), dtype=torch.bool)
    response_mask[:, :] = False

    for ix, (question, output) in enumerate(zip(question_ids, output_ids)):
        pad_len = max_len - len(question) - len(output)
        full_tokens[ix, :len(question) + len(output)] = torch.tensor(question + output)
        full_tokens[ix, len(question) + len(output):] = tokenizer.pad_token_id * torch.ones(pad_len, dtype=torch.int32)
        response_mask[ix, len(question):len(question) + len(output)] = True

    
    input_ids = full_tokens[:, :-1]
    labels = full_tokens[:, 1:]
    response_mask = response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


def compute_entropy(logits: torch.Tensor, return_log_probs: bool = False) -> torch.Tensor:
    """
    Compute per-token-entropy on the dim of 'vocab_size'
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    if return_log_probs:
        return log_probs

    return - (probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool
) -> dict[str, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input_ids.to(device)
    labels.to(device)
    
    with torch.inference_mode():
        logits = model(input_ids).logits
        probs = F.softmax(logits, dim=-1)
        probs = probs.gather(dim=-1, index = labels.unsqueeze(-1)).squeeze(-1)

        if return_token_entropy:
            return {
                "log_probs": probs.log(),
                "token_entropy": compute_entropy(logits, return_token_entropy)
            }
        
        else:
            return {"log_probs": probs.log()}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None
) -> torch.Tensor:
    """
    mask is the same shape as tensor, where 1 means should be included in the sum.
    normalize_constant is the constant to normalize the sum.
    if dim is None, sum over all dimensions
    """
    tensor_masked = tensor * mask
    if dim is None:
        return tensor_masked.sum() / normalize_constant
    
    return tensor_masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    a single microbatch train update for SFT
    """
    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant) / policy_log_probs.shape[0]
    loss /= gradient_accumulation_steps
    loss.backward()

    metadata = {
        "loss_mean": loss.item(),
        "gradient_accumulation_steps": gradient_accumulation_steps
    }

    return loss, metadata

def log_generations(
    model: PreTrainedModel | LLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
):
    """
    Log the generations of the model
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)

    with torch.inference_mode():
        logits = model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)
        next_tokens = log_probs.argmax(dim=-1)

        correct_per_token = next_tokens == labels
        correct_response = (correct_per_token | ~response_mask).all(dim=-1)
        response_lengths = response_mask.sum(dim=-1)
        correct_response_lengths = response_lengths[correct_response]

        average_response_length = response_lengths.float().mean().item()
        average_correct_response_length = (
            correct_response_lengths.float().mean().item()
            if correct_response_lengths.numel() > 0
            else 0.0
        )
        average_entropy = token_entropy[response_mask].mean().item()

    labels_cpu = labels.cpu()
    next_tokens_cpu = next_tokens.cpu()
    response_mask_cpu = response_mask.cpu()

    ground_truth_response = [
        tokenizer.decode(row[mask], skip_special_tokens=True)
        for row, mask in zip(labels_cpu, response_mask_cpu)
    ]
    response = [
        tokenizer.decode(row[mask], skip_special_tokens=True)
        for row, mask in zip(next_tokens_cpu, response_mask_cpu)
    ]

    return {
        "prompt": tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True),
        "ground_truth": ground_truth_response,
        "response": response,
        "average_entropy": average_entropy,
        "average_response_length": average_response_length,
        "average_correct_response_length": average_correct_response_length,
        "num_correct_responses": correct_response.sum().item(),
        "response_accuracy": correct_response.float().mean().item(),
    }
