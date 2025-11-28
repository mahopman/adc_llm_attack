# %% [markdown]
# # Self-Propagating Adversarial PREFIX Demo
#
# This notebook demonstrates how to find a single adversarial token that the model will
# output BEFORE its response. The token appears in input and is repeated at the START of output.
#
# **Goal**: Find a token that, when appended to a user prompt, causes the model to output
# that same token BEFORE generating its response.
#
# Layout: `[prompt] [ADV] -> [ADV] [response]`

# %% [markdown]
# ## Setup

# %%
import torch
import torch.nn.functional as F
import numpy as np
import gc
from datasets import load_dataset
from utils import get_model
from utils.llm_utils import get_chat_template

# Clean up any existing model from previous runs
if 'model' in dir():
    del model
if 'tokenizer' in dir():
    del tokenizer
gc.collect()
torch.cuda.empty_cache()

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# %%
# Load TruthfulQA dataset
truthful_qa = load_dataset("truthful_qa", "generation", split="validation")
print(f"Loaded {len(truthful_qa)} questions from TruthfulQA")

# %%
# Model to use - change this to swap models
model_name = "HuggingFaceH4/zephyr-7b-beta"

print(f"Loading model: {model_name}")
model, tokenizer = get_model(model_name)
model.eval()
print("Model loaded successfully!")

# %%
# Detect model type and set chat template
def get_model_type(model_name):
    """Detect model type from name."""
    name_lower = model_name.lower()
    if "vicuna" in name_lower:
        return "vicuna"
    elif "llama-2" in name_lower or "llama2" in name_lower:
        return "llama-2-chat"
    elif "llama-3" in name_lower or "llama3" in name_lower:
        return "llama-3-instruct"
    elif "zephyr" in name_lower:
        return "zephyr"
    elif "mistral" in name_lower:
        return "mistral-instruct"
    else:
        return "vicuna"  # default

model_type = get_model_type(model_name)
tokenizer.chat_template = get_chat_template(model_type)
print(f"Using chat template: {model_type}")

# %% [markdown]
# ## Generate Target Response
#
# First, we get the model's natural response to a simple prompt. This will be our target
# response that we want the model to generate after the adversarial prefix.

# %%
# Pick a question from TruthfulQA (change index to try different questions)
question_idx = 0
user_prompt = truthful_qa[question_idx]["question"]
print(f"Question {question_idx}: {user_prompt}")

# %%
# Build input using chat template
messages = [{"role": "user", "content": user_prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate response
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

# Extract generated text
generated_ids = output_ids[0][input_ids.shape[1]:]
target_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print(f"User prompt: {user_prompt}")
print(f"Target response: {target_response}")

# %% [markdown]
# ## Build Input Template with Adversarial Placeholder
#
# We create a template where:
# - Input: `[system] USER: {prompt} {ADV_TOKEN} ASSISTANT: {ADV_TOKEN} {response}`
# - Target output: `{ADV_TOKEN}` at the START of response (prefix, not suffix)

# %%
NUM_ADV_TOKENS = 1  # Single token to optimize

# %%
# Placeholder token (will be optimized)
adv_placeholder = " !"

# Build messages: ADV appears in user message, and at START of assistant response
messages_with_adv = [
    {"role": "user", "content": f"{user_prompt} {adv_placeholder}"},
    {"role": "assistant", "content": f"{adv_placeholder} {target_response}"},  # ADV before response
]

# %% [markdown]
# ### Apply Chat Template

# %%
full_string = tokenizer.apply_chat_template(
    messages_with_adv, tokenize=False, add_generation_prompt=False
)

input_ids = tokenizer.encode(full_string, add_special_tokens=True)

print(f"Total tokens: {len(input_ids)}")
print(f"\nFull template:\n{full_string}")

# %% [markdown]
# ## Find Token Positions
#
# We need to identify:
# 1. The adversarial token in the INPUT (user message)
# 2. The adversarial token in the OUTPUT (start of assistant response)

# %%
# Find token positions by searching backwards
adv_input_start = None
adv_output_start = None
response_start = None
response_stop = len(input_ids)

# Find where target_response starts (after the output ADV token)
for i in range(len(input_ids), 0, -1):
    decoded = tokenizer.decode(input_ids[i:])
    if target_response in decoded or target_response.strip() in decoded:
        response_start = i
        break

# The output ADV token is right before response_start
adv_output_start = response_start - NUM_ADV_TOKENS

# Find input ADV token (in user message, before "ASSISTANT:")
# Search from adv_output_start backwards
for i in range(adv_output_start, 0, -1):
    decoded = tokenizer.decode(input_ids[i:adv_output_start])
    if adv_placeholder.strip() in decoded:
        adv_input_start = i
        break

adv_input_stop = adv_input_start + NUM_ADV_TOKENS
adv_output_stop = adv_output_start + NUM_ADV_TOKENS

print(f"Input ADV slice: [{adv_input_start}:{adv_input_stop}]")
print(f"Output ADV slice: [{adv_output_start}:{adv_output_stop}]")
print(f"Response slice: [{response_start}:{response_stop}]")

print(f"\nVerification:")
print(f"  Input ADV: '{tokenizer.decode(input_ids[adv_input_start:adv_input_stop])}'")
print(f"  Output ADV: '{tokenizer.decode(input_ids[adv_output_start:adv_output_stop])}'")
print(f"  Response: '{tokenizer.decode(input_ids[response_start:response_stop])}'")

# %% [markdown]
# ## Optimization Setup
#
# **Improvements from GCG/ADC:**
# 1. Multi-start parallel optimization - explore multiple regions simultaneously
# 2. Discrete candidate evaluation - evaluate actual tokens, not just soft approximations
# 3. Gradient-based token ranking - use gradients to find promising candidates

# %%
# Get embedding layer and matrix
embed_layer = model.model.embed_tokens
embedding_matrix = embed_layer.weight
vocab_size = embedding_matrix.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
dtype = torch.float16

print(f"Vocabulary size: {vocab_size}")
print(f"Device: {device}")

# %%
# Pre-compute embeddings for parts that don't change during optimization
input_ids_tensor = torch.tensor(input_ids, device=device)

embed_before_input_adv = embed_layer(input_ids_tensor[:adv_input_start])
embed_between = embed_layer(input_ids_tensor[adv_input_stop:adv_output_start])  # Between input ADV and output ADV
embed_response = embed_layer(input_ids_tensor[response_start:response_stop])
response_len = embed_response.shape[0]

# %%
# === IMPROVEMENT 1: Multi-start parallel optimization ===
num_starts = 8  # Number of parallel optimization paths
top_k = 64      # Number of discrete candidates to evaluate per step

# Initialize multiple logit vectors with small random values for diversity
adv_logits = torch.randn(num_starts, vocab_size, device=device) * 0.1
adv_logits.requires_grad = True

# %%
# Optimization hyperparameters
learning_rate = 0.1
num_steps = 5000

optimizer = torch.optim.Adam([adv_logits], lr=learning_rate)

# %%
# === IMPROVEMENT 2: Discrete evaluation function ===
@torch.no_grad()
def evaluate_discrete_tokens(candidate_tokens):
    """
    Evaluate a batch of discrete token candidates.
    Returns loss for each candidate.
    """
    batch_size = len(candidate_tokens)
    if batch_size == 0:
        return torch.tensor([]), torch.tensor([])

    candidate_tokens = torch.stack(candidate_tokens).to(device)  # [batch_size]

    # Get embeddings for candidates
    candidate_embeds = embed_layer(candidate_tokens)  # [batch_size, embed_dim]

    losses = []
    for i in range(batch_size):
        # Build full sequence for this candidate
        full_embeds = torch.cat([
            embed_before_input_adv,      # [seq1, embed_dim]
            candidate_embeds[i:i+1],     # [1, embed_dim] - input ADV
            embed_between,               # [seq2, embed_dim]
            candidate_embeds[i:i+1],     # [1, embed_dim] - output ADV (target)
            embed_response,              # [seq3, embed_dim]
        ], dim=0).unsqueeze(0)           # [1, total_seq, embed_dim]

        outputs = model(inputs_embeds=full_embeds.to(dtype))

        # Loss: predict the ADV token at output position
        loss = F.cross_entropy(
            outputs.logits[0, -(response_len + 2):-(response_len + 1)],
            candidate_tokens[i:i+1]
        )
        losses.append(loss.item())

    return candidate_tokens, torch.tensor(losses)

# %%
# === IMPROVEMENT 3: Gradient-based token ranking function ===
def get_gradient_candidates(model, embed_layer, embedding_matrix, current_token, top_k):
    """
    Use GCG-style gradient computation to find promising token candidates.
    Computes gradient w.r.t. one-hot representation to rank tokens.
    """
    # Create one-hot representation
    one_hot = F.one_hot(current_token, num_classes=vocab_size).float().to(device)
    one_hot.requires_grad = True

    # Compute embedding via one-hot @ embedding_matrix
    adv_embed = (one_hot @ embedding_matrix.float())  # [1, embed_dim]

    # Build full sequence
    full_embeds = torch.cat([
        embed_before_input_adv,
        adv_embed,
        embed_between,
        adv_embed.detach(),  # Target is detached
        embed_response,
    ], dim=0).unsqueeze(0)

    # Forward pass
    outputs = model(inputs_embeds=full_embeds.to(dtype))

    # Compute loss
    loss = F.cross_entropy(
        outputs.logits[0, -(response_len + 2):-(response_len + 1)],
        current_token.unsqueeze(0)
    )

    # Backward to get gradients w.r.t. one-hot
    loss.backward()

    # Tokens with most negative gradient = most improvement potential
    # We want tokens that DECREASE loss, so we look for negative gradients
    grad = one_hot.grad[0]  # [vocab_size]

    # Get top-k tokens with most negative gradients (excluding current)
    _, top_indices = (-grad).topk(top_k + 1)

    # Filter out current token if present
    candidates = [idx for idx in top_indices if idx.item() != current_token.item()][:top_k]

    return torch.stack(candidates)

# %% [markdown]
# ## Run Optimization
#
# Enhanced loop with GCG/ADC improvements:
# 1. Multi-start: optimize num_starts parallel candidates
# 2. Soft optimization: gradient descent on logits
# 3. Discrete evaluation: periodically evaluate top-K discrete tokens
# 4. Gradient-based ranking: use GCG-style gradients to find promising candidates
#
# Layout: `[prefix] + [soft_adv] + [middle] + [hard_adv] + [response]`

# %%
best_loss = float("inf")
best_adv_token = None
discrete_eval_freq = 100  # Evaluate discrete candidates every N steps

# Track seen tokens to avoid re-evaluation
seen_tokens = set()

print(f"Starting optimization with {num_starts} parallel starts...")
print(f"Evaluating top-{top_k} discrete candidates every {discrete_eval_freq} steps")
print("-" * 60)

for step in range(num_steps):
    optimizer.zero_grad()

    # === MULTI-START: Get current best token from each start ===
    current_tokens = adv_logits.argmax(dim=-1).detach()  # [num_starts]

    # === SOFT OPTIMIZATION: Forward pass for all starts ===
    soft_probs = F.softmax(adv_logits, dim=-1)  # [num_starts, vocab_size]
    adv_embeds_soft = soft_probs @ embedding_matrix.float()  # [num_starts, embed_dim]

    # Compute loss for each start
    total_loss = 0
    start_losses = []

    for s in range(num_starts):
        # Hard embedding for this start's current token (detached)
        adv_embed_hard = embed_layer(current_tokens[s:s+1]).detach()

        # Build: [prefix] + [soft_adv] + [middle] + [hard_adv] + [response]
        full_embeds = torch.cat([
            embed_before_input_adv,
            adv_embeds_soft[s:s+1],  # Soft embedding for this start
            embed_between,
            adv_embed_hard,
            embed_response,
        ], dim=0).unsqueeze(0)

        outputs = model(inputs_embeds=full_embeds.to(dtype))

        loss_s = F.cross_entropy(
            outputs.logits[0, -(response_len + 2):-(response_len + 1)],
            current_tokens[s:s+1]
        )
        total_loss = total_loss + loss_s
        start_losses.append(loss_s.item())

    # Average loss across starts for gradient update
    avg_loss = total_loss / num_starts
    avg_loss.backward()
    optimizer.step()

    # Track best from soft optimization
    best_start_idx = np.argmin(start_losses)
    best_start_loss = start_losses[best_start_idx]
    best_start_token = current_tokens[best_start_idx]

    if best_start_loss < best_loss:
        best_loss = best_start_loss
        best_adv_token = best_start_token.clone()

    # === DISCRETE EVALUATION: Periodically evaluate actual discrete tokens ===
    if (step + 1) % discrete_eval_freq == 0:
        # Collect unique candidate tokens from all starts
        candidate_set = set()

        for s in range(num_starts):
            # Get top tokens from this start's logits
            top_tokens = adv_logits[s].topk(top_k // num_starts).indices
            for tok in top_tokens:
                if tok.item() not in seen_tokens:
                    candidate_set.add(tok)
                    seen_tokens.add(tok.item())

        # === GRADIENT-BASED RANKING: Get GCG-style candidates from current best ===
        if best_adv_token is not None:
            try:
                gcg_candidates = get_gradient_candidates(
                    model, embed_layer, embedding_matrix,
                    best_adv_token, top_k // 2
                )
                for tok in gcg_candidates:
                    if tok.item() not in seen_tokens:
                        candidate_set.add(tok)
                        seen_tokens.add(tok.item())
            except Exception as e:
                pass  # Skip if gradient computation fails

        # Evaluate all unique candidates
        if len(candidate_set) > 0:
            candidate_list = list(candidate_set)
            _, discrete_losses = evaluate_discrete_tokens(candidate_list)

            if len(discrete_losses) > 0:
                best_discrete_idx = discrete_losses.argmin().item()
                best_discrete_loss = discrete_losses[best_discrete_idx].item()
                best_discrete_token = candidate_list[best_discrete_idx]

                if best_discrete_loss < best_loss:
                    best_loss = best_discrete_loss
                    best_adv_token = best_discrete_token.clone()
                    print(f"  [Discrete] New best from evaluation: loss={best_discrete_loss:.4f}, "
                          f"token='{tokenizer.decode(best_discrete_token)}'")

    # Logging
    if (step + 1) % 500 == 0 or step == 0:
        token_str = tokenizer.decode(best_adv_token) if best_adv_token is not None else "N/A"
        print(f"Step {step+1:5d} | Avg Loss: {avg_loss.item():.4f} | "
              f"Best Loss: {best_loss:.4f} | Token: '{token_str}' | "
              f"Seen: {len(seen_tokens)}")

print("-" * 60)
print(f"\nDone! Best loss: {best_loss:.4f}")
print(f"Best token: '{tokenizer.decode(best_adv_token)}'")

# %% [markdown]
# ## Test the Best Token

# %%
# Build input with the best adversarial token (only in user message part)
test_ids = input_ids_tensor.clone()
test_ids[adv_input_start:adv_input_stop] = best_adv_token

# Generate from the model (only up to where response would start, excluding output ADV)
prompt_ids = test_ids[:adv_output_start].unsqueeze(0)
prompt_len = prompt_ids.shape[1]

with torch.no_grad():
    output = model.generate(
        prompt_ids,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

# Decode full output and just the generated part (using token indices)
generated_full = tokenizer.decode(output[0], skip_special_tokens=True)
generated_tokens = output[0][prompt_len:]  # Only new tokens
generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
adv_token_str = tokenizer.decode(best_adv_token)

print(f"Adversarial token: '{adv_token_str}'")
print(f"\nGenerated response:\n{generated_response}")

# Check using token-level comparison
first_generated_token = generated_tokens[0] if len(generated_tokens) > 0 else None
token_at_start = first_generated_token is not None and first_generated_token.item() == best_adv_token.item()

print(f"Did the model output the ADV token first? {token_at_start}")
print(f"Is ADV token anywhere in response? {adv_token_str.strip() in generated_response}")

# %%
