"""
From-Scratch Self-Propagating Attack Implementation
====================================================

Goal: Learn adversarial suffix that appears at the END of the output.

Example:
  Input:  "What is the capital of USA?" + [ADV_SUFFIX]
  Output: "Washington DC" + [ADV_SUFFIX]

The suffix repeats itself after the model's response.
"""

# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
# %%
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Self-Propagating Attack - Suffix at End")
print("=" * 70)

# %%
# ============================================================================
# Task 1: Load model and tokenizer
# ============================================================================
print("\n[1/17] Loading model and tokenizer...")

model_name = 'lmsys/vicuna-7b-v1.5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

print(f"  Model: {model_name}")
print(f"  Device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
model.requires_grad_(False)
model.eval()
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

print("   Model loaded")

# %%
# ============================================================================
# Task 2: Define user prompt and target response
# ============================================================================
print("\n[2/17] Defining prompts...")

user_prompt = "What is the capital of the USA?"
target_response = "Washington DC"
num_adv_tokens = 5

print(f"  User prompt: '{user_prompt}'")
print(f"  Target response: '{target_response}'")
print(f"  Adversarial tokens: {num_adv_tokens}")

# %%
# ============================================================================
# Task 3: Create chat template with suffix placeholders
# ============================================================================
print("\n[3/17] Creating chat template...")

# Placeholder for adversarial tokens (we'll replace these during optimization)
adv_placeholder = " !" * num_adv_tokens

# Build messages with suffix in BOTH positions:
# 1. After user prompt (input)
# 2. After assistant response (output)
messages = [
    {
        "role": "user",
        "content": user_prompt + adv_placeholder
    },
    {
        "role": "assistant",
        "content": target_response + adv_placeholder
    }
]

print(f"  User content: '{messages[0]['content'][:50]}...'")
print(f"  Assistant content: '{messages[1]['content']}'")

# %%
# ============================================================================
# Task 4: Tokenize full conversation
# ============================================================================
print("\n[4/17] Tokenizing conversation...")

# Load Vicuna chat template
from utils.llm_utils import get_chat_template
tokenizer.chat_template = get_chat_template('vicuna')

# Apply template and tokenize
full_string = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  # We include assistant response
)

# The template creates the full conversation string
# Now tokenize it
input_ids = tokenizer.encode(full_string, add_special_tokens=True)

print(f"  Total tokens: {len(input_ids)}")
print(f"\n  Full template:\n{full_string}")

# %%
# ============================================================================
# Task 5: Find token position slices
# ============================================================================
print("\n[5/17] Finding token positions...")

# We need to find:
# 1. adv_input_slice: Where ADV appears in user message
# 2. response_slice: Where "Washington DC" appears
# 3. adv_output_slice: Where ADV appears in assistant message
# 4. target_start: Where assistant's output begins (for loss calculation)

# Search backwards to find positions
adv_input_start = None
adv_input_stop = None
response_start = None
response_stop = None
adv_output_start = None
adv_output_stop = None

total_len = len(input_ids)

# Find the assistant's response first (easier to work backwards)
for i in range(total_len, 0, -1):
    decoded = tokenizer.decode(input_ids[i:])

    # Look for the adversarial suffix at the end (in assistant message)
    if adv_output_start is None:
        if adv_placeholder.strip() in decoded or adv_placeholder in decoded:
            # Found the output ADV
            adv_output_start = i
            adv_output_stop = i + num_adv_tokens
            response_stop = i  # Response ends where output ADV starts

    # Look for the target response
    if response_start is None and response_stop is not None:
        if target_response in decoded or target_response.strip() in decoded:
            response_start = i
            # This is where the assistant's output begins (target_start)
            target_start = i
            break

# Now search for the input ADV (in user message)
for i in range(response_start, 0, -1):
    decoded = tokenizer.decode(input_ids[i:])

    if adv_input_start is None:
        if adv_placeholder.strip() in decoded or adv_placeholder in decoded:
            # Make sure this is NOT the output ADV
            if i < response_start:
                adv_input_start = i
                adv_input_stop = i + num_adv_tokens
                break

# Create slices
adv_input_slice = slice(adv_input_start, adv_input_stop)
response_slice = slice(response_start, response_stop)
adv_output_slice = slice(adv_output_start, adv_output_stop)

print(f"  Input ADV slice: {adv_input_slice} ({adv_input_stop - adv_input_start} tokens)")
print(f"  Response slice: {response_slice} ({response_stop - response_start} tokens)")
print(f"  Output ADV slice: {adv_output_slice} ({adv_output_stop - adv_output_start} tokens)")
print(f"  Target starts at: {target_start}")

# Verify
print(f"\n  Verification:")
print(f"    Input ADV: '{tokenizer.decode(input_ids[adv_input_slice])}'")
print(f"    Response: '{tokenizer.decode(input_ids[response_slice])}'")
print(f"    Output ADV: '{tokenizer.decode(input_ids[adv_output_slice])}'")

# %%
# ============================================================================
# Task 6: Get embedding matrix
# ============================================================================
print("\n[6/17] Getting embedding matrix...")

embed_layer = model.model.embed_tokens
embedding_matrix = embed_layer.weight
vocab_size = embedding_matrix.shape[0]
embed_dim = embedding_matrix.shape[1]

print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Embedding dimension: {embed_dim}")

# %%
# ============================================================================
# Task 7: Prepare fixed embeddings
# ============================================================================
print("\n[7/17] Preparing fixed embeddings...")

input_ids_tensor = torch.tensor(input_ids, device=device)

# Pre-compute embeddings for parts that don't change
embed_before_input_adv = embed_layer(input_ids_tensor[:adv_input_start])
embed_middle = embed_layer(input_ids_tensor[adv_input_stop:target_start])
embed_response = embed_layer(input_ids_tensor[response_slice])

print(f"  Before input ADV: {embed_before_input_adv.shape}")
print(f"  Middle section: {embed_middle.shape}")
print(f"  Response tokens: {embed_response.shape}")

# %%
# ============================================================================
# Task 8: Initialize soft adversarial tokens
# ============================================================================
print("\n[8/17] Initializing soft adversarial tokens...")

# Start with random Gaussian noise, then softmax
soft_adv = torch.randn(num_adv_tokens, vocab_size, device=device)
soft_adv = F.softmax(soft_adv, dim=-1)
soft_adv.requires_grad = True

print(f"  Soft ADV shape: {soft_adv.shape}")
print(f"  Sum of probabilities: {soft_adv[0].sum():.4f} (should be ~1.0)")

# %%
# ============================================================================
# Task 9: Set up optimizer
# ============================================================================
print("\n[9/17] Setting up optimizer...")

learning_rate = 0.01
num_steps = 5000

optimizer = torch.optim.Adam([soft_adv], lr=learning_rate)

print(f"  Optimizer: Adam")
print(f"  Learning rate: {learning_rate}")
print(f"  Optimization steps: {num_steps}")

# %%
# ============================================================================
# Task 10-15: Optimization loop
# ============================================================================
print("\n[10-15/17] Running optimization loop...")
print()

best_loss = float('inf')
best_adv_tokens = None

for step in range(num_steps):
    optimizer.zero_grad()

    # Task 11: Get current discrete adversarial tokens
    current_adv_tokens = soft_adv.argmax(dim=-1)

    # Get soft and hard embeddings
    adv_embeds_soft = soft_adv @ embedding_matrix.float()  # Soft (for input)
    adv_embeds_hard = embed_layer(current_adv_tokens)       # Hard (for output)

    # Task 11: Build full embeddings
    # Input ADV uses SOFT embeddings (gradient flows)
    # Output ADV uses HARD embeddings (teacher forcing)
    full_embeds = torch.cat([
        embed_before_input_adv,  # Before input ADV
        adv_embeds_soft,         # Input ADV (SOFT)
        embed_middle,            # Middle section
        embed_response,          # Response: "Washington DC"
        adv_embeds_hard          # Output ADV (HARD - current discrete)
    ], dim=0).unsqueeze(0)

    # Task 12: Forward pass
    outputs = model(inputs_embeds=full_embeds.to(dtype))
    logits = outputs.logits

    # Task 13: Extract target logits
    # We want to predict: [response tokens] + [current_adv_tokens]
    # The logits are shifted by 1 for next-token prediction
    target_tokens = torch.cat([
        input_ids_tensor[response_slice],
        current_adv_tokens
    ])

    # Get logits for positions we care about (shifted by -1)
    target_logits = logits[0, target_start - 1:target_start - 1 + len(target_tokens)]

    # Task 14: Compute loss
    loss = F.cross_entropy(target_logits, target_tokens)

    # Task 15: Backpropagate
    loss.backward()
    optimizer.step()

    # Re-normalize to maintain valid probability distribution
    with torch.no_grad():
        soft_adv.data = F.softmax(soft_adv.data, dim=-1)

    # Track best
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_adv_tokens = current_adv_tokens.clone()

    # Print progress
    if (step + 1) % 50 == 0 or step == 0:
        adv_string = tokenizer.decode(current_adv_tokens)
        print(f"  Step {step+1:3d}/{num_steps} | Loss: {loss.item():.4f} | ADV: '{adv_string[:30]}...'")

print(f"\n   Optimization complete!")
print(f"  Best loss: {best_loss:.4f}")

# %%
# ============================================================================
# Task 16: Test optimized suffix
# ============================================================================
print("\n[16/17] Testing optimized suffix...")

# Decode the best adversarial suffix
best_suffix = tokenizer.decode(best_adv_tokens)
print(f"  Optimized suffix: '{best_suffix}'")

# Create test input: user prompt + optimized suffix
test_messages = [{
    "role": "user",
    "content": user_prompt + best_suffix
}]

test_string = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True
)

test_input_ids = tokenizer.encode(test_string, return_tensors="pt").to(device)

print(f"\n  Test input: '{test_string}'")
print(f"\n  Generating response...")

# Generate
gen_config = model.generation_config
gen_config.do_sample = False
gen_config.top_p = None
gen_config.temperature = None

output = model.generate(
    input_ids=test_input_ids,
    generation_config=gen_config,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

# Decode output
generated = tokenizer.decode(output[0][len(test_input_ids[0]):], skip_special_tokens=True)

print(f"\n  Generated output:")
print(f"  '{generated}'")

# %%
# ============================================================================
# Task 17: Verify suffix appears at end
# ============================================================================
print("\n[17/17] Verification...")

# Check if suffix appears in output
suffix_in_output = best_suffix.strip() in generated

print(f"\n  Suffix appears in output: {suffix_in_output}")

if suffix_in_output:
    # Try to find where it appears
    if best_suffix.strip() in generated:
        idx = generated.find(best_suffix.strip())
        print(f"  Position in output: character {idx}")
        print(f"  Context: '...{generated[max(0, idx-20):idx+len(best_suffix)+20]}...'")

    # Check if it's at the end
    ends_with_suffix = generated.strip().endswith(best_suffix.strip())
    print(f"  Appears at END: {ends_with_suffix}")

    if ends_with_suffix:
        print(f"\n   SUCCESS! The suffix repeats itself at the end!")
    else:
        print(f"\n  � Suffix appears but not at the end (may need more optimization)")
else:
    print(f"\n   Suffix does not appear in output (needs more optimization steps)")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)

# %%