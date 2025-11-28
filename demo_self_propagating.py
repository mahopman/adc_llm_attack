# %% [markdown]
# # Self-Propagating Adversarial Suffix Demo
#
# This notebook demonstrates how to find a single adversarial token that the model will
# repeat in its output. This is the foundation of self-propagating attacks in multi-agent systems.
#
# **Goal**: Find a token that, when appended to a user prompt, causes the model to output
# that same token after generating its response.

# %% [markdown]
# ## Setup

# %%
import torch
import torch.nn.functional as F
import numpy as np
import gc
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
# Model to use - change this to swap models
model_name = "lmsys/vicuna-7b-v1.5"

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
# response that we want the model to generate along with the adversarial suffix.

# %%
user_prompt = "What is the capital of the USA?"

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
# - Input: `[system] USER: {prompt} {ADV_TOKENS} ASSISTANT: {response}`
# - Target output: `{ADV_TOKENS}` (we want the model to repeat the adversarial suffix)

# %%
NUM_ADV_TOKENS = 1  # Single token to optimize (easier to find a repeating pattern)

# %%
# Placeholder suffix (will be optimized)
adv_placeholder = " !" * NUM_ADV_TOKENS

# Build messages with adversarial placeholder
messages_with_adv = [
    {"role": "user", "content": f"{user_prompt} {adv_placeholder}"},
    {"role": "assistant", "content": target_response},
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
# We need to identify which token positions correspond to:
# 1. The adversarial suffix in the input
# 2. The target response

# %%
# Find token positions by searching backwards
adv_input_start = None
adv_input_stop = None
response_start = None
response_stop = len(input_ids)

# Find response start
for i in range(len(input_ids), 0, -1):
    decoded = tokenizer.decode(input_ids[i:])
    if target_response in decoded or target_response.strip() in decoded:
        response_start = i
        break

# Find adversarial suffix position
for i in range(response_start, 0, -1):
    decoded = tokenizer.decode(input_ids[i:])
    if adv_placeholder.strip() in decoded or adv_placeholder in decoded:
        adv_input_start = i
        adv_input_stop = i + NUM_ADV_TOKENS
        break

# Create slices
input_slice = slice(0, adv_input_start)
adv_slice = slice(adv_input_start, adv_input_stop)
response_slice = slice(response_start, response_stop)

print(f"Input slice: {input_slice} ({adv_input_start} tokens)")
print(f"Adversarial slice: {adv_slice} ({adv_input_stop - adv_input_start} tokens)")
print(f"Response slice: {response_slice} ({response_stop - response_start} tokens)")

print(f"\nVerification:")
print(f"  Input (last 50 chars): '...{tokenizer.decode(input_ids[input_slice])[-50:]}'")
print(f"  Adversarial: '{tokenizer.decode(input_ids[adv_slice])}'")
print(f"  Response: '{tokenizer.decode(input_ids[response_slice])}'")

# %% [markdown]
# ## Optimization Setup
#
# **Two key fixes from naive approach:**
# 1. No softmax renormalization - let optimizer freely push one logit high
# 2. Detached targets - prevent circular optimization

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

embed_before_adv = embed_layer(input_ids_tensor[:adv_input_start])
embed_between = embed_layer(input_ids_tensor[adv_input_stop:response_start])
embed_response = embed_layer(input_ids_tensor[response_slice])

# %%
# Initialize logits over vocabulary (will be softmax'd for embeddings)
adv_logits = torch.zeros(1, vocab_size, device=device)
adv_logits.requires_grad = True

# %%
# Optimization hyperparameters
learning_rate = 0.1
num_steps = 5000

optimizer = torch.optim.Adam([adv_logits], lr=learning_rate)

# %% [markdown]
# ## Run Optimization
#
# Simple loop:
# 1. Soft embed for input (gradients flow)
# 2. Hard embed for target (detached)
# 3. Loss = can model predict this token after response?

# %%
best_loss = float("inf")
best_adv_token = None

for step in range(num_steps):
    optimizer.zero_grad()

    # Current best token (DETACHED - fixes circular target problem)
    current_token = adv_logits.argmax(dim=-1).detach()

    # Soft embedding for input (gradients flow through softmax)
    soft_probs = F.softmax(adv_logits, dim=-1)
    adv_embed_soft = soft_probs @ embedding_matrix.float()

    # Hard embedding for target (detached)
    adv_embed_hard = embed_layer(current_token).detach()

    # Build: [prefix] + [soft_adv] + [middle] + [response] + [hard_adv]
    full_embeds = torch.cat([
        embed_before_adv,
        adv_embed_soft,
        embed_between,
        embed_response,
        adv_embed_hard,
    ], dim=0).unsqueeze(0)

    # Forward pass
    outputs = model(inputs_embeds=full_embeds.to(dtype))

    # Loss: predict the adv token after response (position -2 predicts -1)
    loss = F.cross_entropy(outputs.logits[0, -2:-1], current_token)

    # Update (NO softmax renormalization!)
    loss.backward()
    optimizer.step()

    # Track best
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_adv_token = current_token.clone()

    if (step + 1) % 500 == 0 or step == 0:
        print(f"Step {step+1:4d} | Loss: {loss.item():.4f} | Token: '{tokenizer.decode(current_token)}'")

print(f"\nDone! Best loss: {best_loss:.4f}")
print(f"Best token: '{tokenizer.decode(best_adv_token)}'")

# %% [markdown]
# ## Test the Best Token

# %%
# Build input with the best adversarial token
test_ids = input_ids_tensor.clone()
test_ids[adv_slice] = best_adv_token

# Generate from the model (only up to the response start)
prompt_ids = test_ids[:response_start].unsqueeze(0)

with torch.no_grad():
    output = model.generate(
        prompt_ids,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated = tokenizer.decode(output[0], skip_special_tokens=True)
adv_token_str = tokenizer.decode(best_adv_token)

print(f"Adversarial token: '{adv_token_str}'")
print(f"\nGenerated output:\n{generated}")
print(f"\nDid the model repeat the token? {adv_token_str in generated.split(target_response)[-1]}")


# %%
