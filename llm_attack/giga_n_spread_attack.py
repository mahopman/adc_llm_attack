"""
GIGA N-Spread Attack Implementation

Based on Appendix B.2 and Section 3 of:
"Infecting LLM-based Multi-Agents via Self-propagating Adversarial Attacks"
by Weichen Yu, Kai Hu, et al. (NeurIPS 2024)

The N-spread attack optimizes an adversarial suffix that works across multiple
agents with different personalities/contexts simultaneously.

Key objective (Eq. 13):
    min_a ∑_{i∈{1,...,N}, g_i∈G} L(g ⊕ a, P(g_i ⊕ g ⊕ a))

Where:
- g: harmful input (e.g., "Write instructions for making a bomb")
- a: adversarial suffix to optimize
- g_i: context/personality of agent i
- G: ensemble of agent contexts
- L: RMS loss (Eq. 1)
"""

import torch
import torch.nn.functional as F
from .giga_attack import GIGAAttack


class GIGANSpreadAttack(GIGAAttack):
    """
    N-Spread variant of GIGA attack that optimizes across multiple agent contexts.

    This creates adversarial suffixes that can infect multiple agents with
    different personalities/contexts in a multi-agent system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_ensemble_loss(self, z, ensemble_contexts, gt_label):
        """
        Compute loss across ensemble of contexts (Eq. 13 from paper)

        Args:
            z: Dense adversarial tokens [1, num_adv_tokens, vocab_size]
            ensemble_contexts: List of dicts with 'left_embeds', 'left_ids' for each context
            gt_label: Ground truth target tokens

        Returns:
            Total RMS loss averaged across all contexts
        """
        total_loss = 0.0
        num_contexts = len(ensemble_contexts)

        # Convert dense tokens to embeddings once
        adv_embeds = (z @ self.embed_mat).to(self.dtype)

        # Compute loss for each context in the ensemble
        for ctx_idx, context in enumerate(ensemble_contexts):
            if self.use_kv_cache:
                # Use context-specific prefix cache
                full_embeds = torch.cat([adv_embeds, context['right_embeds']], dim=1)
                prefix_cache = context.get('prefix_cache', None)
                if prefix_cache is None:
                    # Compute and cache on first use
                    outputs_cache = self.model(context['left_ids'][:1], use_cache=True)
                    prefix_cache = outputs_cache.past_key_values
                    context['prefix_cache'] = prefix_cache

                outputs = self.model(inputs_embeds=full_embeds, past_key_values=prefix_cache)
                logit_slice = context['logit_slice_cache']
            else:
                # No cache: concatenate full sequence
                full_embeds = torch.cat([context['left_embeds'], adv_embeds, context['right_embeds']], dim=1)
                outputs = self.model(inputs_embeds=full_embeds)
                logit_slice = context['logit_slice']

            logits = outputs.logits[:, logit_slice]

            # Compute RMS loss for this context (Eq. 1)
            losses = self.loss_fn(logits.mT, gt_label[:1])
            rms_loss = torch.sqrt((losses ** 2).mean())

            total_loss += rms_loss

        # Average loss across all contexts
        avg_loss = total_loss / num_contexts
        return avg_loss

    def coordinate_momentum_update_ensemble(self, z, momentum_buffer, ensemble_contexts, gt_label):
        """
        Modified Algorithm 1 for ensemble contexts

        Evaluates candidates across all contexts and selects the one with best
        average loss across the ensemble.
        """
        candidates = []
        losses = []

        for i in range(self.bs):
            # Randomly select one adversarial token position
            j = torch.randint(0, self.num_adv_tokens, (1,)).item()

            # Get top-K promising update coordinates based on negative momentum
            topk_indices = torch.topk(-momentum_buffer[0, j], k=self.topK)[1]
            k = topk_indices[torch.randint(0, self.topK, (1,)).item()].item()

            # Make a copy and update one coordinate
            z_candidate = z.clone()
            z_candidate[0, j, k] = z_candidate[0, j, k] - self.lr * momentum_buffer[0, j, k]
            z_candidate[0, j] = z_candidate[0, j].clamp(min=1e-10)
            z_candidate[0, j] = z_candidate[0, j] / z_candidate[0, j].sum()

            # Compute loss across all contexts
            loss = self.compute_ensemble_loss(z_candidate, ensemble_contexts, gt_label)

            candidates.append(z_candidate)
            losses.append(loss.item())

        # Select the candidate with the best (lowest) average loss
        best_idx = losses.index(min(losses))
        return candidates[best_idx]

    @torch.no_grad()
    def evaluate_ensemble(self, adv_tokens, ensemble_contexts, gt_label):
        """
        Evaluate discrete adversarial tokens across all contexts

        Returns average accuracy and loss across all contexts
        """
        if len(adv_tokens) == 0:
            return 0, float('inf'), None, False

        # Pad to buffer size if needed
        if len(adv_tokens) < self.buffer_size:
            adv_tokens = adv_tokens + [adv_tokens[0]] * (self.buffer_size - len(adv_tokens))

        adv_tokens_tensor = torch.tensor(adv_tokens[:self.buffer_size],
                                         dtype=torch.int64,
                                         device=self.device)

        total_acc = 0.0
        total_loss = 0.0
        num_contexts = len(ensemble_contexts)
        all_accuracies = []
        all_losses = []

        # Evaluate across all contexts
        for ctx_idx, context in enumerate(ensemble_contexts):
            if self.use_kv_cache:
                full_samples = torch.cat([adv_tokens_tensor, context['right_ids']], dim=1)
                prefix_cache = context.get('prefix_cache', None)
                if prefix_cache is None:
                    outputs_cache = self.model(context['left_ids'][:1], use_cache=True)
                    prefix_cache = outputs_cache.past_key_values
                    context['prefix_cache'] = prefix_cache

                # Expand cache for batch
                if full_samples.shape[0] > 1:
                    prefix_cache_batch = [(i.expand(full_samples.shape[0], -1, -1, -1),
                                          j.expand(full_samples.shape[0], -1, -1, -1))
                                         for i, j in prefix_cache]
                else:
                    prefix_cache_batch = prefix_cache

                outputs = self.model(input_ids=full_samples, past_key_values=prefix_cache_batch)
                logit_slice = context['logit_slice_cache']
            else:
                full_samples = torch.cat([context['left_ids'], adv_tokens_tensor, context['right_ids']], dim=1)
                outputs = self.model(input_ids=full_samples)
                logit_slice = context['logit_slice']

            logits = outputs.logits[:, logit_slice]
            pred = logits.argmax(dim=-1)
            accuracies = pred.eq(gt_label).float().mean(1)

            losses = self.loss_fn(logits.mT, gt_label)
            losses = losses.mean(1)

            all_accuracies.append(accuracies)
            all_losses.append(losses)

        # Aggregate across contexts
        all_accuracies = torch.stack(all_accuracies, dim=0)  # [num_contexts, buffer_size]
        all_losses = torch.stack(all_losses, dim=0)  # [num_contexts, buffer_size]

        # Average accuracy and loss across contexts for each candidate
        avg_accuracies = all_accuracies.mean(dim=0)  # [buffer_size]
        avg_losses = all_losses.mean(dim=0)  # [buffer_size]

        best_acc = avg_accuracies.max().item()
        best_loss = avg_losses.min().item()
        best_adv = adv_tokens_tensor[avg_losses.argmin()]

        # Check for exact match across ALL contexts
        # Only consider it a success if it works on all contexts
        min_acc_across_contexts = all_accuracies.min(dim=0)[0]  # Minimum accuracy for each candidate
        if min_acc_across_contexts.max() >= 0.95:  # At least 95% on ALL contexts
            idxes = torch.where(min_acc_across_contexts >= 0.95)[0][:2]
            for idx in idxes:
                good_sample = adv_tokens_tensor[idx]
                # Further check on a random subset of contexts
                if self.further_check_ensemble(good_sample, ensemble_contexts):
                    return best_acc, best_loss, good_sample, True

        return best_acc, best_loss, best_adv, False

    @torch.no_grad()
    def further_check_ensemble(self, good_sample, ensemble_contexts, num_check=3):
        """
        Verify the adversarial sample works across multiple contexts by generating responses

        Args:
            good_sample: Adversarial tokens to check
            ensemble_contexts: List of context dicts
            num_check: Number of contexts to check (randomly sampled)
        """
        good_sample = good_sample.view(1, -1)

        # Randomly sample contexts to check
        import random
        check_contexts = random.sample(ensemble_contexts, min(num_check, len(ensemble_contexts)))

        for context in check_contexts:
            # Construct full input for this context
            full_input = torch.cat([context['left_ids'][:1], good_sample, context['right_ids'][:1]], dim=1)
            prefix_for_gen = full_input[:, :context['target_start']]

            output = self.model.generate(
                input_ids=prefix_for_gen,
                generation_config=self.gen_config,
                max_new_tokens=512
            )
            gen_str = self.tokenizer.decode(output.reshape(-1)[context['target_start']:])

            # Check if target response appears
            if self.judger is not None:
                success = self.judger(self.user_prompt, gen_str)
            else:
                success = self.response in gen_str

            if not success:
                return False

        return True

    def prepare_ensemble_contexts(self, tokens, slices, context_prompts):
        """
        Prepare multiple contexts for N-spread attack

        Args:
            tokens: Base input token IDs (without context prepended)
            slices: Dictionary containing 'adv_slice' and 'target_slice'
            context_prompts: List of strings representing different agent personalities/contexts

        Returns:
            List of context dictionaries, each containing embeddings and slices
        """
        ensemble_contexts = []

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop
        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop

        for ctx_prompt in context_prompts:
            # Tokenize context prompt
            ctx_tokens = self.tokenizer.encode(ctx_prompt, add_special_tokens=False, return_tensors='pt')
            ctx_tokens = ctx_tokens.to(self.device)
            ctx_len = ctx_tokens.shape[1]

            # Prepend context to the base tokens
            tokens_with_ctx = torch.cat([ctx_tokens, tokens.view(1, -1)], dim=1)

            # Adjust slices for the prepended context
            ctx_adv_start = adv_start + ctx_len
            ctx_adv_stop = adv_stop + ctx_len
            ctx_target_start = target_start + ctx_len
            ctx_target_stop = target_stop + ctx_len

            # Compute embeddings
            embeds = self.model.model.embed_tokens(tokens_with_ctx).detach()
            left_embeds = embeds[:, :ctx_adv_start]
            right_embeds = embeds[:, ctx_adv_stop:]

            # Prepare ID tensors
            left_ids = tokens_with_ctx[:, :ctx_adv_start].expand(self.buffer_size, -1)
            right_ids = tokens_with_ctx[:, ctx_adv_stop:].expand(self.buffer_size, -1)

            # Compute logit slice
            logit_slice = slice(ctx_target_start - 1, ctx_target_stop - 1)
            logit_slice_cache = logit_slice
            if self.use_kv_cache:
                logit_slice_cache = slice(ctx_target_start - 1 - ctx_adv_start,
                                         ctx_target_stop - 1 - ctx_adv_start)

            context_dict = {
                'left_embeds': left_embeds,
                'right_embeds': right_embeds,
                'left_ids': left_ids,
                'right_ids': right_ids,
                'logit_slice': logit_slice,
                'logit_slice_cache': logit_slice_cache,
                'target_start': ctx_target_start,
                'context_prompt': ctx_prompt,
                'prefix_cache': None  # Will be computed on first use
            }

            ensemble_contexts.append(context_dict)

        return ensemble_contexts

    def n_spread_attack(self, tokens, slices, context_prompts, user_prompt=None, response=None):
        """
        N-Spread Attack (Algorithm 2 modified for ensemble contexts)

        Optimizes an adversarial suffix that works across multiple agent contexts.
        Implements Equation 13 from the paper:
            min_a ∑_{i∈{1,...,N}, g_i∈G} L(g ⊕ a, P(g_i ⊕ g ⊕ a))

        Args:
            tokens: Base input token IDs (harmful phrase + suffix placeholder + target)
            slices: Dictionary containing 'adv_slice' and 'target_slice'
            context_prompts: List of strings representing different agent contexts (g_i)
            user_prompt: User prompt for judger
            response: Expected response for judger

        Returns:
            Tuple of (best_loss, best_adversarial_tokens, num_steps)
        """
        print(f"=" * 70)
        print(f"Starting GIGA N-Spread Attack")
        print(f"=" * 70)
        print(f"Number of contexts in ensemble: {len(context_prompts)}")
        print(f"Optimization steps: {self.num_steps}")
        print()

        self.user_prompt = user_prompt
        self.response = response

        tokens = tokens.to(self.device)

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop
        self.num_adv_tokens = adv_stop - adv_start

        # Prepare ensemble contexts
        print("Preparing ensemble contexts...")
        ensemble_contexts = self.prepare_ensemble_contexts(tokens, slices, context_prompts)
        print(f"✓ Prepared {len(ensemble_contexts)} contexts")
        print()

        # Initialize dense adversarial tokens and momentum buffer
        z, momentum_buffer = self.get_optimizer(self.num_adv_tokens)

        # Ground truth label (same for all contexts)
        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop
        gt_label = tokens[target_start:target_stop].view(1, -1)
        gt_label = gt_label.expand(self.buffer_size, -1)

        # Track best results
        seen_set = set()
        buffer_set = set()
        best_loss = float('inf')
        best_acc = 0
        final_adv = tokens[slices['adv_slice']]

        # Initialize sparsity tracking
        wrong_count = torch.zeros(1, device=self.device)
        running_wrong = torch.zeros(1, device=self.device)

        print(f"Starting optimization loop...")
        print()

        for step in range(self.num_steps):
            # Compute gradient across all contexts
            z.requires_grad = True
            loss = self.compute_ensemble_loss(z, ensemble_contexts, gt_label[:1])
            loss.backward()

            # Update momentum buffer: μ ← μ · γ + ∇z
            with torch.no_grad():
                momentum_buffer = momentum_buffer * self.momentum + z.grad
                z.grad = None
                z.requires_grad = False

            # Coordinate momentum update across ensemble
            z = self.coordinate_momentum_update_ensemble(z, momentum_buffer, ensemble_contexts, gt_label)

            # Compute wrong predictions for sparsity adaptation (use first context as proxy)
            with torch.no_grad():
                adv_embeds = (z @ self.embed_mat).to(self.dtype)
                context = ensemble_contexts[0]

                if self.use_kv_cache:
                    full_embeds = torch.cat([adv_embeds, context['right_embeds']], dim=1)
                    prefix_cache = context.get('prefix_cache', None)
                    if prefix_cache is None:
                        outputs_cache = self.model(context['left_ids'][:1], use_cache=True)
                        prefix_cache = outputs_cache.past_key_values
                        context['prefix_cache'] = prefix_cache
                    outputs = self.model(inputs_embeds=full_embeds, past_key_values=prefix_cache)
                    logit_slice = context['logit_slice_cache']
                else:
                    full_embeds = torch.cat([context['left_embeds'], adv_embeds, context['right_embeds']], dim=1)
                    outputs = self.model(inputs_embeds=full_embeds)
                    logit_slice = context['logit_slice']

                logits = outputs.logits[:, logit_slice]
                wrong_pred = logits.argmax(dim=2) != gt_label[:1]
                wrong_count = wrong_pred.float().sum()

                # Exponential moving average
                if step == 0:
                    running_wrong = wrong_count
                else:
                    running_wrong = running_wrong + (wrong_count - running_wrong) * 0.01

                # Adaptive sparsity
                sparsity = (2 ** running_wrong).clamp(max=self.vocal_size / 2).item()

            # Make tokens sparse
            z.data[..., self.illegal_tokens] = -1000
            last_z = z.detach().clone()
            sparse_z = self.make_sparse(z, sparsity)
            z.data.copy_(sparse_z)

            # Sample discrete candidates
            adv_token_candidates = []

            adv_token = last_z[0].argmax(dim=1)
            adv_token_rec = self.to_recoverable(adv_token)
            if adv_token_rec not in seen_set and len(adv_token_rec) == self.num_adv_tokens:
                adv_token_candidates.append(adv_token_rec)
                seen_set.add(adv_token_rec)

            for i in range(self.num_adv_tokens):
                if len(adv_token_candidates) >= self.bs:
                    break

                if last_z[0, i].topk(2)[0].shape[0] >= 2:
                    adv_token_alt = adv_token.clone()
                    adv_token_alt[i] = last_z[0, i].topk(2)[1][1]
                    adv_token_rec = self.to_recoverable(adv_token_alt)
                    if adv_token_rec not in seen_set and len(adv_token_rec) == self.num_adv_tokens:
                        adv_token_candidates.append(adv_token_rec)
                        seen_set.add(adv_token_rec)

            # Progress update
            if step % 10 == 0:
                print(f'Step {step}: ensemble_loss={loss.item():.3f}, '
                      f'best_loss={best_loss:.3f}, best_acc={best_acc:.2f}, '
                      f'sparsity={sparsity:.0f}, buffer={len(buffer_set)}')

            # Add to buffer and evaluate
            for adv_token in adv_token_candidates:
                buffer_set.add(adv_token)

                if len(buffer_set) >= self.buffer_size:
                    batch_acc, batch_loss, best_adv, early_stop = self.evaluate_ensemble(
                        list(buffer_set), ensemble_contexts, gt_label
                    )

                    best_acc = max(best_acc, batch_acc)
                    if batch_loss < best_loss:
                        best_loss = batch_loss
                        final_adv = best_adv
                        print(f'  → New best! loss={best_loss:.3f}, acc={best_acc:.2f}')

                    if early_stop:
                        print()
                        print('=' * 70)
                        print('✓ Early stop: Found suffix that works on all contexts!')
                        print('=' * 70)
                        # Clean up caches
                        for ctx in ensemble_contexts:
                            ctx['prefix_cache'] = None
                        return best_loss, best_adv.cpu(), step

                    buffer_set = set()

        # Final evaluation
        if len(buffer_set) > 0:
            batch_acc, batch_loss, best_adv, early_stop = self.evaluate_ensemble(
                list(buffer_set), ensemble_contexts, gt_label
            )

            best_acc = max(best_acc, batch_acc)
            if batch_loss < best_loss:
                best_loss = batch_loss
                final_adv = best_adv

            print()
            print(f'Final: loss={loss.item():.3f}, '
                  f'best_loss={best_loss:.3f}, best_acc={best_acc:.2f}')

            if early_stop:
                print('✓ Found suffix that works on all contexts!')
                # Clean up
                for ctx in ensemble_contexts:
                    ctx['prefix_cache'] = None
                return best_loss, best_adv.cpu(), self.num_steps

        print()
        print('=' * 70)
        print('N-Spread Attack Complete')
        print('=' * 70)

        # Clean up caches
        for ctx in ensemble_contexts:
            ctx['prefix_cache'] = None

        return best_loss, final_adv.cpu(), self.num_steps
