import torch
import torch.nn.functional as F

from .tools import check_legal_input, get_embedding_matrix, get_illegal_tokens


class GIGAAttack:
    """
    Generalizable Infectious Gradient Attack (GIGA)

    Based on "Infecting LLM-based Multi-Agents via Self-propagating Adversarial Attacks"
    by Weichen Yu, Kai Hu, et al. (NeurIPS 2024)

    GIGA combines continuous momentum optimization with discrete coordinate updates
    to create self-propagating adversarial inputs for multi-agent LLM systems.
    """

    def __init__(self,
                 model,
                 tokenizer=None,
                 num_steps=5000,
                 learning_rate=1,
                 momentum=0.99,
                 topK=20,
                 batch_size=8,
                 use_kv_cache=True,
                 judger=None):

        self.model = model
        self.tokenizer = tokenizer
        self.num_steps = num_steps

        self.lr = learning_rate
        self.momentum = momentum
        self.topK = topK
        self.bs = batch_size

        self.device = model.device
        self.dtype = model.dtype
        self.use_kv_cache = use_kv_cache

        embed_mat = get_embedding_matrix(model)
        self.embed_mat = embed_mat.float()
        self.vocal_size = embed_mat.shape[0]

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.buffer_size = 64

        # Get illegal tokens (non-ASCII, special tokens)
        self.illegal_tokens = get_illegal_tokens(tokenizer)

        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.top_p = None
        gen_config.temperature = None
        self.gen_config = gen_config
        self.judger = judger

    def get_optimizer(self, num_adv_tokens):
        """Initialize dense probability vector with random Gaussian noise"""
        soft_opt = torch.randn(1, num_adv_tokens, self.vocal_size)
        soft_opt[..., self.illegal_tokens] = -10**10
        soft_opt = soft_opt.softmax(dim=2)

        soft_opt = soft_opt.to(self.device)
        soft_opt.requires_grad = True

        # Initialize momentum buffer
        momentum_buffer = torch.zeros_like(soft_opt)

        return soft_opt, momentum_buffer

    def coordinate_momentum_update(self, z, momentum_buffer, gt_label):
        """
        Algorithm 1: Coordinate Momentum Update

        Selects one coordinate at a time to update based on momentum gradients.
        Returns the best candidate after evaluating batch_size candidates.
        """
        candidates = []
        losses = []

        for i in range(self.bs):
            # Randomly select one adversarial token position
            j = torch.randint(0, self.num_adv_tokens, (1,)).item()

            # Get top-K promising update coordinates based on negative momentum
            topk_indices = torch.topk(-momentum_buffer[0, j], k=self.topK)[1]
            k = topk_indices[torch.randint(0, self.topK, (1,)).item()].item()

            # Make a copy and update one coordinate while maintaining simplex constraint
            z_candidate = z.clone()
            # Update the single coordinate k at position j
            z_candidate[0, j, k] = z_candidate[0, j, k] - self.lr * momentum_buffer[0, j, k]
            # Clamp to prevent negative values
            z_candidate[0, j] = z_candidate[0, j].clamp(min=1e-10)
            # Renormalize the entire position j to maintain probability simplex (sum to 1)
            z_candidate[0, j] = z_candidate[0, j] / z_candidate[0, j].sum()

            # Compute loss for this candidate
            loss = self.compute_loss(z_candidate, gt_label)

            candidates.append(z_candidate)
            losses.append(loss.item())

        # Select the candidate with the best (lowest) loss
        best_idx = losses.index(min(losses))
        return candidates[best_idx]

    def compute_loss(self, z, gt_label):
        """Compute loss for dense tokens z"""
        # Convert dense tokens to embeddings
        adv_embeds = (z @ self.embed_mat).to(self.dtype)

        if self.use_kv_cache:
            full_embeds = torch.cat([adv_embeds, self.right_embeds], dim=1)
            prefix_cache = self.get_cache(batch_size=adv_embeds.shape[0])
            outputs = self.model(inputs_embeds=full_embeds,
                               past_key_values=prefix_cache)
        else:
            full_embeds = torch.cat([self.left_embeds, adv_embeds, self.right_embeds], dim=1)
            outputs = self.model(inputs_embeds=full_embeds)

        logits = outputs.logits[:, self.logit_slice]

        # Compute RMS loss (root mean square) as per paper Equation 1
        losses = self.loss_fn(logits.mT, gt_label[:adv_embeds.shape[0]])
        rms_loss = torch.sqrt((losses ** 2).mean())

        return rms_loss

    def make_sparse(self, soft_opt, sparsity_level):
        """Convert dense probability vector to sparse based on sparsity level"""
        point = soft_opt.detach().clone()

        # For each token position, keep only top-k values
        mask = torch.zeros_like(soft_opt, dtype=torch.bool)
        for j in range(self.num_adv_tokens):
            k = max(5, int(sparsity_level))  # Minimum sparsity of 5
            top_k = point[0, j].topk(k=k)[1]
            mask[0, j, top_k] = True

        # Zero out non-top-k values and renormalize
        point = torch.where(mask, point.relu() + 1e-6, torch.zeros_like(point))
        point = point / point.sum(dim=2, keepdim=True)

        return point

    @torch.no_grad()
    def evaluate(self, adv_tokens, gt_label):
        """Evaluate discrete adversarial tokens"""
        if len(adv_tokens) == 0:
            return 0, float('inf'), None, False

        # Pad to buffer size if needed
        if len(adv_tokens) < self.buffer_size:
            adv_tokens = adv_tokens + [adv_tokens[0]] * (self.buffer_size - len(adv_tokens))

        adv_tokens = torch.tensor(adv_tokens[:self.buffer_size],
                                 dtype=torch.int64,
                                 device=self.device)

        if self.use_kv_cache:
            full_samples = torch.cat([adv_tokens, self.right_ids], dim=1)
            prefix_cache = self.get_cache(batch_size=full_samples.shape[0])
            outputs = self.model(input_ids=full_samples,
                               past_key_values=prefix_cache)
        else:
            full_samples = torch.cat([self.left_ids, adv_tokens, self.right_ids], dim=1)
            outputs = self.model(input_ids=full_samples)

        outputs = outputs.logits[:, self.logit_slice]
        pred = outputs.argmax(dim=-1)
        accuracies = pred.eq(gt_label).float().mean(1)
        best_acc = accuracies.max().item()

        losses = self.loss_fn(outputs.mT, gt_label)
        losses = losses.mean(1)
        best_loss = losses.min().item()

        best_adv = adv_tokens[losses.argmin()]

        # Check for exact match and further verification
        if best_acc == 1:
            idxes = torch.where(accuracies == 1)[0][:2]
            for idx in idxes:
                good_sample = adv_tokens[idx]
                if self.further_check(good_sample):
                    return best_acc, best_loss, good_sample, True

        return best_acc, best_loss, best_adv, False

    @torch.no_grad()
    def further_check(self, good_sample):
        """Verify the adversarial sample by generating full response"""
        good_sample = good_sample.view(1, -1)
        good_sample = torch.cat([self.left_ids[:1], good_sample, self.right_ids[:1]], dim=1)

        good_sample = good_sample[:, :self.target_start]
        output = self.model.generate(input_ids=good_sample,
                                    generation_config=self.gen_config,
                                    max_new_tokens=512)
        gen_str = self.tokenizer.decode(output.reshape(-1)[self.target_start:])

        if self.judger is not None:
            return self.judger(self.user_prompt, gen_str)
        else:
            return self.response in gen_str

    @torch.no_grad()
    def get_cache(self, batch_size):
        """Get or create KV cache for prefix"""
        assert self.use_kv_cache
        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids[:1], use_cache=True)
            self.prefix_cache = outputs.past_key_values

        if batch_size == 1:
            prefix_cache = self.prefix_cache
        else:
            prefix_cache = [(i.expand(batch_size, -1, -1, -1),
                           j.expand(batch_size, -1, -1, -1))
                          for i, j in self.prefix_cache]
        return prefix_cache

    def to_recoverable(self, x):
        """Check if token sequence is recoverable after decode-encode"""
        gen_str = self.tokenizer.decode(x)
        y = self.tokenizer.encode(gen_str, add_special_tokens=False)
        return tuple(y)

    def clean_cache(self):
        """Clean up cached variables"""
        self.num_adv_tokens = None
        self.left_ids = None
        self.right_ids = None
        self.left_embeds = None
        self.right_embeds = None
        self.logit_slice = None
        self.target_start = None
        self.user_prompt = None
        self.response = None
        if self.use_kv_cache:
            self.prefix_cache = None
        torch.cuda.empty_cache()

    def attack(self, tokens, slices, user_prompt=None, response=None):
        """
        Main GIGA attack algorithm (Algorithm 2 from paper)

        Args:
            tokens: Input token IDs
            slices: Dictionary containing 'adv_slice' and 'target_slice'
            user_prompt: User prompt for judger
            response: Expected response for judger

        Returns:
            Tuple of (best_loss, best_adversarial_tokens, num_steps)
        """
        self.user_prompt = user_prompt
        self.response = response

        tokens = tokens.view(1, -1).to(self.device)
        check_legal_input(tokens, slices)

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop
        self.num_adv_tokens = adv_stop - adv_start

        # Initialize dense adversarial tokens and momentum buffer
        z, momentum_buffer = self.get_optimizer(self.num_adv_tokens)

        # Prepare embeddings and IDs
        embeds = self.model.model.embed_tokens(tokens).detach()
        self.left_embeds = embeds[:, :adv_start]
        self.right_embeds = embeds[:, adv_stop:]

        self.left_ids = tokens[:, :adv_start].expand(self.buffer_size, -1)
        self.right_ids = tokens[:, adv_stop:].expand(self.buffer_size, -1)

        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop
        self.target_start = target_start

        gt_label = tokens[:, target_start:target_stop]
        gt_label = gt_label.expand(self.buffer_size, -1)

        self.logit_slice = slice(target_start - 1, target_stop - 1)
        if self.use_kv_cache:
            self.logit_slice = slice(target_start - 1 - adv_start,
                                   target_stop - 1 - adv_start)

        # Track best results
        seen_set = set()
        buffer_set = set()
        best_loss = float('inf')
        best_acc = 0
        final_adv = tokens[0, slices['adv_slice']]

        # Initialize sparsity tracking
        wrong_count = torch.zeros(1, device=self.device)
        running_wrong = torch.zeros(1, device=self.device)

        print(f"Starting GIGA attack with {self.num_steps} steps...")

        for step in range(self.num_steps):
            # Compute gradient of dense tokens
            z.requires_grad = True
            loss = self.compute_loss(z, gt_label[:1])
            loss.backward()

            # Update momentum buffer: μ ← μ · γ + ∇z
            with torch.no_grad():
                momentum_buffer = momentum_buffer * self.momentum + z.grad
                z.grad = None
                z.requires_grad = False

            # Coordinate momentum update (Algorithm 1)
            z = self.coordinate_momentum_update(z, momentum_buffer, gt_label)

            # Compute wrong predictions for sparsity adaptation
            with torch.no_grad():
                adv_embeds = (z @ self.embed_mat).to(self.dtype)
                if self.use_kv_cache:
                    full_embeds = torch.cat([adv_embeds, self.right_embeds], dim=1)
                    prefix_cache = self.get_cache(batch_size=1)
                    outputs = self.model(inputs_embeds=full_embeds, past_key_values=prefix_cache)
                else:
                    full_embeds = torch.cat([self.left_embeds, adv_embeds, self.right_embeds], dim=1)
                    outputs = self.model(inputs_embeds=full_embeds)

                logits = outputs.logits[:, self.logit_slice]
                wrong_pred = logits.argmax(dim=2) != gt_label[:1]
                wrong_count = wrong_pred.float().sum()

                # Exponential moving average of wrong count
                if step == 0:
                    running_wrong = wrong_count
                else:
                    running_wrong = running_wrong + (wrong_count - running_wrong) * 0.01

                # Adaptive sparsity: sparsity ∝ 2^(wrong_count)
                sparsity = (2 ** running_wrong).clamp(max=self.vocal_size / 2).item()

            # Make tokens sparse and mask illegal tokens
            z.data[..., self.illegal_tokens] = -1000
            last_z = z.detach().clone()

            sparse_z = self.make_sparse(z, sparsity)
            z.data.copy_(sparse_z)

            # Sample discrete candidates from dense probability vector
            adv_token_candidates = []

            # Generate the top candidate first
            adv_token = last_z[0].argmax(dim=1)
            adv_token_rec = self.to_recoverable(adv_token)
            if adv_token_rec not in seen_set and len(adv_token_rec) == self.num_adv_tokens:
                adv_token_candidates.append(adv_token_rec)
                seen_set.add(adv_token_rec)

            # Try variations by using second-best tokens at different positions
            for i in range(self.num_adv_tokens):
                if len(adv_token_candidates) >= self.bs:
                    break

                # Try top-2 token at position i
                if last_z[0, i].topk(2)[0].shape[0] >= 2:
                    adv_token_alt = adv_token.clone()
                    adv_token_alt[i] = last_z[0, i].topk(2)[1][1]
                    adv_token_rec = self.to_recoverable(adv_token_alt)
                    if adv_token_rec not in seen_set and len(adv_token_rec) == self.num_adv_tokens:
                        adv_token_candidates.append(adv_token_rec)
                        seen_set.add(adv_token_rec)

                # Try top-3 token at position i if available
                if len(adv_token_candidates) >= self.bs:
                    break
                if last_z[0, i].topk(min(3, last_z[0, i].shape[0]))[0].shape[0] >= 3:
                    adv_token_alt = adv_token.clone()
                    adv_token_alt[i] = last_z[0, i].topk(3)[1][2]
                    adv_token_rec = self.to_recoverable(adv_token_alt)
                    if adv_token_rec not in seen_set and len(adv_token_rec) == self.num_adv_tokens:
                        adv_token_candidates.append(adv_token_rec)
                        seen_set.add(adv_token_rec)

            # Print progress update
            if step % 10 == 0:
                print(f'Step {step}: loss={loss.item():.2f}, '
                      f'best_loss={best_loss:.2f}, best_acc={best_acc:.2f}, '
                      f'sparsity={sparsity:.0f}, buffer_size={len(buffer_set)}')

            # Add candidates to buffer and evaluate when buffer is full
            for adv_token in adv_token_candidates:
                buffer_set.add(adv_token)

                if len(buffer_set) >= self.buffer_size:
                    batch_acc, batch_loss, best_adv, early_stop = self.evaluate(
                        list(buffer_set), gt_label
                    )

                    best_acc = max(best_acc, batch_acc)
                    if batch_loss < best_loss:
                        best_loss = batch_loss
                        final_adv = best_adv

                    if early_stop:
                        print('Early stop with exact match!')
                        self.clean_cache()
                        return best_loss, best_adv.cpu(), step

                    buffer_set = set()

        # Final evaluation if buffer has remaining candidates
        if len(buffer_set) > 0:
            batch_acc, batch_loss, best_adv, early_stop = self.evaluate(
                list(buffer_set), gt_label
            )

            best_acc = max(best_acc, batch_acc)
            if batch_loss < best_loss:
                best_loss = batch_loss
                final_adv = best_adv

            print(f'Final: loss={loss.item():.2f}, '
                  f'best_loss={best_loss:.2f}, best_acc={best_acc:.2f}')

            if early_stop:
                print('Early stop with exact match!')
                self.clean_cache()
                return best_loss, best_adv.cpu(), step

        self.clean_cache()
        return best_loss, final_adv.cpu(), self.num_steps
