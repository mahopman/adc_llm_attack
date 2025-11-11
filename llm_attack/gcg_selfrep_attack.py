import torch
import torch.nn.functional as F

from .tools import check_legal_input, get_embedding_matrix
from .gcg_attack import get_illegal_tokens


class GCGSelfRepAttack:
    """
    GCG with Self-Replication Loss

    This variant of GCG adds an explicit self-replication loss term that encourages
    the model to output the adversarial tokens in its response. This creates a
    "viral" effect similar to GIGA, where the adversarial input propagates.

    The loss function is:
        L = L_jailbreak + λ * L_replication

    where:
    - L_jailbreak: Standard cross-entropy loss for generating the target response
    - L_replication: Cross-entropy loss for outputting the adversarial tokens
    - λ: Replication weight parameter (default: 1.0)
    """

    def __init__(self,
                 model,
                 tokenizer,
                 num_steps=500,
                 topK=256,
                 batch_size=512,
                 use_kv_cache=True,
                 replication_weight=1.0,
                 replication_position='start',  # 'start' or 'after_adv'
                 judger=None):
        """
        Args:
            replication_weight: Weight for the self-replication loss term (λ)
            replication_position: Where to encourage replication
                - 'start': At the start of the model's response (target_start)
                - 'after_adv': Right after the adversarial tokens (adv_stop)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_steps = num_steps

        self.topK = topK
        self.bs = batch_size

        self.device = model.device
        self.dtype = model.dtype
        self.use_kv_cache = use_kv_cache

        # Self-replication parameters
        self.replication_weight = replication_weight
        self.replication_position = replication_position

        embed_mat = get_embedding_matrix(model)
        self.embed_mat = embed_mat.float()
        self.vocal_size = embed_mat.shape[0]

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.illegal_tokens = get_illegal_tokens(tokenizer)

        gen_config = self.model.generation_config
        gen_config.do_sample = False
        gen_config.top_p = None
        gen_config.temperature = None
        self.gen_config = gen_config
        self.judger = judger

    @torch.no_grad()
    def evaluate(self, resamples, gt_label):
        bs = resamples.shape[0]
        buffer_size = 64
        rounds = bs // buffer_size + (bs % buffer_size != 0)
        if self.use_kv_cache:
            full_samples = torch.cat([resamples, self.right_ids], dim=1)
            outputs = []
            for k in range(rounds):
                part_samples = full_samples[k*buffer_size: (k+1)*buffer_size]
                batch_size = part_samples.shape[0]
                prefix_cache = self.get_cache(batch_size=batch_size)
                part_outputs = self.model(input_ids=part_samples,
                                          past_key_values=prefix_cache)
                outputs.append(part_outputs.logits)
                del part_samples, prefix_cache, part_outputs
                torch.cuda.empty_cache()

        else:
            full_samples = torch.cat([self.left_ids, resamples, self.right_ids],
                                     dim=1)
            outputs = []
            for k in range(rounds):
                part_samples = full_samples[k*128: (k+1)*128]
                part_outputs = self.model(input_ids=part_samples)
                outputs.append(part_outputs.logits)
                del part_samples, part_outputs
                torch.cuda.empty_cache()

        outputs = torch.cat(outputs, dim=0)[:, self.logit_slice]

        pred = outputs.argmax(dim=-1)
        accuracies = pred.eq(gt_label).float().mean(1)
        best_acc = accuracies.max().item()

        losses = self.loss_fn(outputs.mT, gt_label)
        losses = losses.mean(1)
        best_loss = losses.min().item()

        best_adv = resamples[losses.argmin()]

        if best_acc == 1:
            idxes = torch.where(accuracies == 1)[0][:2]
            for idx in idxes:
                good_sample = resamples[idx]
                if self.further_check(good_sample, gt_label[0]):
                    return best_acc, best_loss, good_sample, True

        return best_acc, best_loss, best_adv, False

    def get_resample_ids(self):
        lookup = set()
        token_idx = torch.randint(self.num_adv_tokens, size=[self.bs])
        cand_idx = torch.randint(self.topK, size=[self.bs])
        for i, j in zip(token_idx, cand_idx):
            lookup.add((i.item(), j.item()))

        while len(lookup) < self.bs:
            i = torch.randint(self.num_adv_tokens, size=[1])
            j = torch.randint(self.topK, size=[1])
            lookup.add((i.item(), j.item()))
        return lookup

    @torch.no_grad()
    def further_check(self, good_sample, gt_label):
        good_sample = good_sample.view(1, -1)
        good_sample = torch.cat([self.left_ids[:1], good_sample, self.right_ids[:1]],
                                dim=1)

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
        assert self.use_kv_cache
        if not hasattr(self, 'prefix_cache') or self.prefix_cache is None:
            outputs = self.model(self.left_ids[:1], use_cache=True)
            prefix_cache = outputs.past_key_values
            self.prefix_cache = prefix_cache

        prefix_cache = [(torch.tile(i, dims=[batch_size, 1, 1, 1]),
                         torch.tile(j, dims=[batch_size, 1, 1, 1]))
                            for i, j in self.prefix_cache]
        return prefix_cache

    def clean_cache(self):
        self.num_adv_tokens = None
        self.left_ids = None
        self.right_ids = None
        self.logit_slice = None
        self.replication_slice = None
        self.target_start = None
        self.request = None
        self.response = None
        if self.use_kv_cache:
            self.prefix_cache = None
        torch.cuda.empty_cache()

    def attack(self, tokens, slices, user_prompt=None, response=None):
        self.user_prompt = user_prompt
        self.response = response

        tokens = tokens.view(1, -1).to(self.device)
        check_legal_input(tokens, slices)

        adv_start = slices['adv_slice'].start
        adv_stop = slices['adv_slice'].stop

        target_start = slices['target_slice'].start
        target_stop = slices['target_slice'].stop
        self.target_start = target_start

        self.num_adv_tokens = adv_stop - adv_start
        adv_token = tokens[:, adv_start:adv_stop]

        embeds = self.model.model.embed_tokens(tokens).detach()
        left = embeds[:, :adv_start]
        right = embeds[:, adv_stop:]

        self.left_ids = tokens[:, :adv_start].expand(self.bs, -1)
        self.right_ids = tokens[:, adv_stop:].expand(self.bs, -1)

        # Standard jailbreak loss slice (for generating target response)
        self.logit_slice = slice(target_start - 1, target_stop - 1)
        if self.use_kv_cache:
            self.logit_slice = slice(target_start - 1 - adv_start,
                                     target_stop - 1 - adv_start)

        # Self-replication loss slice (for outputting adversarial tokens)
        if self.replication_position == 'start':
            # Replicate at the start of the target response
            rep_start = target_start - 1
            rep_stop = min(rep_start + self.num_adv_tokens, target_stop - 1)
            if self.use_kv_cache:
                rep_start = rep_start - adv_start
                rep_stop = rep_stop - adv_start
            self.replication_slice = slice(rep_start, rep_stop)
        elif self.replication_position == 'after_adv':
            # Replicate right after adversarial tokens
            rep_start = adv_stop - 1
            rep_stop = rep_start + self.num_adv_tokens
            if self.use_kv_cache:
                rep_start = self.num_adv_tokens - 1
                rep_stop = 2 * self.num_adv_tokens - 1
            self.replication_slice = slice(rep_start, rep_stop)

        # Actual length of replication tokens (may be less than num_adv_tokens)
        self.rep_length = self.replication_slice.stop - self.replication_slice.start

        gt_label = tokens[:, target_start:target_stop]
        gt_label = gt_label.expand(self.bs, -1)

        # Replication label is the adversarial tokens themselves
        rep_label = adv_token[:, :self.rep_length]
        rep_label = rep_label.expand(self.bs, -1)

        best_loss, best_acc = 1000, 0
        final_adv = None

        print(f"Starting GCG with Self-Replication Attack")
        print(f"  Replication weight (λ): {self.replication_weight}")
        print(f"  Replication position: {self.replication_position}")
        print(f"  Adversarial tokens: {self.num_adv_tokens}")
        print(f"  Replication tokens: {self.rep_length}")

        for step_ in range(self.num_steps + 1):
            adv_onehot = F.one_hot(adv_token, num_classes=self.vocal_size)
            adv_onehot = adv_onehot.float()
            adv_onehot.requires_grad = True

            adv_embeds = (adv_onehot @ self.embed_mat).to(self.dtype)
            if self.use_kv_cache:
                full_embeds = torch.cat([adv_embeds, right], dim=1)
                prefix_cache = self.get_cache(batch_size=adv_embeds.shape[0])
                outputs = self.model(inputs_embeds=full_embeds,
                                     past_key_values=prefix_cache)
            else:
                full_embeds = torch.cat([left, adv_embeds, right], dim=1)
                outputs = self.model(inputs_embeds=full_embeds)

            # Compute jailbreak loss (standard GCG objective)
            logits_jailbreak = outputs.logits[:, self.logit_slice]
            loss_jailbreak = self.loss_fn(logits_jailbreak.mT, gt_label[:1])
            loss_jailbreak = loss_jailbreak.mean()

            # Compute self-replication loss
            logits_replication = outputs.logits[:, self.replication_slice]
            loss_replication = self.loss_fn(logits_replication.mT, rep_label[:1])
            loss_replication = loss_replication.mean()

            # Combined loss
            ell = loss_jailbreak + self.replication_weight * loss_replication
            ell.backward()

            grad = adv_onehot.grad[0]
            grad[..., self.illegal_tokens] = 10 ** 10

            candidates = grad.topk(k=self.topK, dim=1, largest=False)[1]

            # Filter ids that aren't the same after detokenize->retokenize
            def recoverable_x(x):
                gen_str = self.tokenizer.decode(x)
                y = self.tokenizer.encode(gen_str, add_special_tokens=False)
                return torch.tensor(y).to(x.device)

            resample_ids = self.get_resample_ids()
            resamples = []
            for (i, j) in resample_ids:
                tmp = adv_token.clone()[0]
                tmp[i] = candidates[i, j]
                tmp = recoverable_x(tmp).view(1, -1)
                if tmp.shape[1] == adv_token.shape[1]:
                    resamples.append(tmp)

            if len(resamples) < self.bs:
                resamples += resamples[:1] * (self.bs - len(resamples))

            resamples = torch.cat(resamples, dim=0)
            out = self.evaluate(resamples, gt_label)
            batch_best_acc, batch_best_loss, best_batch_adv, early_stop = out
            adv_token = best_batch_adv.view(1, -1)

            best_acc = max(best_acc, batch_best_acc)
            if batch_best_loss < best_loss:
                final_adv = best_batch_adv
                best_loss = batch_best_loss

            if step_ % 50 == 0 or early_stop:
                print(f'iter:{step_}, '
                      f'loss_jail:{loss_jailbreak.item():.2f}, '
                      f'loss_rep:{loss_replication.item():.2f}, '
                      f'loss_total:{ell.item():.2f}, '
                      f'best_loss:{best_loss:.2f}, '
                      f'best_acc:{best_acc:.2f}')

            if early_stop:
                print('Early Stop with an Exact Match!')
                self.clean_cache()
                return best_loss, best_batch_adv.cpu(), step_

        self.clean_cache()
        return best_loss, final_adv.cpu(), step_
