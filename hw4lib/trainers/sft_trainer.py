"""Response-only supervised fine-tuning for the P1 decoder."""
import json
import math
import yaml
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from ..decoding.sequence_generator import SequenceGenerator
from ..utils.webnlg import format_webnlg_prompt, score_webnlg


class SFTTrainer(BaseTrainer):
    """Train a decoder on prompt/response pairs.

    Complete the marked training, validation, and generation methods.
    Reuse your P1 model and decoding implementation.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.best_metric = -float("inf")  # Higher Kaggle score is better.
        self.amp_enabled = bool(config["training"].get("amp", True)) and torch.device(self.device).type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        # TODO: Use summed cross-entropy, ignoring labels equal to -100.
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")  # TODO

    def _init_experiment(self, run_name, config_file):
        """Provided: use a new output directory, or explicitly resume an existing run."""
        from time import time_ns
        root = (
            Path(self.config.get("output_dir", "expts"))
            .expanduser()
            .resolve()
            / run_name
        )
        resume = self.config["training"].get("resume", False)
        if resume:
            last_checkpoint = root / "checkpoints" / "last.pth"
            if not last_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: {last_checkpoint} was not found. "
                    "If training stopped before the first checkpoint, set this run's resume flag to false and rerun."
                )
        else:
            if root.exists():
                # Preserve any experiment containing saved model weights.
                has_checkpoint = any(
                    path.is_file()
                    and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
                    for path in root.rglob("*")
                )
                if has_checkpoint:
                    raise FileExistsError(
                        f"Experiment {run_name!r} already contains saved model weights. "
                        "Set resume to true to continue from last.pth, or choose a new run name."
                    )
                # Finish the notebook's previous W&B session before
                # moving its files.
                if self.use_wandb:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                archived_root = root.with_name(
                    f"{root.name}-before-first-checkpoint-{time_ns()}"
                )
                root.rename(archived_root)
                print(
                    "Note: The previous attempt has no saved checkpoint. "
                    f"Its files were preserved at {archived_root}. "
                    "Starting this experiment again from the P1 checkpoint."
                )
            root.mkdir(parents=True, exist_ok=False)
            (root / "config.yaml").write_text(yaml.safe_dump(self.config))
            (root / "model.txt").write_text(str(self.model) + "\n")
        checkpoint, attn, text = (root / part for part in ("checkpoints", "attn", "text"))
        for directory in (checkpoint, attn, text):
            directory.mkdir(exist_ok=True)
        if self.use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            self.wandb_run = wandb.init(
                project=self.config["training"].get("wandb_project", "hw4p2"),
                entity=self.config["training"].get("wandb_entity") or None,
                name=run_name, config=self.config,
                id=self.config["training"].get("wandb_run_id") or None,
                resume="must" if resume else None, dir=str(root))
        return root, checkpoint, attn, text, checkpoint / "best.pth", checkpoint / "last.pth"


    @staticmethod
    def _metrics(loss_sum, token_count):
        """Provided: perplexity here is response-token PPL, not character PPL."""
        if token_count <= 0:
            raise ValueError("No supervised response tokens")
        ce = loss_sum / token_count
        if not math.isfinite(ce):
            raise FloatingPointError("Non-finite response loss")
        return {"response_ce": ce, "response_ppl": math.exp(ce), "response_tokens": token_count}

    def _batch(self, batch):
        """Provided: move tensors and reject a batch without supervised targets."""
        inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
        if inputs.ndim != 2 or inputs.shape != labels.shape:
            raise ValueError("input_ids and labels must have matching (B, T) shapes")
        count = int((labels != -100).sum())
        if count == 0:
            raise ValueError("Batch has no supervised response tokens")
        return inputs, labels, count

    def _train_epoch(self, dataloader):
        """Accumulate summed losses; normalize gradients by the cycle's token count.

        Handle a partial final accumulation cycle. Unscale before clipping.
        Advance the scheduler only when the optimizer step ran.
        """
        # Provided: keep this call when implementing the training loop.
        self._bind_model_to_active_wandb()
        # TODO: Implement one training epoch using the instructions above.
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized")
        num_batches = len(dataloader)
        if num_batches == 0:
            raise ValueError("Training dataloader must contain at least one batch")
        accumulation = self.config["training"].get("gradient_accumulation_steps", 1)
        if not isinstance(accumulation, int) or isinstance(accumulation, bool) or accumulation <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        clip = float(self.config["training"].get("gradient_clip", 1.0))
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")

        self.model.train()
        self.optimizer.zero_grad()
        device_type = torch.device(self.device).type
        total_loss, total_tokens, cycle_tokens = 0.0, 0, 0

        bar = tqdm(total=num_batches, dynamic_ncols=True, leave=False, desc="[SFT train]")
        for index, batch in enumerate(dataloader):
            inputs, labels, count = self._batch(batch)
            with torch.autocast(device_type=device_type, dtype=torch.float16,
                                enabled=self.amp_enabled):
                logits, _ = self.model(inputs)
                # Summed CE over response (and EOS) targets only
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            total_tokens += count
            cycle_tokens += count

            if (index + 1) % accumulation == 0 or index + 1 == num_batches:
                # Unscale, then convert summed grads to per-token mean grads.
                self.scaler.unscale_(self.optimizer)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(cycle_tokens)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= old_scale:
                    self.global_optimizer_step += 1
                    if self.scheduler is not None:
                        self.scheduler.step()
                self.optimizer.zero_grad()
                cycle_tokens = 0

            bar.set_postfix(response_ce=f"{total_loss / total_tokens:.4f}")
            bar.update()
        bar.close()
        return self._metrics(total_loss, total_tokens), {}

    @torch.inference_mode()
    def _validate_epoch(self, dataloader):
        """Compute response CE from total NLL / total supervised tokens."""
        # TODO: Evaluate without gradients; exclude prompt and padding targets.
        self.model.eval()
        device_type = torch.device(self.device).type
        total_loss, total_tokens = 0.0, 0
        for batch in tqdm(dataloader, dynamic_ncols=True, leave=False, desc="[SFT dev CE]"):
            inputs, labels, count = self._batch(batch)
            with torch.autocast(device_type=device_type, dtype=torch.float16,
                                enabled=self.amp_enabled):
                logits, _ = self.model(inputs)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1)
            )
            total_loss += loss.item()
            total_tokens += count
        return self._metrics(total_loss, total_tokens)

    @torch.inference_mode()
    def generate(self, records):
        """Generate from the triples alone, using your P1 greedy decoder.

        Do not append references to prompts. Decode one unpadded prompt at a time.
        Limit total context and stop at generated EOS, excluding prompt text.
        """
        self.model.eval()
        generation = self.config.get("generation", {})
        budget = generation.get("max_new_tokens", 128)
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("generation.max_new_tokens must be a positive integer")
        max_len = self.model.max_len
        cache_fn = self.model.score_cached if getattr(self.model, "use_rope", False) else None
        repeat_penalty = float(generation.get("repeat_penalty", 1.0))

        # Optional deterministic beam search + entity-coverage reranking (beam_width 1 = greedy).
        beam_width = int(generation.get("beam_width", 1))
        if beam_width > 1:
            from ..decoding.beam_rerank import beam_search, rerank

        # Optional delexicalisation: names -> ENT1, ENT2, ... in the triples; restored after generation.
        use_delex = bool(self.config.get("delex", {}).get("enabled", False))
        if use_delex:
            from ..data.delex import delex_triples, relex

        predictions = []
        for record in tqdm(records, dynamic_ncols=True, leave=False, desc="[SFT generate]"):
            triples, mapping = (delex_triples(record["triples"]) if use_delex
                                else (record["triples"], None))
            # Prompt from the triples only; references are never appended.
            prompt_ids = self.tokenizer.encode(format_webnlg_prompt(triples))
            if not prompt_ids or len(prompt_ids) >= max_len:
                raise ValueError(
                    f"Prompt for {record.get('id')} is empty or does not fit context {max_len}"
                )
            if beam_width > 1:
                candidates = beam_search(
                    self.model, prompt_ids, self.tokenizer.eos_id,
                    beam_width=beam_width, max_new_tokens=budget, max_len=max_len,
                    no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 0)),
                    min_new_tokens=int(generation.get("min_new_tokens", 0)),
                    length_penalty=float(generation.get("length_penalty", 1.0)),
                    device=self.device,
                )
                texts = [(self.tokenizer.decode(ids, skip_special_tokens=True).strip(), score)
                         for ids, score in candidates]
                text = rerank(texts, triples) if generation.get("rerank", True) else texts[0][0]
                predictions.append(relex(text, mapping) if use_delex else text)
                continue

            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            decoder = SequenceGenerator(
                score_fn=self.model.score,
                tokenizer=self.tokenizer,
                max_length=min(max_len, len(prompt_ids) + budget),
                device=self.device,
                cache_score_fn=cache_fn,
            )
            sequences, _ = decoder.generate_greedy(prompt, repeat_penalty=repeat_penalty)
            sequence = decoder.post_process_sequence(
                sequences[0], self.tokenizer, start_index=len(prompt_ids)
            )
            generated = sequence[len(prompt_ids):].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(relex(text, mapping) if use_delex else text)
        return predictions

    def evaluate(self, dataloader, records=None):
        """Provided: combine teacher-forced response loss with generated-text metrics."""
        metrics = self._validate_epoch(dataloader)
        if records is not None:
            predictions = self.generate(records)
            scores = score_webnlg(predictions, [row["references"] for row in records])
            metrics.update(bleu=scores["bleu"], rouge_l=scores["rouge_l"],
                           kaggle_score=scores["kaggle_score"], examples=len(records))
        return metrics

    def train(self, train_dataloader, val_dataloader, dev_records):
        """Run epochs and select the best generated dev Kaggle-score checkpoint.

        Save last.pth for resuming the next epoch and best.pth for prediction.
        Log token loss and generation scores as different quantities.
        """
        # TODO: Run training/evaluation, update best Kaggle score, log, and save checkpoints.
        if not dev_records:
            raise ValueError("dev_records must contain at least one input")
        epochs = self.config["training"]["epochs"]
        for epoch in range(self.current_epoch, epochs):
            sampler = getattr(train_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_metrics, _ = self._train_epoch(train_dataloader)
            dev_metrics = self.evaluate(val_dataloader, dev_records)

            # Select on generated dev Kaggle score, (BLEU + ROUGE-L) / 2.
            improved = dev_metrics["kaggle_score"] > self.best_metric
            if improved:
                self.best_metric = dev_metrics["kaggle_score"]
            self._log_metrics({"train": train_metrics, "dev": dev_metrics}, epoch + 1)
            self.current_epoch = epoch + 1

            if improved:
                self.save_checkpoint("best.pth")
            self.save_checkpoint("last.pth")
            (self.expt_root / "history.json").write_text(
                json.dumps(self.training_history, indent=2, default=float)
            )

