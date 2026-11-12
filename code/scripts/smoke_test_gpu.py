"""End-to-end smoke test of the real PyTorch / transformers / PEFT code paths with TINY random models.

Run this on the GPU machine BEFORE the long runs (about 2-3 minutes, needs only the Qwen3 tokenizer):

    python scripts/smoke_test_gpu.py            # everything
    python scripts/smoke_test_gpu.py --only task1
    python scripts/smoke_test_gpu.py --only task2

It checks what the (mock-based) unit tests cannot: FP32 master weights + GradScaler, L2-SP gradient equivalence,
LoRA/OPLoRA training + checkpoint + resume, FP16 merge/unwrap loading, the diagnostics delta loader, the
visual-token splice (token id, gradients reach the connector), LoRA stages, batched left-padded generation and the
Task 2 evaluator. Exit status != 0 if any check fails.
"""

import argparse
import copy
import os
import shutil
import sys
import tempfile
import traceback
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

RESULTS = []


def check(name, fn):
    try:
        info = fn()
        print(f"[PASS] {name}" + (f"  ({info})" if info else ""))
        RESULTS.append((name, True))
    except Exception:  # noqa: BLE001
        print(f"[FAIL] {name}")
        traceback.print_exc()
        RESULTS.append((name, False))


def tiny_model(vocab, hidden=64, device="cpu", dtype=torch.float32):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                      tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).to(dtype=dtype)


# ================================================================================================
def task1(tok, tmp, device):
    from transformers import AutoModelForCausalLM
    from task1_llm_ft.dataset import MetaMathQADataset, create_dataloader, format_qwen_chat
    from task1_llm_ft.trainer import Task1Trainer, TrainingConfig, _dtype_for
    from task1_llm_ft.l2_sp import L2SPRegularizer
    from task1_llm_ft import checkpoints as ck
    from task1_llm_ft.models import LoRALinear, OPLoRALinear
    from common.metrics import compute_subspace_interference

    base = os.path.join(tmp, "base1")
    tiny_model(len(tok)).save_pretrained(base)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")

    examples = []
    for i in range(48):
        ids, labels = format_qwen_chat(tok, f"What is {i}+{i}?", f"We compute {i}+{i}={2 * i}. The answer is {2 * i}.")
        examples.append({"id": i, "input_ids": ids, "labels": labels, "length": len(ids)})

    def labels_end_at_im_end():
        ids, labels = examples[0]["input_ids"], examples[0]["labels"]
        sup = [i for i, l in enumerate(labels) if l != -100]
        assert ids[sup[-1]] == end_id, "last supervised token must be <|im_end|>"
        assert len(ids) - 1 == sup[-1] + 1 or ids[sup[-1] + 1:] != [], "template newline stays in the input"
        assert all(l == -100 for l in labels[sup[-1] + 1:]), "trailing newline must not be supervised"
        return f"{len(sup)} supervised tokens"
    check("t1 labels end at <|im_end|> (no trailing newline target)", labels_end_at_im_end)

    def make_loader():
        return create_dataloader(MetaMathQADataset(examples), batch_size=2, pad_token_id=tok.pad_token_id or 0, shuffle=False)

    out_root, res_root = os.path.join(tmp, "t1out"), os.path.join(tmp, "t1res")
    trainers = {}

    def train_variant(method, variant, lr, lam=0.0):
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=_dtype_for(method, True))
        cfg = TrainingConfig(model_name_or_path=base, output_dir=os.path.join(out_root, variant),
                             results_dir=os.path.join(res_root, variant), method=method, max_updates=4,
                             gradient_accumulation_steps=2, warmup_updates=1, learning_rate=lr, lambda_l2sp=lam,
                             oplora_protection_rank=8, checkpoint_updates=(2, 4), displacement_updates=(2, 4))
        tr = Task1Trainer(cfg, model, make_loader(), tok)
        summary = tr.train()
        trainers[variant] = tr
        losses = [h["sft_loss"] for h in tr.history]
        assert len(losses) == 4 and all(np.isfinite(losses)), losses
        return summary, tr

    def full_ft():
        summary, tr = train_variant("full_ft", "full_ft", 2e-5)
        assert all(p.dtype == torch.float32 for p in tr.model.parameters()), "Full FT must keep FP32 master weights"
        assert any("displacement" in h for h in tr.history)
        return f"trainable={summary['trainable_parameters']:,} skipped={summary['skipped_updates']}"
    check("t1 Full FT trains (FP32 masters + GradScaler unscale)", full_ft)

    def l2sp():
        summary, tr = train_variant("l2_sp", "full_ft_l2_sp", 2e-5, 1e-3)
        h = tr.history[-1]
        assert h["omega"] > 0 and abs(h["total_loss"] - (h["sft_loss"] + 1e-3 * h["omega"])) < 1e-9
        return f"Omega={h['omega']:.4f} displacement={h.get('displacement', 0):.4f}"
    check("t1 Full FT + L2-SP trains and logs Omega / lambda*Omega / total", l2sp)

    for method, rank in (("lora_4", 4), ("lora_16", 16), ("oplora_16", 16)):
        def adapter(method=method, rank=rank):
            summary, tr = train_variant(method, method, 2e-4)
            mods = [m for m in tr.model.modules() if isinstance(m, (LoRALinear, OPLoRALinear))]
            assert len(mods) == 14, len(mods)  # 2 layers x 7 targets
            assert summary["trainable_parameters"] > 0 and next(tr.model.parameters()).dtype in (torch.float16, torch.float32)
            base_dtypes = {m.base_layer.weight.dtype for m in mods}
            return f"trainable={summary['trainable_parameters']:,} base dtypes={base_dtypes}"
        check(f"t1 {method} trains", adapter)

    def l2_grad_equivalence():
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).to(device)
        reg = L2SPRegularizer(m, lambda_reg=1e-3, store_on_cpu=False)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        pen, omega = reg.compute_penalty(m)
        pen.backward()
        g_auto = {n: p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        omega2 = reg.add_penalty_gradient(m)
        assert abs(omega - omega2) < 1e-4 * max(1.0, omega)
        for n, p in m.named_parameters():
            torch.testing.assert_close(p.grad, 1e-3 * g_auto[n], rtol=1e-4, atol=1e-8)
        return f"Omega={omega:.4f}"
    check("t1 L2-SP gradient term == autograd of lambda*Omega", l2_grad_equivalence)

    def selective_head():
        tr = trainers["full_ft"]
        batch = next(iter(make_loader()))
        args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")] + [batch["num_valid_tokens"]]
        tr.model.eval()
        with torch.no_grad():
            tr.config.selective_head = True
            a = tr._microbatch_loss(*args)
            tr.config.selective_head = False
            b = tr._microbatch_loss(*args)
        tr.config.selective_head = True
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        return f"loss={float(a):.5f}"
    check("t1 selective-head loss == full-logits loss", selective_head)

    # Condition loading: map the real steps (625 / 1250) onto the smoke run's (2 / 4)
    specs = copy.deepcopy(ck.CONDITION_SPECS)
    for s in ck.CONDITION_SPECS.values():
        s["step"] = {625: 2, 1250: 4}.get(s["step"], s["step"])
    try:
        merge_batch = ck.make_merge_check_batch(tok, ["The capital of France is", "What is 3+3? The answer is"])
        for cond in ck.ALL_CONDITIONS:
            def load_one(cond=cond):
                model, info = ck.load_condition_model(cond, out_root, base, device, merge_check_inputs=merge_batch)
                assert not any(isinstance(m, (LoRALinear, OPLoRALinear)) for m in model.modules()), "adapters must be unwrapped"
                assert next(model.parameters()).dtype == torch.float16
                ids = tok("hello world", return_tensors="pt").to(device)
                assert torch.isfinite(model(**ids).logits.float()).all()
                mc = info.get("merge_check")
                return f"top1={mc['top1_agreement']:.3f} maxdiff={mc['max_abs_logit_diff']:.3g}" if mc else "ok"
            check(f"t1 load condition '{cond}' (fresh FP16 backbone, merged, unwrapped)", load_one)

        def deltas():
            w0 = {i: AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).model.layers[i].mlp.down_proj.weight.detach().numpy()
                  for i in (0, 1)}
            U = {i: np.linalg.svd(w0[i], full_matrices=False) for i in (0, 1)}
            out = []
            for cond in ("full_ft_final", "lora_16_final", "oplora_16_final"):
                d = ck.load_down_proj_deltas(cond, out_root, [0, 1], w0)
                rho = [compute_subspace_interference(d[i], U[i][0][:, :8]) for i in (0, 1)]
                rho_in = [float(np.sum((d[i] @ U[i][2][:8].T) ** 2) / max(np.sum(d[i] ** 2), 1e-30)) for i in (0, 1)]
                if cond == "oplora_16_final":
                    # The tiny random base is saved in FP32 but the adapters were initialised from its FP16 cast (W0 differs by ~5e-4 relative,
                    # so rho ~ 1e-7 here). Real Qwen3 weights are BF16 on the hub, where the FP16 cast is exact.
                    assert max(rho) < 1e-5 and max(rho_in) < 1e-5, (rho, rho_in)  # the projections hold numerically
                out.append(f"{cond}: rho_k={rho[0]:.2e}")
            return "; ".join(out)
        check("t1 diagnostics delta loader; OPLoRA rho_k == 0 (output and input side)", deltas)
    finally:
        ck.CONDITION_SPECS.clear()
        ck.CONDITION_SPECS.update(specs)

    def resume():
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
        cfg = copy.deepcopy(trainers["lora_16"].config)
        cfg.output_dir = os.path.join(out_root, "lora_16_resumed")
        cfg.results_dir = os.path.join(res_root, "lora_16_resumed")
        tr = Task1Trainer(cfg, model, make_loader(), tok)
        summary = tr.train(resume_from=os.path.join(out_root, "lora_16", "checkpoint_step_2"))
        assert summary["total_updates"] == 4 and len(tr.history) == 4
        ref = torch.load(os.path.join(out_root, "lora_16", "checkpoint_step_4", "adapter_state.pt"), weights_only=True)
        got = torch.load(os.path.join(cfg.output_dir, "checkpoint_step_4", "adapter_state.pt"), weights_only=True)
        diff = max(float((ref[k].float() - got[k].float()).abs().max()) for k in ref)
        return f"max |resumed - uninterrupted| adapter diff = {diff:.2e} (small = deterministic resume)"
    check("t1 resume from checkpoint_step_2 completes", resume)


# ================================================================================================
class FakeCache:
    """Deterministic fake SigLIP features for any image key."""

    def exists(self, key):
        return True

    def preload(self, keys):
        pass

    def load(self, key, as_torch=True, device=None, num_tokens=196):
        g = torch.Generator().manual_seed(zlib.crc32(str(key).encode()))
        x = torch.randn(196, 768, generator=g).half()
        if num_tokens == 49:
            x = x.view(7, 2, 7, 2, 768).float().mean(dim=(1, 3)).reshape(49, 768).half()
        return x


def task2(tok, tmp, device):
    from task2_vlm.models import SiglipQwenVLM, encode_example
    from task2_vlm.dataset import VLMCollator, CurriculumBatchSampler, create_synthetic_task2_datasets
    from task2_vlm.trainer import VLMCurriculumTrainer
    from task2_vlm.evaluate import run_full_evaluation

    cache = FakeCache()
    dtype = torch.float16 if device == "cuda" else torch.float32
    vlm = SiglipQwenVLM(llm_model_name="tiny", visual_dim=768, llm_dim=64, device=device, load_pretrained=False)
    vlm.tokenizer = tok
    vlm.llm = tiny_model(len(tok), device=device, dtype=dtype).to(device)
    for p in vlm.llm.parameters():
        p.requires_grad = False
    vlm.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.image_pad_token_id = int(tok.convert_tokens_to_ids("<|image_pad|>"))
    vlm.end_of_turn_token_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    vlm.to(device)

    def token_ids():
        assert vlm.image_pad_token_id == 151655, vlm.image_pad_token_id
        p, a = encode_example(tok, "How many dogs can you see?", "3", True, 196, 16, 80)
        assert p.count(vlm.image_pad_token_id) == 196
        assert tok.decode(p[-8:]).replace("\n", "\\n").endswith("<think>\\n\\n</think>\\n\\n"), tok.decode(p[-8:])
        p0, _ = encode_example(tok, "How many dogs can you see?", None, False, 196, 16, 80)
        assert p0.count(vlm.image_pad_token_id) == 0 and len(p) - len(p0) == 197
        long_q, long_a = "word " * 300, "token " * 200
        pq, aq = encode_example(tok, long_q, long_a, True, 196, 16, 80)
        assert len(aq) == 16
        assert len(tok.encode(("word " * 80).strip(), add_special_tokens=False)) == 80
        return "id=151655, 196 placeholders, empty <think> block, caps ok"
    check("t2 prompt/token ids (placeholder id, think block, caps)", token_ids)

    ds = create_synthetic_task2_datasets(seed=6304)  # synthetic TEXT only; features are fake - only for plumbing
    collator = VLMCollator(tok, cache, 196, vlm.image_pad_token_id, device=device)

    def forward_backward():
        batch = collator(ds["train_captions"][:8] + [], include_images=True)
        assert batch["visual_features"].shape == (8, 196, 768)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        vlm.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            loss = vlm(input_ids=batch["input_ids"], visual_features=batch["visual_features"],
                       attention_mask=batch["attention_mask"], labels=batch["labels"])["loss"]
        scaler.scale(loss).backward()
        g = [p.grad for p in vlm.connector.parameters()]
        assert all(x is not None for x in g) and any(float(x.abs().sum()) > 0 for x in g), "connector must receive gradients"
        assert all(p.grad is None for p in vlm.llm.parameters()), "frozen backbone must not get gradients"
        vlm.eval()
        with torch.no_grad():
            f1 = batch["visual_features"]
            f2 = torch.randn_like(f1)
            a = vlm(input_ids=batch["input_ids"], visual_features=f1, attention_mask=batch["attention_mask"])["logits"]
            b = vlm(input_ids=batch["input_ids"], visual_features=f2, attention_mask=batch["attention_mask"])["logits"]
        assert float((a.float() - b.float()).abs().max()) > 0, "logits must depend on the image features"
        return f"loss={float(loss):.3f}"
    check("t2 splice: connector gradients, frozen LLM, logits depend on image", forward_backward)

    def lora_roundtrip():
        vlm.enable_lora()
        keys = [k for k in vlm.trainable_state_dict() if "lora_" in k]
        assert keys and sum(p.numel() for p in vlm.llm.parameters() if p.requires_grad) > 0
        path = os.path.join(tmp, "ad.pt")
        vlm.save_adapters(path)
        before = {k: v.clone() for k, v in vlm.llm.state_dict().items() if "lora_" in k}
        with torch.no_grad():
            for k, v in vlm.llm.state_dict().items():
                if "lora_B" in k:
                    v.add_(1.0)
        vlm.load_adapters(path)
        for k, v in vlm.llm.state_dict().items():
            if "lora_" in k:
                torch.testing.assert_close(v.float(), before[k].float())
        return f"{len(keys)} LoRA tensors; dtype={next(v for k, v in before.items()).dtype}"
    check("t2 enable_lora + save/load adapters round trip", lora_roundtrip)

    def batched_generation():
        vlm.eval()
        prompts = [encode_example(tok, q, None, img, 196, 0, 80)[0] for q, img in
                   (("Is it a dog?", True), ("How many cars can you see in this image today?", True), ("Is it a dog?", False))]
        feats = [cache.load("val2014_1").to(device), cache.load("val2014_2").to(device), None]
        out = vlm.generate_batch(prompts, feats, max_new_tokens=8)
        assert len(out) == 3 and all(isinstance(x, list) and len(x) <= 8 for x in out)
        alone = vlm.generate_batch([prompts[0]], [feats[0]], max_new_tokens=8)[0]
        return f"batched[0]==alone: {out[0] == alone}"
    check("t2 batched left-padded generation (image / different length / no image)", batched_generation)

    def curriculum_and_eval():
        sampler = CurriculumBatchSampler(ds["train_captions"], ds["train_presence"], ds["train_counting"], ds["train_vqav2"], seed=6304)
        out_dir, res_dir = os.path.join(tmp, "t2out"), os.path.join(tmp, "t2res")
        tr = VLMCurriculumTrainer(vlm, sampler, collator, out_dir, res_dir, ds["dev_captions"][:8], ds["dev_grounding"][:4],
                                  ds["dev_vqav2"][:4], device=device, stage_budgets={1: 2, 2: 2, 3: 2})
        summary = tr.run_training()
        for f in ("connector_stage_1.pt", "connector_stage_2.pt", "connector_stage_3.pt", "lora_adapters_stage_2.pt", "lora_adapters_stage_3.pt"):
            assert os.path.exists(os.path.join(out_dir, f)), f
        ts = {s: summary["stage_stats"][s]["trainable_parameters"] for s in (1, 2, 3)}
        assert ts[1] < ts[2] == ts[3], ts
        assert len(tr.dev_logs) >= 4 and tr.dev_logs[0]["global_update"] == 0
        report = run_full_evaluation(vlm, collator, ds["test_grounding"][:6], ds["test_vqav2"][:6], out_dir, res_dir,
                                     n_resamples=50, seed=6304, batch_size=8)
        for suite in ("grounding_suite", "vqav2_suite"):
            assert {"correct", "swap", "mismatched", "no_image", "correct_advantage_over_swap", "swap_behavior"} <= set(report[suite])
        return f"trainable by stage={ts}"
    check("t2 3-stage curriculum, checkpoints, dev eval, full evaluation", curriculum_and_eval)


# ================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["task1", "task2"], default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "  [CPU: GradScaler/FP16 paths NOT exercised]"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tmp = tempfile.mkdtemp(prefix="pa3_smoke_")
    try:
        if args.only in (None, "task1"):
            task1(tok, tmp, device)
        if args.only in (None, "task2"):
            task2(tok, tmp, device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
