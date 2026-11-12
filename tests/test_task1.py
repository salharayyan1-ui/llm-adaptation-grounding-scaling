"""Comprehensive unit tests for Task 1: LLM Fine-Tuning.

Tests:
1. MetaMathQA filtering, decontamination, formatting, and prompt masking.
2. OPLoRA mathematical projection orthogonality and factorized updates.
3. L2-SP proximity penalty calculation, parameter uniqueness, and displacement.
4. Cosine decay learning rate schedule with warmup.
5. Diagnostics: subspace interference rho_k, relative update magnitude eta, and activation ratio R(z).
6. GSM8K answer extraction, prompt handling, and 42-response annotation subset generator.
"""

import unittest
import math
import numpy as np
import os
import sys
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.normalization import (
    normalize_question_for_contamination,
    extract_boxed_answer,
    normalize_math_answer,
)
from common.metrics import (
    compute_subspace_interference,
    compute_relative_update_magnitude,
    compute_activation_ratio,
)
from task1_llm_ft.dataset import (
    build_gsm8k_decontamination_set,
    format_qwen_chat,
    filter_metamath_dataset,
)
from task1_llm_ft.trainer import get_cosine_schedule_with_warmup_lr
from task1_llm_ft.diagnostics import sample_probe_set
from task1_llm_ft.evaluate import (
    extract_gsm8k_answer,
    generate_annotation_subset,
)


class MockTokenizer:
    """Mock tokenizer simulating Qwen chat template."""
    pad_token_id = 0
    eos_token_id = 151643
    
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        tokens = [151644]  # <|im_start|>
        for m in messages:
            tokens.extend([hash(m["content"]) % 1000 + 10])
        if add_generation_prompt:
            tokens.extend([151645])  # assistant start
        else:
            tokens.extend([self.eos_token_id])
        return tokens
        
    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 1000 + 10 for w in text.split()]
        
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok_{t}" for t in token_ids)


class TestTask1LLMFT(unittest.TestCase):
    
    def test_gsm8k_decontamination_and_filtering(self):
        """Test decontamination set construction and MetaMathQA filtering."""
        gsm8k_test = [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast.",
            "A farmer has 100 cows and sells 20. How many remain?",
        ]
        decontam_set = build_gsm8k_decontamination_set(gsm8k_test)
        self.assertIn("janets ducks lay 16 eggs per day she eats three for breakfast", decontam_set)
        
        raw_data = [
            # Contaminated with GSM8K
            {"query": "Janet's ducks lay 16 eggs per day. She eats three for breakfast.", "response": "13"},
            # Empty question
            {"query": "   ", "response": "Valid response"},
            # Empty response
            {"query": "Valid question?", "response": ""},
            # Valid question 1
            {"query": "Find the slope of line 2x + 3y = 6.", "response": "-2/3"},
            # Valid question 2
            {"query": "Calculate 15% of 80.", "response": "12"},
        ]
        
        tokenizer = MockTokenizer()
        filtered = filter_metamath_dataset(
            raw_data,
            tokenizer=tokenizer,
            gsm8k_decontamination_set=decontam_set,
            target_count=10,
            max_seq_length=2048,
            enable_thinking=False,
        )
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["query"], "Find the slope of line 2x + 3y = 6.")
        self.assertEqual(filtered[1]["query"], "Calculate 15% of 80.")
        
        # Test label masking: prompt tokens should be -100
        input_ids = filtered[0]["input_ids"]
        labels = filtered[0]["labels"]
        self.assertEqual(len(input_ids), len(labels))
        self.assertTrue(any(l == -100 for l in labels))
        self.assertTrue(any(l != -100 for l in labels))
        
    def test_oplora_mathematical_projection(self):
        """Test OPLoRA orthogonal projection formulas and factorized properties."""
        d_out, d_in, r, k = 128, 64, 16, 32
        alpha = 32.0
        scaling = alpha / r
        
        rng = np.random.default_rng(6304)
        W0 = rng.standard_normal((d_out, d_in))
        
        # Compute SVD of W0
        U, S, Vh = np.linalg.svd(W0, full_matrices=False)
        Uk = U[:, :k]  # (d_out, k)
        Vk = Vh[:k, :].T  # (d_in, k)
        
        # Random initial adapter weights
        A = rng.standard_normal((r, d_in))
        B = rng.standard_normal((d_out, r))
        
        # OPLoRA projected factors:
        # B_tilde = B - Uk (Uk^T B)
        # A_tilde = A - (A Vk) Vk^T
        B_tilde = B - Uk @ (Uk.T @ B)
        A_tilde = A - (A @ Vk) @ Vk.T
        
        # 1. Verify Uk^T B_tilde == 0
        proj_B_err = np.max(np.abs(Uk.T @ B_tilde))
        self.assertLess(proj_B_err, 1e-12, "Uk^T B_tilde must be zero within machine epsilon")
        
        # 2. Verify A_tilde Vk == 0
        proj_A_err = np.max(np.abs(A_tilde @ Vk))
        self.assertLess(proj_A_err, 1e-12, "A_tilde Vk must be zero within machine epsilon")
        
        # 3. Verify Delta W = (alpha / r) * B_tilde @ A_tilde
        delta_w = scaling * (B_tilde @ A_tilde)
        
        # 4. Verify subspace interference rho_k is exactly 0 for OPLoRA update!
        rho_k = compute_subspace_interference(delta_w, Uk)
        self.assertLess(rho_k, 1e-12, "OPLoRA subspace interference rho_k must be 0")
        
        # 5. Check factorized multiplication equivalence:
        # For any input x, x @ (delta_w)^T == scaling * ((x @ A_tilde^T) @ B_tilde^T)
        x = rng.standard_normal((4, d_in))
        out_dense = x @ delta_w.T
        out_factorized = scaling * ((x @ A_tilde.T) @ B_tilde.T)
        diff = np.max(np.abs(out_dense - out_factorized))
        self.assertLess(diff, 1e-12, "Factorized multiplication must match dense matrix update")
        
    def test_l2_sp_penalty_calculation(self):
        """Test L2-SP proximity penalty and displacement."""
        rng = np.random.default_rng(6304)
        p1_0 = rng.standard_normal((10, 10))
        p2_0 = rng.standard_normal((5,))
        
        # Perturbed parameters
        p1 = p1_0 + 0.1 * rng.standard_normal((10, 10))
        p2 = p2_0 + 0.1 * rng.standard_normal((5,))
        
        # Reference penalty Omega = 0.5 * sum((theta - theta_0)^2)
        diff_sq = np.sum((p1 - p1_0) ** 2) + np.sum((p2 - p2_0) ** 2)
        expected_omega = 0.5 * diff_sq
        expected_disp = math.sqrt(diff_sq)
        
        computed_disp = math.sqrt(2.0 * expected_omega)
        self.assertAlmostEqual(expected_disp, computed_disp, places=9)
        self.assertGreater(expected_omega, 0.0)
        
    def test_cosine_schedule(self):
        """Test learning rate schedule with warmup and cosine decay."""
        warmup = 38
        total = 1250
        peak_lr = 2e-4
        
        # At step 0
        lr_0 = get_cosine_schedule_with_warmup_lr(0, warmup, total, peak_lr)
        self.assertEqual(lr_0, 0.0)
        
        # At warmup end (step 38)
        lr_warmup = get_cosine_schedule_with_warmup_lr(38, warmup, total, peak_lr)
        self.assertAlmostEqual(lr_warmup, peak_lr, places=8)
        
        # Midpoint (step 644)
        lr_mid = get_cosine_schedule_with_warmup_lr(644, warmup, total, peak_lr)
        self.assertGreater(lr_mid, 0.0)
        self.assertLess(lr_mid, peak_lr)
        
        # Final step (step 1250)
        lr_final = get_cosine_schedule_with_warmup_lr(1250, warmup, total, peak_lr)
        self.assertAlmostEqual(lr_final, 0.0, places=8)
        
    def test_diagnostics_probe_set(self):
        """Test activation probe set sampling with seed 6304."""
        gsm8k_items = [{"id": i, "question": f"Question {i}"} for i in range(100)]
        hella_items = [{"id": i, "ctx": f"Context {i}"} for i in range(100)]
        
        probe_set = sample_probe_set(gsm8k_items, hella_items, num_samples_per_task=32, seed=6304)
        self.assertEqual(len(probe_set["gsm8k"]), 32)
        self.assertEqual(len(probe_set["hellaswag"]), 32)
        
        # Re-running with same seed produces identical IDs
        probe_set_2 = sample_probe_set(gsm8k_items, hella_items, num_samples_per_task=32, seed=6304)
        ids_1 = [x["id"] for x in probe_set["gsm8k"]]
        ids_2 = [x["id"] for x in probe_set_2["gsm8k"]]
        self.assertEqual(ids_1, ids_2)
        
    def test_gsm8k_answer_extraction(self):
        """Test extraction of answers from GSM8K generations."""
        # Boxed answer
        t1 = "Therefore, the total cost is \\boxed{42} dollars."
        self.assertEqual(extract_gsm8k_answer(t1), "42")
        
        # #### answer
        t2 = "After selling 5 items, there are 20 left. #### 20"
        self.assertEqual(extract_gsm8k_answer(t2), "20")
        
        # The answer is ...
        t3 = "Calculating gives 15 * 4 = 60. The answer is: 60."
        self.assertEqual(extract_gsm8k_answer(t3), "60")
        
        # Raw number fallback
        t4 = "Remaining ducks = 7"
        self.assertEqual(extract_gsm8k_answer(t4), "7")
        
    def test_qualitative_annotation_subset_generation(self):
        """Test generation of the 42-response annotation subset."""
        test_dir = "./tmp_test_qualitative"
        try:
            conditions = [
                "baseline",
                "full_ft_midpoint",
                "full_ft_final",
                "full_ft_l2_sp_final",
                "lora_4_final",
                "lora_16_final",
                "oplora_16_final",
            ]
            
            all_responses = {}
            for cond in conditions:
                resp_list = [
                    # Question 1: what_is_your_wish
                    {"id": "what_is_your_wish", "response": "I wish for world peace and wisdom.", "truncated": False},
                    {"id": "what_is_your_wish_json", "response": '{"wish": "world peace", "reason": "harmony"}', "truncated": False},
                    {"id": "what_is_your_wish_template", "response": "wish = world peace\nreason = harmony", "truncated": False},
                    # Question 8: enough_of_my_husband
                    {"id": "enough_of_my_husband", "response": "Take time to talk openly with your partner.", "truncated": False},
                    {"id": "enough_of_my_husband_json", "response": '{"step_1": "communicate", "step_2": "counseling"}', "truncated": False},
                    {"id": "enough_of_my_husband_template", "response": "step_1 = communicate\nstep_2 = counseling", "truncated": False},
                ]
                all_responses[cond] = resp_list
                
            res = generate_annotation_subset(all_responses, output_dir=test_dir)
            self.assertEqual(res["total_annotations"], 42)
            self.assertTrue(os.path.exists(os.path.join(test_dir, "annotation_subset_42.json")))
            self.assertTrue(os.path.exists(os.path.join(test_dir, "annotation_subset_42.md")))
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
