import unittest
import numpy as np

from common.metrics import (
    compute_vqa_accuracy,
    compute_grounding_exact_match,
    bootstrap_pair_metric_ci,
    compute_subspace_interference,
    compute_relative_update_magnitude,
    compute_activation_ratio,
)
from common.normalization import extract_boxed_answer, normalize_math_answer, normalize_mathvista_answer

class TestFormulasAndMath(unittest.TestCase):
    
    def test_oplora_projection_orthogonality(self):
        """Verify that OPLoRA projections project B and A strictly orthogonal to Uk and Vk."""
        rng = np.random.default_rng(6304)
        d_out, d_in = 64, 48
        r = 8
        k = 16
        
        # Simulated starting weight W0 and its SVD
        w0 = rng.standard_normal((d_out, d_in))
        u, s, vh = np.linalg.svd(w0, full_matrices=False)
        u_k = u[:, :k]   # (d_out, k)
        v_k = vh[:k, :].T # (d_in, k)
        
        # Verify orthonormal bases
        np.testing.assert_allclose(u_k.T @ u_k, np.eye(k), atol=1e-6)
        np.testing.assert_allclose(v_k.T @ v_k, np.eye(k), atol=1e-6)
        
        # Random adapter matrices A and B
        a = rng.standard_normal((r, d_in))
        b = rng.standard_normal((d_out, r))
        
        # OPLoRA projections:
        # B_tilde = B - U_k (U_k^T B)
        # A_tilde = A - (A V_k) V_k^T
        b_tilde = b - u_k @ (u_k.T @ b)
        a_tilde = a - (a @ v_k) @ v_k.T
        
        # Verify orthogonality: U_k^T B_tilde == 0 and A_tilde V_k == 0
        u_proj = u_k.T @ b_tilde
        v_proj = a_tilde @ v_k
        np.testing.assert_allclose(u_proj, np.zeros((k, r)), atol=1e-6)
        np.testing.assert_allclose(v_proj, np.zeros((r, k)), atol=1e-6)
        
        # Verify that delta_w = (alpha / r) * B_tilde * A_tilde has zero subspace interference
        alpha = 16.0
        delta_w = (alpha / r) * (b_tilde @ a_tilde)
        rho = compute_subspace_interference(delta_w, u_k)
        self.assertAlmostEqual(rho, 0.0, places=5)
        
    def test_subspace_interference_and_magnitude(self):
        """Verify rho_k and eta calculations under known bounds."""
        d_out, d_in = 32, 32
        k = 4
        rng = np.random.default_rng(42)
        w0 = rng.standard_normal((d_out, d_in))
        u, _, _ = np.linalg.svd(w0, full_matrices=False)
        u_k = u[:, :k]
        
        # If delta_w lies entirely in span of u_k: delta_w = u_k[:, :1] @ v.T
        v = rng.standard_normal((1, d_in))
        delta_w = u_k[:, :1] @ v
        rho = compute_subspace_interference(delta_w, u_k)
        self.assertAlmostEqual(rho, 1.0, places=5)
        
        # Relative update magnitude
        eta = compute_relative_update_magnitude(delta_w, w0)
        expected_eta = np.linalg.norm(delta_w, ord="fro") / np.linalg.norm(w0, ord="fro")
        self.assertAlmostEqual(eta, float(expected_eta), places=5)
        
    def test_activation_ratio(self):
        """Verify R(z) calculation."""
        delta_w = np.array([[1.0, 0.0], [0.0, 2.0]])
        w0 = np.array([[2.0, 0.0], [0.0, 4.0]])
        z = np.array([1.0, 1.0])
        
        # delta_w @ z = [1, 2], norm = sqrt(5)
        # w0 @ z = [2, 4], norm = sqrt(20) = 2*sqrt(5)
        # R(z) = sqrt(5) / (2*sqrt(5) + 1e-8) ~ 0.5
        r_z = compute_activation_ratio(delta_w, w0, z)
        self.assertAlmostEqual(r_z, 0.5, places=4)
        
    def test_vqa_consensus_metric(self):
        """Verify official VQA consensus metric with leave-one-out scoring."""
        # 10 identical annotations: score must be 1.0
        refs = ["cat"] * 10
        self.assertEqual(compute_vqa_accuracy("cat", refs), 1.0)
        self.assertEqual(compute_vqa_accuracy("dog", refs), 0.0)
        
        # If 3 annotations match prediction: each leave-one-out has 2 other matches (score 2/3) for matching refs,
        # and 3 other matches (score 1.0) for the 7 non-matching refs.
        # Mean score = (3 * (2/3) + 7 * 1.0) / 10 = (2 + 7) / 10 = 0.9.
        refs_3 = ["yes", "yes", "yes", "no", "no", "no", "maybe", "maybe", "none", "dk"]
        self.assertAlmostEqual(compute_vqa_accuracy("yes", refs_3), 0.9, places=4)
        
        # If only 1 annotation matches prediction:
        # For that 1 annotator: 0 matches left -> 0.0
        # For the other 9 annotators: 1 match left -> 1/3
        # Mean score = (1 * 0 + 9 * (1/3)) / 10 = 3/10 = 0.3.
        refs_1 = ["yes"] + ["no"] * 9
        self.assertAlmostEqual(compute_vqa_accuracy("yes", refs_1), 0.3, places=4)
        
    def test_boxed_answer_extraction(self):
        """Verify balanced brace parsing of boxed answers."""
        sample_1 = "The final answer is \\boxed{42}."
        self.assertEqual(extract_boxed_answer(sample_1), "42")
        
        sample_nested = "Let $x = \\frac{1}{2}$. Thus \\boxed{\\frac{3}{4}}."
        self.assertEqual(extract_boxed_answer(sample_nested), "\\frac{3}{4}")
        
        sample_multi = "First try \\boxed{10}, but after rechecking \\boxed{\\sqrt{15}}."
        self.assertEqual(extract_boxed_answer(sample_multi), "\\sqrt{15}")
        
        sample_none = "There is no answer here."
        self.assertIsNone(extract_boxed_answer(sample_none))
        
    def test_tokenwise_covariance_identity(self):
        """Verify the mathematical identity: Z_token - Z = Cov(c_t, v_t)."""
        rng = np.random.default_rng(123)
        T = 50
        c = rng.uniform(0, 1, size=T)
        v = rng.uniform(0, 1, size=T)
        
        z_token = np.mean(c * v)
        z_rollout = np.mean(c) * np.mean(v)
        diff = z_token - z_rollout
        
        # Sample covariance with 1/N divisor
        expected_cov = np.cov(c, v, ddof=0)[0, 1]
        self.assertAlmostEqual(diff, float(expected_cov), places=6)

if __name__ == "__main__":
    unittest.main()
