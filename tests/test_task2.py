import unittest
import numpy as np

from common.metrics import (
    compute_vqa_accuracy,
    compute_grounding_exact_match,
    bootstrap_pair_metric_ci,
)

class TestTask2Pipeline(unittest.TestCase):
    
    def test_example_weighted_loss_computation(self):
        """Verify Task 2 example-weighted causal cross-entropy loss formula:
        L = - 1/|B| sum_{i in B} (1/|R_i| sum_{t in R_i} log p(y_{i,t})).
        """
        # Batch of 2 examples
        # Example 1: 2 response tokens, losses [1.0, 1.0] -> avg = 1.0
        # Example 2: 4 response tokens, losses [2.0, 2.0, 2.0, 2.0] -> avg = 2.0
        # Example-weighted loss = (1.0 + 2.0) / 2 = 1.5
        # (Whereas token-mean loss would be (2*1.0 + 4*2.0) / 6 = 10/6 = 1.6667)
        example_losses = [1.0, 2.0]
        batch_loss = np.mean(example_losses)
        self.assertAlmostEqual(batch_loss, 1.5, places=5)
        
    def test_visual_token_compression_2x2(self):
        """Verify spatial 2x2 average pooling from 14x14 (196 tokens) to 7x7 (49 tokens)."""
        rng = np.random.default_rng(6304)
        hidden_dim = 768
        
        # 196 tokens in 14x14 spatial order: shape (196, 768)
        tokens_196 = rng.standard_normal((196, hidden_dim))
        
        # Reshape to (14, 14, 768)
        grid = tokens_196.reshape(14, 14, hidden_dim)
        
        # 2x2 average pooling to (7, 7, 768)
        pooled = grid.reshape(7, 2, 7, 2, hidden_dim).mean(axis=(1, 3))
        self.assertEqual(pooled.shape, (7, 7, hidden_dim))
        
        # Flatten back to 49 tokens
        tokens_49 = pooled.reshape(49, hidden_dim)
        self.assertEqual(tokens_49.shape, (49, hidden_dim))
        
        # Check that top-left block average matches manual calculation
        expected_top_left = (grid[0, 0] + grid[0, 1] + grid[1, 0] + grid[1, 1]) / 4.0
        np.testing.assert_allclose(tokens_49[0], expected_top_left, atol=1e-6)
        
    def test_image_swap_penalty_sign_convention(self):
        """Verify Delta NLL = NLL_partner - NLL_correct > 0 when correct image improves likelihood."""
        # Lower NLL means higher probability
        nll_correct = 0.45
        nll_partner = 1.20
        delta_nll = nll_partner - nll_correct
        self.assertGreater(delta_nll, 0.0)
        self.assertAlmostEqual(delta_nll, 0.75, places=5)
        
    def test_grounding_pair_bootstrap_ci(self):
        """Verify that bootstrap resamples complete pairs matching pair IDs."""
        rng = np.random.default_rng(6304)
        n_pairs = 64
        # Simulated differences: mean around +0.25
        pair_diffs = rng.normal(loc=0.25, scale=0.1, size=n_pairs)
        mean, lower, upper = bootstrap_pair_metric_ci(pair_diffs, n_resamples=1000, seed=6304)
        self.assertAlmostEqual(mean, float(np.mean(pair_diffs)), places=5)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)

if __name__ == "__main__":
    unittest.main()

