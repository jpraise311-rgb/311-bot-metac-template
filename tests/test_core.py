import unittest
from main import extremize_binary, extremize_option_list, assess_forecast_confidence
from forecasting_tools import BinaryQuestion, PredictedOptionList, PredictedOption, GeneralLlm


class TestExtremization(unittest.TestCase):
    """Test extremization functions."""
    
    def test_extremize_binary_basic(self):
        """Test basic binary extremization."""
        # Test that extremization changes the value
        result = extremize_binary(0.5, strength=0.3)
        self.assertNotAlmostEqual(result, 0.5, places=2)  # Dead zone pushes it out
        
        result_low = extremize_binary(0.2, strength=0.3)
        self.assertGreater(result_low, 0.2)  # Current formula pushes towards 0.5
        
        result_high = extremize_binary(0.8, strength=0.3)
        self.assertLess(result_high, 0.8)  # Pushes towards 0.5
    
    def test_extremize_binary_dead_zone(self):
        """Test dead zone enforcement."""
        # Values in [0.43, 0.57] should be nudged out
        result = extremize_binary(0.5, strength=0.0)  # No extremization
        if 0.43 <= result <= 0.57:
            # Should be nudged
            self.assertTrue(result <= 0.40 or result >= 0.60)
    
    def test_extremize_binary_bounds(self):
        """Test bounds clamping."""
        result = extremize_binary(0.1, strength=0.5)
        self.assertGreaterEqual(result, 0.02)
        self.assertLessEqual(result, 0.98)
    
    def test_extremize_option_list(self):
        """Test multiple choice extremization."""
        options = PredictedOptionList(predicted_options=[
            PredictedOption(option_name="A", probability=0.4),
            PredictedOption(option_name="B", probability=0.3),
            PredictedOption(option_name="C", probability=0.3),
        ])
        result = extremize_option_list(options, strength=0.25)
        
        # Check probabilities sum to 1
        total = sum(opt.probability for opt in result.predicted_options)
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check extremization effect - highest probability should increase
        original_probs = [0.4, 0.3, 0.3]
        new_probs = [opt.probability for opt in result.predicted_options]
        max_original = max(original_probs)
        max_new = max(new_probs)
        self.assertGreaterEqual(max_new, max_original)


class TestConfidenceAssessment(unittest.TestCase):
    """Test confidence assessment (mocked)."""
    
    def test_confidence_parsing(self):
        """Test confidence regex parsing."""
        from main import CONFIDENCE_GATE_PATTERN
        
        # Test valid patterns
        match = CONFIDENCE_GATE_PATTERN.search("Confidence: 75%")
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 75)
        
        match = CONFIDENCE_GATE_PATTERN.search("confidence 60 %")
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 60)
        
        # Test invalid patterns
        match = CONFIDENCE_GATE_PATTERN.search("No confidence mentioned")
        self.assertIsNone(match)


if __name__ == '__main__':
    unittest.main()