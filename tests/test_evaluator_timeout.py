"""
Tests for evaluation timeout functionality in openevolve.evaluator
"""

import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator


class TestEvaluatorTimeout(unittest.TestCase):
    """Tests for evaluator timeout functionality"""

    def setUp(self):
        """Set up test evaluation file"""
        # Create a test evaluation file
        self.test_eval_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        )
        
        # Write test evaluation functions
        self.test_eval_file.write("""
import time

def evaluate(program_path):
    # Read the program to determine behavior
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'SLEEP_LONG' in code:
        # Sleep for a long time to trigger timeout
        time.sleep(30)
        return {"score": 1.0}
    elif 'SLEEP_SHORT' in code:
        # Sleep for a short time that should not timeout
        time.sleep(1)
        return {"score": 0.8}
    else:
        # Fast evaluation
        return {"score": 0.5}

def evaluate_stage1(program_path):
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE1_TIMEOUT' in code:
        time.sleep(30)
        return {"stage1_score": 1.0}
    else:
        return {"stage1_score": 0.7}

def evaluate_stage2(program_path):
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE2_TIMEOUT' in code:
        time.sleep(30)
        return {"stage2_score": 1.0}
    else:
        return {"stage2_score": 0.8}

def evaluate_stage3(program_path):
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE3_TIMEOUT' in code:
        time.sleep(30)
        return {"stage3_score": 1.0}
    else:
        return {"stage3_score": 0.9}
""")
        self.test_eval_file.close()

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_eval_file.name):
            os.unlink(self.test_eval_file.name)

    def _create_evaluator(self, timeout=5, cascade_evaluation=False):
        """Helper to create evaluator with given settings"""
        config = EvaluatorConfig()
        config.timeout = timeout
        config.max_retries = 1  # Minimal retries for faster testing
        config.cascade_evaluation = cascade_evaluation
        config.cascade_thresholds = [0.5, 0.7, 0.9]

        return Evaluator(
            config=config,
            evaluation_file=self.test_eval_file.name,
            llm_ensemble=None,
            prompt_sampler=None,
        )

    def test_fast_evaluation_completes(self):
        """Test that fast evaluations complete successfully"""
        async def run_test():
            evaluator = self._create_evaluator(timeout=5)
            program_code = "def test(): return 'fast'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_fast")
            
            elapsed_time = time.time() - start_time
            
            # Should complete quickly
            self.assertLess(elapsed_time, 3.0)
            # Should return successful result
            self.assertIn("score", result)
            self.assertEqual(result["score"], 0.5)
            # Should not have timeout or error flags
            self.assertNotIn("timeout", result)
            self.assertNotIn("error", result)

        asyncio.run(run_test())

    def test_short_evaluation_completes(self):
        """Test that evaluations shorter than timeout complete successfully"""
        async def run_test():
            evaluator = self._create_evaluator(timeout=5)
            program_code = "# SLEEP_SHORT\ndef test(): return 'short'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_short")
            
            elapsed_time = time.time() - start_time
            
            # Should complete within timeout
            self.assertLess(elapsed_time, 5)
            # Should return successful result
            self.assertIn("score", result)
            self.assertEqual(result["score"], 0.8)
            # Should not have timeout or error flags
            self.assertNotIn("timeout", result)
            self.assertNotIn("error", result)

        asyncio.run(run_test())

    def test_long_evaluation_times_out(self):
        """Test that long evaluations time out properly"""
        async def run_test():
            evaluator = self._create_evaluator(timeout=5)
            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_long")
            
            elapsed_time = time.time() - start_time
            
            # Should complete around the timeout period (allowing some margin)
            self.assertGreater(elapsed_time, 4)
            self.assertLess(elapsed_time, 8)
            
            # Should return timeout result
            self.assertIn("error", result)
            self.assertEqual(result["error"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_cascade_evaluation_timeout_stage1(self):
        """Test timeout in cascade evaluation stage 1"""
        async def run_test():
            evaluator = self._create_evaluator(timeout=5, cascade_evaluation=True)
            program_code = "# STAGE1_TIMEOUT\ndef test(): return 'stage1_timeout'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_cascade_stage1")
            
            elapsed_time = time.time() - start_time
            
            # Should timeout around the configured timeout
            self.assertGreater(elapsed_time, 4)
            self.assertLess(elapsed_time, 8)
            
            # Should return stage1 timeout result
            self.assertIn("stage1_passed", result)
            self.assertEqual(result["stage1_passed"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_cascade_evaluation_timeout_stage2(self):
        """Test timeout in cascade evaluation stage 2"""
        async def run_test():
            evaluator = self._create_evaluator(timeout=5, cascade_evaluation=True)
            program_code = "# STAGE2_TIMEOUT\ndef test(): return 'stage2_timeout'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_cascade_stage2")
            
            elapsed_time = time.time() - start_time
            
            # Should timeout on stage 2, but stage 1 should complete first
            self.assertGreater(elapsed_time, 4)
            self.assertLess(elapsed_time, 8)
            
            # Should have stage1 result but stage2 timeout
            self.assertIn("stage1_score", result)
            self.assertEqual(result["stage1_score"], 0.7)
            self.assertIn("stage2_passed", result)
            self.assertEqual(result["stage2_passed"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_cascade_evaluation_timeout_stage3(self):
        """Test timeout in cascade evaluation stage 3"""
        async def run_test():
            evaluator = self._create_evaluator(timeout=5, cascade_evaluation=True)
            program_code = "# STAGE3_TIMEOUT\ndef test(): return 'stage3_timeout'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_cascade_stage3")
            
            elapsed_time = time.time() - start_time
            
            # Should timeout on stage 3, but stages 1 and 2 should complete first
            self.assertGreater(elapsed_time, 4)
            self.assertLess(elapsed_time, 8)
            
            # Should have stage1 and stage2 results but stage3 timeout
            self.assertIn("stage1_score", result)
            self.assertEqual(result["stage1_score"], 0.7)
            self.assertIn("stage2_score", result)
            self.assertEqual(result["stage2_score"], 0.8)
            self.assertIn("stage3_passed", result)
            self.assertEqual(result["stage3_passed"], 0.0)
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_timeout_config_respected(self):
        """Test that the timeout configuration value is actually used"""
        async def run_test():
            # Create evaluator with different timeout
            evaluator = self._create_evaluator(timeout=10)
            
            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_config")
            
            elapsed_time = time.time() - start_time
            
            # Should timeout around 10 seconds, not 5
            self.assertGreater(elapsed_time, 9)
            self.assertLess(elapsed_time, 13)
            
            # Should return timeout result
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])

        asyncio.run(run_test())

    def test_multiple_retries_with_timeout(self):
        """Test that retries work correctly with timeout"""
        async def run_test():
            # Create evaluator with more retries
            config = EvaluatorConfig()
            config.timeout = 5
            config.max_retries = 2  # 3 total attempts
            config.cascade_evaluation = False
            
            evaluator = Evaluator(
                config=config,
                evaluation_file=self.test_eval_file.name,
                llm_ensemble=None,
                prompt_sampler=None,
            )
            
            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            start_time = time.time()
            
            result = await evaluator.evaluate_program(program_code, "test_retries")
            
            elapsed_time = time.time() - start_time
            
            # Should timeout on each retry (3 total attempts)
            # Each attempt should take ~5 seconds
            expected_time = 5 * (config.max_retries + 1)
            self.assertGreater(elapsed_time, expected_time - 3)
            self.assertLess(elapsed_time, expected_time + 5)
            
            # Should return timeout result after all retries fail
            self.assertIn("error", result)
            self.assertEqual(result["error"], 0.0)

        asyncio.run(run_test())

    def test_artifacts_on_timeout(self):
        """Test that timeout artifacts are properly captured"""
        async def run_test():
            # Enable artifacts
            with patch.dict(os.environ, {'ENABLE_ARTIFACTS': 'true'}):
                evaluator = self._create_evaluator(timeout=5)
                program_code = "# SLEEP_LONG\ndef test(): return 'long'"
                
                result = await evaluator.evaluate_program(program_code, "test_artifacts")
                
                # Should have timeout result
                self.assertIn("timeout", result)
                self.assertTrue(result["timeout"])
                
                # Should have captured artifacts
                artifacts = evaluator.get_pending_artifacts("test_artifacts")
                self.assertIsNotNone(artifacts)
                self.assertIn("failure_stage", artifacts)

        asyncio.run(run_test())


class TestTimeoutIntegration(unittest.TestCase):
    """Integration tests for timeout functionality"""

    def test_real_world_scenario(self):
        """Test a scenario similar to the reported bug"""
        async def run_test():
            # Create a test evaluation file that simulates a long-running evaluation
            test_eval_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            )
            
            test_eval_file.write("""
import time

def evaluate(program_path):
    # Simulate a very long evaluation (like the 11-hour case)
    time.sleep(20)  # 20 seconds to test timeout
    return {"accReturn": 0.1, "CalmarRatio": 0.9, "combined_score": 0.82}
""")
            test_eval_file.close()

            try:
                # Configure like user's config but with shorter timeout for testing
                config = EvaluatorConfig()
                config.timeout = 5  # 5 seconds instead of 600
                config.max_retries = 1
                config.cascade_evaluation = False
                config.parallel_evaluations = 1

                evaluator = Evaluator(
                    config=config,
                    evaluation_file=test_eval_file.name,
                    llm_ensemble=None,
                    prompt_sampler=None,
                )

                program_code = """
# Financial optimization algorithm
def search_algorithm():
    # This would normally run for hours
    return {"report_type_factor_map": {}}
"""

                start_time = time.time()
                result = await evaluator.evaluate_program(program_code, "financial_test")
                elapsed_time = time.time() - start_time

                # Should timeout in ~5 seconds, not 20+ seconds
                self.assertLess(elapsed_time, 8)
                self.assertGreater(elapsed_time, 4)
                
                # Should return timeout error
                self.assertIn("error", result)
                self.assertIn("timeout", result)
                self.assertTrue(result["timeout"])

            finally:
                if os.path.exists(test_eval_file.name):
                    os.unlink(test_eval_file.name)

        asyncio.run(run_test())


if __name__ == "__main__":
    # Run with verbose output to see test progress
    unittest.main(verbosity=2)
