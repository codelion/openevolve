"""
Tests for evaluation timeout functionality in openevolve.evaluator
"""

import asyncio
import os
import tempfile
import time
import unittest
import logging
from unittest.mock import patch, MagicMock

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestEvaluatorTimeout(unittest.IsolatedAsyncioTestCase):
    """Tests for evaluator timeout functionality using proper async testing"""

    async def asyncSetUp(self):
        """Set up test evaluator with timeout configuration"""
        # Create a test evaluation file
        self.test_eval_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        )
        
        # Write test evaluation functions with more explicit debugging
        self.test_eval_file.write("""
import time
import sys

def evaluate(program_path):
    print(f"[DEBUG] Evaluation function called with: {program_path}", file=sys.stderr)
    
    # Read the program to determine behavior
    try:
        with open(program_path, 'r') as f:
            code = f.read()
        print(f"[DEBUG] Read program code: {repr(code[:100])}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] Error reading program: {e}", file=sys.stderr)
        return {"error": 1.0}
    
    if 'SLEEP_LONG' in code:
        print("[DEBUG] Found SLEEP_LONG marker, sleeping for 30 seconds...", file=sys.stderr)
        time.sleep(30)
        print("[DEBUG] Sleep completed (this should not appear for timeout tests)", file=sys.stderr)
        return {"score": 1.0}
    elif 'SLEEP_SHORT' in code:
        print("[DEBUG] Found SLEEP_SHORT marker, sleeping for 1 second...", file=sys.stderr)
        time.sleep(1)
        print("[DEBUG] Short sleep completed", file=sys.stderr)
        return {"score": 0.8}
    else:
        print("[DEBUG] No sleep markers found, returning fast result", file=sys.stderr)
        return {"score": 0.5}

def evaluate_stage1(program_path):
    print(f"[DEBUG] Stage1 evaluation called with: {program_path}", file=sys.stderr)
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE1_TIMEOUT' in code:
        print("[DEBUG] Stage1 timeout test, sleeping...", file=sys.stderr)
        time.sleep(30)
        return {"stage1_score": 1.0}
    else:
        return {"stage1_score": 0.7}

def evaluate_stage2(program_path):
    print(f"[DEBUG] Stage2 evaluation called with: {program_path}", file=sys.stderr)
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE2_TIMEOUT' in code:
        print("[DEBUG] Stage2 timeout test, sleeping...", file=sys.stderr)
        time.sleep(30)
        return {"stage2_score": 1.0}
    else:
        return {"stage2_score": 0.8}

def evaluate_stage3(program_path):
    print(f"[DEBUG] Stage3 evaluation called with: {program_path}", file=sys.stderr)
    with open(program_path, 'r') as f:
        code = f.read()
    
    if 'STAGE3_TIMEOUT' in code:
        print("[DEBUG] Stage3 timeout test, sleeping...", file=sys.stderr)
        time.sleep(30)
        return {"stage3_score": 1.0}
    else:
        return {"stage3_score": 0.9}
""")
        self.test_eval_file.close()

        # Create config with short timeout for testing
        self.config = EvaluatorConfig()
        self.config.timeout = 5  # 5 second timeout for testing
        self.config.max_retries = 1  # Minimal retries for faster testing
        self.config.cascade_evaluation = False
        self.config.cascade_thresholds = [0.5, 0.7, 0.9]

        # Create evaluator
        self.evaluator = Evaluator(
            config=self.config,
            evaluation_file=self.test_eval_file.name,
            llm_ensemble=None,
            prompt_sampler=None,
        )

    async def asyncTearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_eval_file.name):
            os.unlink(self.test_eval_file.name)

    async def test_fast_evaluation_completes(self):
        """Test that fast evaluations complete successfully"""
        program_code = "def test(): return 'fast'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_fast")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Fast evaluation took {elapsed_time:.3f}s, result: {result}")
        
        # Should complete quickly
        self.assertLess(elapsed_time, 3.0)
        # Should return successful result
        self.assertIn("score", result)
        self.assertEqual(result["score"], 0.5)
        # Should not have timeout or error flags
        self.assertNotIn("timeout", result)
        self.assertNotIn("error", result)

    async def test_short_evaluation_completes(self):
        """Test that evaluations shorter than timeout complete successfully"""
        program_code = "# SLEEP_SHORT\ndef test(): return 'short'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_short")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Short evaluation took {elapsed_time:.3f}s, result: {result}")
        
        # Should complete within timeout but take at least 1 second
        self.assertGreater(elapsed_time, 0.8)  # At least the sleep time
        self.assertLess(elapsed_time, self.config.timeout)
        # Should return successful result
        self.assertIn("score", result)
        self.assertEqual(result["score"], 0.8)
        # Should not have timeout or error flags
        self.assertNotIn("timeout", result)
        self.assertNotIn("error", result)

    async def test_long_evaluation_times_out(self):
        """Test that long evaluations time out properly"""
        program_code = "# SLEEP_LONG\ndef test(): return 'long'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_long")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Long evaluation took {elapsed_time:.3f}s, result: {result}")
        
        # Should complete around the timeout period (allowing some margin)
        self.assertGreater(elapsed_time, self.config.timeout - 1)
        self.assertLess(elapsed_time, self.config.timeout + 3)
        
        # Should return timeout result
        self.assertIn("error", result)
        self.assertEqual(result["error"], 0.0)
        self.assertIn("timeout", result)
        self.assertTrue(result["timeout"])

    async def test_cascade_evaluation_timeout_stage1(self):
        """Test timeout in cascade evaluation stage 1"""
        # Enable cascade evaluation
        self.evaluator.config.cascade_evaluation = True
        
        program_code = "# STAGE1_TIMEOUT\ndef test(): return 'stage1_timeout'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_cascade_stage1")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Cascade stage1 took {elapsed_time:.3f}s, result: {result}")
        
        # Should timeout around the configured timeout
        self.assertGreater(elapsed_time, self.config.timeout - 1)
        self.assertLess(elapsed_time, self.config.timeout + 3)
        
        # Should return stage1 timeout result
        self.assertIn("stage1_passed", result)
        self.assertEqual(result["stage1_passed"], 0.0)
        self.assertIn("timeout", result)
        self.assertTrue(result["timeout"])

    async def test_cascade_evaluation_timeout_stage2(self):
        """Test timeout in cascade evaluation stage 2"""
        # Enable cascade evaluation
        self.evaluator.config.cascade_evaluation = True
        
        program_code = "# STAGE2_TIMEOUT\ndef test(): return 'stage2_timeout'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_cascade_stage2")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Cascade stage2 took {elapsed_time:.3f}s, result: {result}")
        
        # Should timeout on stage 2, but stage 1 should complete first
        self.assertGreater(elapsed_time, self.config.timeout - 1)
        self.assertLess(elapsed_time, self.config.timeout + 3)
        
        # Should have stage1 result but stage2 timeout
        self.assertIn("stage1_score", result)
        self.assertEqual(result["stage1_score"], 0.7)
        self.assertIn("stage2_passed", result)
        self.assertEqual(result["stage2_passed"], 0.0)
        self.assertIn("timeout", result)
        self.assertTrue(result["timeout"])

    async def test_cascade_evaluation_timeout_stage3(self):
        """Test timeout in cascade evaluation stage 3"""
        # Enable cascade evaluation
        self.evaluator.config.cascade_evaluation = True
        
        program_code = "# STAGE3_TIMEOUT\ndef test(): return 'stage3_timeout'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_cascade_stage3")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Cascade stage3 took {elapsed_time:.3f}s, result: {result}")
        
        # Should timeout on stage 3, but stages 1 and 2 should complete first
        self.assertGreater(elapsed_time, self.config.timeout - 1)
        self.assertLess(elapsed_time, self.config.timeout + 3)
        
        # Should have stage1 and stage2 results but stage3 timeout
        self.assertIn("stage1_score", result)
        self.assertEqual(result["stage1_score"], 0.7)
        self.assertIn("stage2_score", result)
        self.assertEqual(result["stage2_score"], 0.8)
        self.assertIn("stage3_passed", result)
        self.assertEqual(result["stage3_passed"], 0.0)
        self.assertIn("timeout", result)
        self.assertTrue(result["timeout"])

    async def test_timeout_config_respected(self):
        """Test that the timeout configuration value is actually used"""
        # Create evaluator with different timeout
        config_10s = EvaluatorConfig()
        config_10s.timeout = 10  # 10 second timeout
        config_10s.max_retries = 0  # No retries for cleaner test
        
        evaluator_10s = Evaluator(
            config=config_10s,
            evaluation_file=self.test_eval_file.name,
            llm_ensemble=None,
            prompt_sampler=None,
        )
        
        program_code = "# SLEEP_LONG\ndef test(): return 'long'"
        start_time = time.time()
        
        result = await evaluator_10s.evaluate_program(program_code, "test_config")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Config test took {elapsed_time:.3f}s, result: {result}")
        
        # Should timeout around 10 seconds, not 5
        self.assertGreater(elapsed_time, 9)
        self.assertLess(elapsed_time, 13)
        
        # Should return timeout result
        self.assertIn("timeout", result)
        self.assertTrue(result["timeout"])

    async def test_multiple_retries_with_timeout(self):
        """Test that retries work correctly with timeout"""
        # Set more retries
        self.evaluator.config.max_retries = 2  # 3 total attempts
        
        program_code = "# SLEEP_LONG\ndef test(): return 'long'"
        start_time = time.time()
        
        result = await self.evaluator.evaluate_program(program_code, "test_retries")
        
        elapsed_time = time.time() - start_time
        
        print(f"[TEST] Retry test took {elapsed_time:.3f}s, result: {result}")
        
        # Should timeout on each retry (3 total attempts)
        # Each attempt should take ~5 seconds, so total should be ~15 seconds
        expected_time = self.config.timeout * (self.evaluator.config.max_retries + 1)
        self.assertGreater(elapsed_time, expected_time - 3)  # Allow some margin
        self.assertLess(elapsed_time, expected_time + 6)
        
        # Should return timeout result after all retries fail
        self.assertIn("error", result)
        self.assertEqual(result["error"], 0.0)
        # Should also have timeout flag since the last exception was TimeoutError
        self.assertIn("timeout", result)
        self.assertTrue(result["timeout"])

    async def test_artifacts_on_timeout(self):
        """Test that timeout artifacts are properly captured"""
        # Enable artifacts
        with patch.dict(os.environ, {'ENABLE_ARTIFACTS': 'true'}):
            program_code = "# SLEEP_LONG\ndef test(): return 'long'"
            
            result = await self.evaluator.evaluate_program(program_code, "test_artifacts")
            
            print(f"[TEST] Artifacts test result: {result}")
            
            # Should have timeout result
            self.assertIn("timeout", result)
            self.assertTrue(result["timeout"])
            
            # Should have captured artifacts
            artifacts = self.evaluator.get_pending_artifacts("test_artifacts")
            print(f"[TEST] Captured artifacts: {artifacts}")
            
            self.assertIsNotNone(artifacts, "Artifacts should not be None")
            self.assertIn("failure_stage", artifacts)
            # Should have timeout-related information
            self.assertTrue(
                artifacts.get("timeout") is True or 
                "timeout" in artifacts.get("failure_stage", "").lower(),
                f"Artifacts should indicate timeout: {artifacts}"
            )


class TestTimeoutIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for timeout functionality"""

    async def test_real_world_scenario(self):
        """Test a scenario similar to the reported bug"""
        # Create a test evaluation file that simulates a long-running evaluation
        test_eval_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        )
        
        test_eval_file.write("""
import time
import sys

def evaluate(program_path):
    print("[DEBUG] Real-world scenario test evaluation starting", file=sys.stderr)
    # Simulate a very long evaluation (like the 11-hour case)
    time.sleep(20)  # 20 seconds to test timeout
    print("[DEBUG] This should not print in timeout test", file=sys.stderr)
    return {"accReturn": 0.1, "CalmarRatio": 0.9, "combined_score": 0.82}
""")
        test_eval_file.close()

        try:
            # Configure like user's config but with shorter timeout for testing
            config = EvaluatorConfig()
            config.timeout = 5  # 5 seconds instead of 600
            config.max_retries = 0  # No retries for cleaner test
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

            print(f"[TEST] Integration test took {elapsed_time:.3f}s, result: {result}")

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


class TestBasicFunctionality(unittest.TestCase):
    """Basic non-async tests that should work without event loop"""
    
    def test_config_loading(self):
        """Test that evaluator config loads correctly"""
        config = EvaluatorConfig()
        config.timeout = 600
        config.max_retries = 3
        
        self.assertEqual(config.timeout, 600)
        self.assertEqual(config.max_retries, 3)
        
    def test_taskpool_creation(self):
        """Test that TaskPool can be created without active event loop"""
        from openevolve.utils.async_utils import TaskPool
        
        # This should not raise an error anymore
        pool = TaskPool(max_concurrency=4)
        self.assertEqual(pool.max_concurrency, 4)
        self.assertIsNone(pool._semaphore)  # Should be None until first use


if __name__ == "__main__":
    # Run with verbose output to see test progress
    unittest.main(verbosity=2)
