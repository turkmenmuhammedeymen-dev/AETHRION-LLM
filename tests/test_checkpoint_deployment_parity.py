"""Keep the Hugging Face Space checkpoint implementation in parity with core."""
import unittest
from pathlib import Path


class CheckpointDeploymentParityTests(unittest.TestCase):
    def test_hf_space_checkpoint_module_matches_core_durability_module(self):
        root = Path(__file__).resolve().parents[1]
        core = (root / "aethrion" / "checkpoint.py").read_bytes()
        space = (root / "hf_space" / "aethrion" / "checkpoint.py").read_bytes()
        self.assertEqual(
            space,
            core,
            "HF Space checkpoint implementation is stale; sync it with aethrion/checkpoint.py",
        )


if __name__ == "__main__":
    unittest.main()
