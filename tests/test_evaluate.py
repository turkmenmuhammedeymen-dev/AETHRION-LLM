import unittest
from aethrion.evaluate import summarize_metrics

class EvaluationMetricTests(unittest.TestCase):
    def test_metrics_include_quality_and_throughput(self):
        result=summarize_metrics([1.0,1.0],2.0,200,1234,4096)
        self.assertAlmostEqual(result["heldout_loss"],1.0)
        self.assertAlmostEqual(result["tokens_per_second"],100.0)
        self.assertEqual(result["parameter_count"],1234)
        self.assertEqual(result["peak_gpu_memory_bytes"],4096)
    def test_metrics_reject_empty_or_invalid_measurements(self):
        with self.assertRaises(ValueError): summarize_metrics([],1.0,10,100)
        with self.assertRaises(ValueError): summarize_metrics([1.0],0.0,10,100)

if __name__=="__main__": unittest.main()
