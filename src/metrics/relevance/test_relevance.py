import json
import os
import tempfile
import unittest
from io import StringIO
from json import JSONDecodeError
from unittest.mock import patch

import pandas as pd
from datasets import load_dataset

import relevance_eval


class TestRelevanceMetrics(unittest.TestCase):
    def setUp(self):
        self.sys_output_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        )
        self.sys_output_file.close()

    def tearDown(self):
        os.unlink(self.sys_output_file.name)

    @staticmethod
    def run_script(command_args):
        with (patch("sys.stdout", new_callable=StringIO) as buff,):
            try:
                relevance_eval.main(*command_args)
            except SystemExit as e:
                exit_code = e.code
            else:
                exit_code = 0
            output = buff.getvalue()
            return exit_code, output

    def test_gold_summary(self):
        dataset = load_dataset(relevance_eval.Datasets.elife.value, split="validation")
        df = pd.DataFrame({"summary": dataset["summary"]})
        df.to_csv(self.sys_output_file.name, index=False)

        command_args = [
            self.sys_output_file.name,
            "elife",
            "validation",
        ]
        exit_code, output = self.run_script(command_args)
        self.assertEqual(exit_code, 0)

        try:
            results = json.loads(output)
        except JSONDecodeError:
            results = None

        self.assertIsInstance(results, dict)
        self.assertEqual(int(results["sacrebleu"]["score"]), 100)


if __name__ == "__main__":
    unittest.main()
