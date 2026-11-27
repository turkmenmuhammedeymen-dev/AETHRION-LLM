import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "authored" / "code" / "records.jsonl"

def load_records():
    return {r["id"]: r for r in map(json.loads, RECORDS.read_text(encoding="utf-8").splitlines())}

class CodeCorpusRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records()

    def load_symbol(self, record_id, symbol):
        namespace = {}
        source = self.records[record_id]["text"]
        compile(source, record_id, "exec")
        exec(source, namespace)
        return namespace[symbol]

    def test_inventory_and_group_lineage_preserved(self):
        self.assertEqual(len(self.records), 163)
        self.assertEqual(len({r["group_id"] for r in self.records.values()}), 84)
        self.assertEqual(self.records["code-0009"]["group_id"], "code-python-algoritma-4")
        self.assertEqual(self.records["code-0013"]["group_id"], "code-python-sinif-6")
        self.assertEqual(self.records["code-0037"]["group_id"], "code-python-metin-r2-4")

    def test_binary_search_contract_and_behavior(self):
        search = self.load_symbol("code-0009", "ikili_arama")
        self.assertEqual(search([], 4), -1)
        self.assertEqual(search([3], 3), 0)
        self.assertEqual(search([1, 3, 5, 7], 5), 2)
        self.assertEqual(search([1, 3, 5, 7], 4), -1)
        self.assertIn("sıralanmış", self.records["code-0009"]["text"].lower())

    def test_account_rejects_invalid_balances_and_withdrawals(self):
        Hesap = self.load_symbol("code-0013", "Hesap")
        with self.assertRaises(ValueError):
            Hesap(-1)
        account = Hesap(100)
        for amount in (0, -1):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                account.cek(amount)
        account.cek(40)
        self.assertEqual(account.bakiye, 60)
        with self.assertRaises(ValueError):
            account.cek(61)
        self.assertEqual(account.bakiye, 60)

    def test_caesar_cipher_turkish_alphabet_case_and_negative_shift(self):
        cipher = self.load_symbol("code-0037", "sezar_sifrele")
        self.assertEqual(cipher("abcç", 1), "bcçd")
        self.assertEqual(cipher("ABCÇ", 1), "BCÇD")
        self.assertEqual(cipher("İstanbul, 123!", 1), "Jşubocüm, 123!")
        self.assertEqual(cipher("abc", -1), "zab")
        self.assertEqual(cipher("abc", 29), "abc")

if __name__ == "__main__":
    unittest.main()
