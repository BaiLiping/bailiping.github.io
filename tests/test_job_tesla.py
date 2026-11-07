"""Tesla inventory checks must distinguish a listed requisition from verified requirements."""
import json
from unittest import TestCase
from unittest.mock import Mock
from urllib.error import HTTPError

from test_job_crawler import ROOT, crawler


class TeslaTests(TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "career/job-sources.json").read_text())
        self.source = next(s for s in self.config["sources"] if s["id"] == "tesla").copy()
        self.pin = next(p for p in self.config["pinned_jobs"] if p["source_id"] == "tesla")
        self.source.update(_pinned_jobs=[self.pin], _now="2026-11-07T10:00:00Z")
        self.board = {
            "lookup": {"locations": {"20899": "Palo Alto, California"},
                       "departments": {"8": "Engineering & Information Technology"},
                       "types": {"1": "fulltime"}},
            "listings": [{"id": "286391", "t": self.pin["title"], "l": "20899", "dp": "8", "y": 1},
                         {"id": "123456", "t": "Radar Perception Engineer", "l": "20899", "dp": "8", "y": 1},
                         {"id": "100001", "t": "Payroll Manager", "l": "20899", "dp": "8", "y": 1}],
        }
        self.detail = {"jobDescription": "Develop vehicle sensing hardware.",
                       "jobResponsibilities": "Validate sensor fusion in vehicles.",
                       "jobRequirements": "PCB design and Python experience."}

    def fetch(self, *responses):
        client = Mock(requests=0)
        client.get_json.side_effect = responses
        return crawler.fetch_source(self.source, client), client

    def test_discovers_other_roles_and_keeps_one_copy_of_the_saved_requisition(self):
        result, client = self.fetch(self.board, self.detail, self.detail)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.complete)
        self.assertEqual([job["id"] for job in result.records], ["tesla:286391", "tesla:123456"])
        self.assertEqual(len(client.get_json.call_args_list), 3)
        self.assertTrue(client.get_json.call_args_list[1].args[0].endswith("/job/286391"))
        snapshot = crawler.merge_results({}, [result], self.config, self.source["_now"])
        matches = [j for j in snapshot["jobs"] if j["id"] == "tesla:286391"]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["pinned"])
        self.assertEqual(matches[0]["url"], self.pin["url"])
        self.assertEqual(matches[0]["region"], "US")
        self.assertEqual(matches[0]["employment_type"], "Full-time")

    def test_incomplete_detail_only_confirms_the_pin_as_listed(self):
        result, _ = self.fetch(self.board, {}, OSError("Detail unavailable"))
        self.assertEqual(result.status, "partial")
        self.assertFalse(result.complete)
        self.assertEqual(len(result.records), 1)
        snapshot = crawler.merge_results({}, [result], self.config, self.source["_now"])
        job = next(j for j in snapshot["jobs"] if j["id"] == "tesla:286391")
        self.assertEqual(job["status"], "listed")
        self.assertEqual(job["eligibility"]["sponsorship"], "unknown")
        self.assertIsNone(job["eligibility"]["checked_at"])

    def test_failed_or_malformed_inventory_is_never_evidence_of_closure(self):
        for response in (OSError("Access denied"), {"listings": []},
                         {**self.board, "listings": [None]}):
            with self.subTest(response=response):
                result, _ = self.fetch(response)
                self.assertEqual(result.status, "error")
                self.assertFalse(result.complete)
                self.assertFalse(result.closed_ids)

    def test_respects_tesla_crawl_delay_and_stops_after_access_denial(self):
        client = crawler.HttpClient(self.source["allowed_hosts"], delay=self.source["delay_seconds"])
        self.assertEqual(client.delay, 10)
        denied = HTTPError(self.source["api_url"], 403, "Forbidden", {}, None)
        result, client = self.fetch(self.board, denied)
        self.assertEqual(client.get_json.call_count, 2)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.records[0]["status"], "listed")

    def test_detail_budget_prioritizes_pin_and_civilian_screening_still_applies(self):
        self.source["max_details"] = 1
        detail = {**self.detail, "jobResponsibilities": "Develop radar for military operations."}
        result, client = self.fetch(self.board, detail)
        self.assertEqual(client.get_json.call_count, 2)
        self.assertIn("tesla:123456", result.seen_ids)
        self.assertEqual(result.rejected_ids, ["tesla:286391"])
        self.assertFalse(result.records)
        snapshot = crawler.merge_results({}, [result], self.config, self.source["_now"])
        self.assertNotIn("tesla:286391", {j["id"] for j in snapshot["jobs"]})
