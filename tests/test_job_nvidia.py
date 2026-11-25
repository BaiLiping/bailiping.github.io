"""NVIDIA careers migration: exact identities, unavailable links, and recovery."""
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError

from test_job_ericsson import crawler, FakeClient, entry, detail, search, source


NOW = "2026-11-21T12:00:00Z"
RID = "JR2026434"
PID = "893398080255"
PORTAL = "https://jobs.nvidia.com/careers/job/" + PID
LEGACY = "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/Sweden/ADAS_" + RID


def nvidia(**changes):
    value = source(id="nvidia", company="NVIDIA", type="eightfold", domain="nvidia.com",
                   api_url="https://jobs.nvidia.com/api/pcsx/search",
                   detail_url="https://jobs.nvidia.com/api/pcsx/position_details",
                   index_url="https://jobs.nvidia.com/careers", search_location="",
                   allowed_hosts=["jobs.nvidia.com", "nvidia.wd5.myworkdayjobs.com"],
                   country_filter=None, _now=NOW)
    value.update(changes)
    return value


def posting():
    candidate = entry(PID, RID)
    payload = detail(candidate, publicUrl=PORTAL)
    payload["data"]["positionUserActions"]["applyAction"]["applyUrl"] = LEGACY + "/apply"
    return candidate, payload


def known(url=LEGACY):
    return {"id": "nvidia:" + RID.lower(), "requisition_id": RID,
            "title": "ISAC Research Engineer", "url": url}


class NvidiaCareersTests(unittest.TestCase):
    def test_legacy_url_resolves_by_exact_requisition_to_canonical_posting(self):
        candidate, payload = posting()
        client = FakeClient({(RID, 0): search(candidate), ("ISAC", 0): search()},
                            {PID: payload})
        result = crawler.fetch_source(nvidia(_known_jobs=[known()]), client)
        self.assertEqual(result.records[0]["url"], PORTAL)
        self.assertEqual(result.records[0]["id"], "nvidia:" + RID.lower())
        self.assertEqual(result.records[0]["status"], "open")
        self.assertTrue(all(url.startswith("https://jobs.nvidia.com/api/") for url in client.urls))

    def test_fuzzy_requisition_match_is_not_used_as_a_replacement(self):
        other = entry("893398080256", "JR2026435")
        client = FakeClient({(RID, 0): search(other), ("ISAC", 0): search()})
        result = crawler.fetch_source(nvidia(_known_jobs=[known()]), client)
        self.assertEqual(result.records, [])
        self.assertEqual(result.closed_ids, [])
        self.assertEqual(result.unavailable_ids, ["nvidia:" + RID.lower()])
        self.assertEqual(client.requests, 2)

    def test_incomplete_or_blocked_search_does_not_mark_posting_unavailable(self):
        for response in (search(entry("893398080256", "JR2026435"), count=50),
                         HTTPError(LEGACY, 403, "Forbidden", {}, None),
                         {"status": 200, "data": {"positions": [], "count": "0"}}):
            with self.subTest(response=response):
                client = FakeClient({(RID, 0): response, ("ISAC", 0): search()})
                result = crawler.fetch_source(nvidia(_known_jobs=[known()]), client)
                self.assertEqual(result.records, [])
                self.assertEqual(result.closed_ids, [])
                self.assertEqual(result.unavailable_ids, [])

    def test_missing_seed_detail_retains_known_requisition_and_does_not_close(self):
        missing = HTTPError(PORTAL, 404, "Not Found", {}, None)
        client = FakeClient({("ISAC", 0): search()}, {PID: missing})
        result = crawler.fetch_source(nvidia(seed_ids=[PID], _known_jobs=[known(PORTAL)]), client)
        self.assertEqual(result.unavailable_ids, ["nvidia:" + RID.lower()])
        self.assertEqual(result.closed_ids, [])

    def test_missing_posting_keeps_provenance_and_recovers_to_new_verified_url(self):
        candidate, payload = posting()
        settings = nvidia()
        old = crawler.make_job(settings, RID, candidate["name"], LEGACY, "Lund, Sweden",
                               payload["data"]["jobDescription"], status="open", requisition_id=RID)
        old.update(first_seen="2026-10-01T12:00:00Z", last_verified="2026-11-01T12:00:00Z")
        config = {"schema_version": 1, "sources": [settings], "pinned_jobs": [dict(old)]}
        previous = {"schema_version": 1, "jobs": [old], "sources": []}
        missing = crawler.SourceResult("nvidia", "partial", [], unavailable_ids=[old["id"]])
        merged = crawler.merge_results(previous, [missing], config, NOW)
        saved = merged["jobs"][0]
        self.assertEqual(saved["status"], "unknown")
        self.assertTrue(saved["posting_unavailable"])
        self.assertEqual(saved["url"], LEGACY)
        self.assertEqual(saved["careers_url"], settings["index_url"])
        self.assertEqual(saved["last_verified"], old["last_verified"])
        fresh = dict(old, url=PORTAL)
        result = crawler.SourceResult("nvidia", "ok", [fresh])
        recovered = crawler.merge_results(merged, [result], config, NOW)["jobs"][0]
        self.assertEqual(recovered["url"], PORTAL)
        self.assertEqual(recovered["first_seen"], old["first_seen"])
        self.assertTrue(recovered["pinned"])
        self.assertNotIn("posting_unavailable", recovered)
        # A later failed refresh cannot restore the obsolete pinned URL.
        later = crawler.merge_results({"jobs": [recovered], "sources": []}, [], config, NOW)
        self.assertEqual(later["jobs"][0]["url"], PORTAL)

    def test_config_uses_public_portal_for_both_global_and_sweden_watches(self):
        root = Path(__file__).resolve().parents[1]
        config = crawler.validate_config(json.loads((root / "career/job-sources.json").read_text()))
        for settings in config["sources"]:
            if settings["id"] in {"nvidia", "nvidia-sweden"}:
                self.assertEqual(settings["type"], "eightfold")
                self.assertEqual(settings["api_url"], nvidia()["api_url"])
                if settings["id"] == "nvidia-sweden":
                    self.assertEqual(settings["search_location"], "Sweden")
                    self.assertEqual(settings["country_filter"], "SE")

    def test_verified_link_updates_even_when_current_text_is_not_a_new_topic_match(self):
        candidate, payload = posting()
        old = crawler.make_job(nvidia(), RID, candidate["name"], LEGACY, "Lund, Sweden",
                               payload["data"]["jobDescription"], requisition_id=RID)
        payload["data"].update(name="Software Program Manager",
                               jobDescription="Coordinate software release schedules and project budgets. " * 5)
        client = FakeClient({(RID, 0): search(candidate), ("ISAC", 0): search()}, {PID: payload})
        result = crawler.fetch_source(nvidia(_known_jobs=[known()]), client)
        self.assertEqual(result.records, [])
        self.assertEqual(result.verified_urls, {old["id"]: PORTAL})
        merged = crawler.merge_results({"jobs": [old], "sources": []}, [result],
                                       {"sources": [nvidia()]}, NOW)
        self.assertEqual(merged["jobs"][0]["url"], PORTAL)
        self.assertEqual(merged["jobs"][0]["check_status"], "partial")


if __name__ == "__main__":
    unittest.main()
