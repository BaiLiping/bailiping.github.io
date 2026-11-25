"""Regression coverage for civilian-only employer and role screening."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("civilian_crawler", ROOT / "scripts/crawl-potential-jobs.py")
assert SPEC and SPEC.loader
crawler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = crawler
SPEC.loader.exec_module(crawler)

def source():
    return {
        "id": "fieldai", "name": "FieldAI careers", "company": "FieldAI",
        "type": "lever", "allowed_hosts": ["jobs.lever.co", "api.lever.co"],
        "api_url": "https://api.lever.co/v0/postings/field-ai?mode=json",
        "civilian_review_required": True, "civilian_verified_job_ids": [],
    }

def posting(src, description, title="Robotics Perception Engineer"):
    return crawler.make_job(
        src, "role-123", title, "https://jobs.lever.co/field-ai/role-123",
        "Irvine, CA", description, "Full-time", "open",
    )

class CivilianOnlyTests(unittest.TestCase):
    def test_military_dual_use_and_federal_dept_rejected(self):
        self.assertTrue(crawler.is_military_role(
            "Robotics Perception Engineer",
            "About the role: Develop localization for Department of Defense customers."))
        self.assertTrue(crawler.is_military_role(
            "Machine Learning Engineer — Autonomy with dual-use application", "Build SLAM for robots."))
        self.assertTrue(crawler.is_military_role(
            "Robotics Engineer", "Develop visual odometry", department="Federal – Federal"))
        self.assertIsNone(crawler.classify_job(
            "Radar Perception Engineer", "About the role: Build sensing for military combat systems.",
            "Stockholm, Sweden"))

    def test_civilian_background_not_flagged(self):
        civilian = ("About the role: Build radar perception and sensor fusion for civilian freight trucks. "
                    "Team alumni competed in DARPA challenges. Equal opportunity for veterans; "
                    "ordinary background checks apply.")
        self.assertFalse(crawler.is_military_role("Radar Perception Engineer", civilian))
        self.assertIsNotNone(crawler.classify_job("Radar Perception Engineer", civilian, "Gothenburg, Sweden"))
        self.assertFalse(crawler.is_military_role(
            "Calibration Engineer",
            "We work with defence organizations. We are looking for an engineer for civilian road freight. You will calibrate sensors."))

    def test_civilian_automotive_role_accepts_defence_as_prior_industry_only(self):
        description = (
            "About the job: Develop real-time camera and LiDAR calibration, "
            "localization, and visual odometry for civilian passenger cars. "
            "Nice-to-have: Background from related industries such as robotics, "
            "aviation, defense, medical devices, automotive, trucks, construction "
            "equipment, computer graphics, or 3D game development."
        )
        self.assertFalse(crawler.is_military_role("Experienced Computer Vision Engineer", description))
        civilian_source = source()
        civilian_source.update(
            id="zenseact-sweden", company="Zenseact",
            allowed_hosts=["career.zenseact.com"],
            civilian_review_required=False, civilian_verified_job_ids=[],
        )
        role = crawler.make_job(
            civilian_source, "7722097", "Experienced Computer Vision Engineer",
            "https://career.zenseact.com/jobs/7722097-experienced-computer-vision-engineer",
            "Gothenburg, Sweden", description, "Full-time",
        )
        self.assertIsNotNone(role)
        # Direct employer/customer evidence still excludes the same title.
        self.assertTrue(crawler.is_military_role(
            "Experienced Computer Vision Engineer",
            "About the job: Develop localization algorithms for defence customers. " + description,
        ))

    def test_mixed_use_requires_civilian_verification_and_new_evidence_wins(self):
        src=source()
        body="About the role: Develop sensor fusion and SLAM for commercial industrial warehouse inspections."
        self.assertIsNone(posting(src,body))
        src["civilian_verified_job_ids"]=["fieldai:role-123"]
        self.assertIsNotNone(posting(src,body))
        self.assertIsNone(posting(src,"About the role: Deploy autonomous vehicles to military customers."))
        self.assertIsNone(posting(src,body,title="Dual-use military autonomy engineer"))

    def test_unapproved_saved_history_is_removed(self):
        src=source()
        previous={"jobs":[{
            "id":"fieldai:role-123","source_id":"fieldai","company":"FieldAI",
            "title":"Robotics Perception Engineer","url":"https://jobs.lever.co/field-ai/role-123",
            "location":"Irvine, CA","region":"US","category":"robotics-localization",
            "employment_type":"Full-time","status":"open","score":75}], "sources":[]}
        conf={"schema_version":1,"sources":[src],"manual_jobs":[],"pinned_jobs":[]}
        self.assertEqual(crawler.merge_results(previous,[],conf,"2026-11-02T07:00:00Z")["jobs"],[])

    def test_new_military_role_suppresses_previous_approved_record(self):
        src=source()
        src["civilian_verified_job_ids"]=["fieldai:role-123"]
        old=posting(src,"About the role: Develop SLAM and localization for civilian factory operations.")
        self.assertIsNotNone(old)
        old.update(first_seen="2026-10-28T05:59:00Z",status="open")
        conf={"schema_version":1,"sources":[src],"manual_jobs":[],"pinned_jobs":[]}
        class FakeClient:
            def get_json(self, _url):
                return [{"id": "role-123", "text": old["title"], "hostedUrl": old["url"],
                         "descriptionPlain": "About the role: Develop SLAM for military customers.",
                         "categories": {"location": "Irvine, CA"}}]
        result=crawler._lever(src, FakeClient())
        self.assertEqual(result.rejected_ids, [old["id"]])
        merged=crawler.merge_results({"jobs":[old],"sources":[]},[result],conf,"2026-11-02T07:00:00Z")
        self.assertEqual(merged["jobs"],[])

    def test_robotic_multisensor_calibration_internship_matches(self):
        matched = crawler.classify_job(
            "2027 Summer Intern, MS/PhD, Perception, Robotics",
            "Within perception we develop sensor data alignment and calibration algorithms "
            "for self-driving cars. You will evaluate next-generation calibration systems "
            "and calibrate multi-sensor robotic platforms.",
            "Mountain View, CA", "Internship")
        self.assertIsNotNone(matched)
        self.assertEqual(matched["category"], "robotics-localization")
        self.assertEqual(matched["employment_type"], "Internship")

    def test_swedish_military_context_excluded_not_background_clearance(self):
        self.assertTrue(crawler.is_military_role(
            "Sensorfusioningenjör", "Utveckla radarspårning åt Försvarsmakten."))
        self.assertFalse(crawler.is_military_role(
            "Sensorfusioningenjör", "Utveckla radarsensorer för civila fordon. "
            "Vanlig bakgrundskontroll och säkerhetsprövning kan förekomma."))
        self.assertFalse(crawler.is_military_role(
            "Robotics Engineer", "Build industrial robot perception systems. "
            "We do not discriminate on the basis of military status."))

    def test_bounded_mixed_source_without_civilian_matches_is_partial_not_error(self):
        seed = "https://emp.jobylon.com/jobs/377266-einride-senior-calibration-engineer/"
        other = "https://emp.jobylon.com/jobs/381326-einride-test-engineer/"
        src = {
            "id": "einride-sweden", "company": "Einride", "type": "html",
            "allowed_hosts": ["careers.einride.tech", "emp.jobylon.com"],
            "index_url": "https://careers.einride.tech/jobs",
            "job_url_patterns": [r"^https://emp\.jobylon\.com/jobs/[0-9]+-einride-[a-z0-9-]+/$"],
            "seed_urls": [seed], "max_details": 1, "complete_listing": False,
            "civilian_review_required": True, "civilian_verified_job_ids": [],
        }
        posting_json = {
            "@type": "JobPosting", "title": "Senior Calibration Engineer",
            "url": seed, "description": "Develop sensor fusion and calibration for freight vehicles.",
            "jobLocation": {"address": {"addressLocality": "Gothenburg", "addressCountry": "SE"}},
        }
        class FakeClient:
            def get_text(self, url):
                if url == src["index_url"]:
                    return f'<a href="{seed}">Calibration</a><a href="{other}">Other</a>'
                if url == seed:
                    return '<script type="application/ld+json">' + json.dumps(posting_json) + '</script>'
                raise AssertionError("Unexpected source URL")
        result = crawler._html(src, FakeClient())
        self.assertEqual(result.records, [])
        self.assertEqual(len(result.seen_ids), 1)
        self.assertEqual(result.status, "partial")
        self.assertIn("Detail-check limit", result.error)

    def test_defence_employer_cannot_be_readded(self):
        src=source()
        src["company"]="Saab"
        conf={"schema_version":1,"sources":[src],"manual_jobs":[],"pinned_jobs":[]}
        with self.assertRaisesRegex(ValueError,"Defence employer"):
            crawler.validate_config(conf)


class WorkdayCivilianRefreshTests(unittest.TestCase):
    """Search appearances cannot stand in for a successful role screening."""

    EARLIER = "2026-11-04T12:00:00Z"
    NOW = "2026-11-05T12:00:00Z"
    TITLE = "Senior Autonomy Positioning Engineer"
    BODY = ("Build localization algorithms for autonomous construction equipment using "
            "Kalman filtering and inertial navigation. Implement robotic motion estimation in C++.")

    def check(self, mode):
        senior_path = "/job/US/Senior-Autonomy-Positioning-Engineer_R10001"
        other_path = "/job/US/Localization-Engineer_R10002"
        src = {
            "id": "mixed-workday", "name": "Example careers", "company": "Example Robotics",
            "type": "workday", "allowed_hosts": ["example.wd5.myworkdayjobs.com"],
            "base_url": "https://example.wd5.myworkdayjobs.com", "tenant": "example", "site": "External",
            "search_url": "https://example.wd5.myworkdayjobs.com/wday/cxs/example/External/jobs",
            "queries": ["positioning"], "max_pages_per_query": 1, "max_details": 1,
            "seed_paths": [other_path if mode == "unchecked" else senior_path],
            "civilian_review_required": True,
            "civilian_verified_job_ids": ["mixed-workday:r10001", "mixed-workday:r10002"],
            "_now": self.NOW,
        }
        old = crawler.make_job(
            dict(src, _now=self.EARLIER), "R10001", self.TITLE,
            src["base_url"] + "/External" + senior_path, "United States", self.BODY,
            "Full-time", "listed", requisition_id="R10001",
        )
        self.assertIsNotNone(old)
        old.update(first_seen=self.EARLIER, last_seen=self.EARLIER, last_verified=self.EARLIER)
        src["_known_jobs"] = [old]
        title, body = self.TITLE, self.BODY

        class FakeClient:
            def __init__(self):
                self.details = []

            def post_json(self, _url, _payload):
                return {"total": 2, "jobPostings": [
                    {"externalPath": senior_path, "title": title, "locationsText": "United States"},
                    {"externalPath": other_path, "title": "Localization Engineer", "locationsText": "United States"},
                ]}

            def get_json(self, url):
                self.details.append(url)
                is_senior = url.endswith(senior_path)
                if is_senior and mode == "failed":
                    raise crawler.FetchError("Temporary detail failure")
                if is_senior and mode == "malformed":
                    return {"unexpected": "not a job detail"}
                description = ("About the role: Develop localization for military customers."
                               if is_senior and mode == "military" else body)
                return {"jobPostingInfo": {
                    "title": title if is_senior else "Localization Engineer",
                    "jobReqId": "R10001" if is_senior else "R10002",
                    "externalUrl": src["base_url"] + "/External" + (senior_path if is_senior else other_path),
                    "location": "United States", "jobDescription": description,
                    "timeType": "Full-time", "canApply": True, "posted": True,
                }}

        client = FakeClient()
        fetched = crawler._workday(src, client)
        conf = {"schema_version": 1, "sources": [src], "manual_jobs": [], "pinned_jobs": []}
        merged = crawler.merge_results({"jobs": [old], "sources": []}, [fetched], conf, self.NOW)
        return old, fetched, merged, client

    def test_autonomy_positioning_role_with_inertial_filtering_is_reconfirmed(self):
        old, fetched, merged, _client = self.check("civilian")
        current = next(job for job in merged["jobs"] if job["id"] == old["id"])
        self.assertEqual(current["category"], "robotics-localization")
        self.assertEqual(current["status"], "open")
        self.assertEqual(current["check_status"], "ok")
        self.assertEqual(current["first_seen"], self.EARLIER)
        self.assertEqual(current["eligibility"]["checked_at"], self.NOW)
        self.assertNotIn(old["id"], fetched.rejected_ids)
        for title, body in (
            ("Market Positioning Engineer", self.BODY),
            (self.TITLE, "Research market positioning and competitive strategy for construction software."),
        ):
            with self.subTest(title=title, body=body):
                self.assertIsNone(crawler.classify_job(title, body, "United States"))

    def test_search_only_or_failed_detail_preserves_approved_role_and_old_review_date(self):
        for mode in ("unchecked", "failed", "malformed"):
            with self.subTest(mode=mode):
                old, fetched, merged, _client = self.check(mode)
                self.assertEqual(fetched.status, "partial")
                self.assertIn(old["id"], fetched.seen_ids)
                self.assertNotIn(old["id"], fetched.rejected_ids)
                self.assertEqual(old["id"] in fetched.attempted_ids, mode != "unchecked")
                current = next(job for job in merged["jobs"] if job["id"] == old["id"])
                self.assertEqual(current["first_seen"], self.EARLIER)
                self.assertEqual(current["check_status"], "partial")
                self.assertEqual(current["eligibility"]["checked_at"], self.EARLIER)
                self.assertEqual(current["last_verified"], self.EARLIER)

    def test_successfully_read_military_assignment_is_removed_even_on_partial_refresh(self):
        old, fetched, merged, _client = self.check("military")
        self.assertEqual(fetched.status, "partial")
        self.assertIn(old["id"], fetched.rejected_ids)
        self.assertNotIn(old["id"], {job["id"] for job in merged["jobs"]})


if __name__=="__main__":
    unittest.main()

