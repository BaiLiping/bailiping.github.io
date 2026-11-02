"""Regression coverage for civilian-only employer and role screening."""
from __future__ import annotations
import importlib.util
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
        result=crawler.SourceResult("fieldai","ok",[],complete=True,seen_ids=[old["id"]])
        merged=crawler.merge_results({"jobs":[old],"sources":[]},[result],conf,"2026-11-02T07:00:00Z")
        self.assertEqual(merged["jobs"],[])

    def test_defence_employer_cannot_be_readded(self):
        src=source()
        src["company"]="Saab"
        conf={"schema_version":1,"sources":[src],"manual_jobs":[],"pinned_jobs":[]}
        with self.assertRaisesRegex(ValueError,"Defence employer"):
            crawler.validate_config(conf)

if __name__=="__main__":
    unittest.main()
