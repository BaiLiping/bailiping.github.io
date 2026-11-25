"""Keep physical estimation/calibration discovery distinct from adjacent jobs."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "job_state_estimation_crawler", ROOT / "scripts" / "crawl-potential-jobs.py")
crawler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = crawler
SPEC.loader.exec_module(crawler)


class StateEstimationMatchingTests(unittest.TestCase):
    def test_localization_and_calibration_engineering_are_in_scope(self):
        examples = [
            ("State Estimation Engineer - Localization",
             "Responsibilities: Develop vehicle localization and sensor calibration "
             "for autonomous driving. Deploy real-time C++ and Python software."),
            ("State-Estimation Engineer",
             "Build robust navigation algorithms for autonomous vehicles."),
            ("Sensor Calibration Engineer",
             "Develop extrinsic camera and LiDAR calibration for automated driving."),
            ("Camera Calibration Research Engineer",
             "Estimate multi-camera geometry and validate sensor alignment for robots."),
            ("Software Engineer, Navigation",
             "Develop state estimation and geometric optimization for autonomous vehicles."),
            ("Research Scientist, Geometric Estimation",
             "Use probabilistic estimation and bundle adjustment to estimate robot pose."),
        ]
        for title, body in examples:
            with self.subTest(title=title):
                found = crawler.classify_job(title, body, "Sunnyvale, CA", "Full-time")
                self.assertIsNotNone(found)
                self.assertEqual(found["category"], "robotics-localization")
                self.assertEqual(found["employment_type"], "Full-time")
                self.assertEqual(found["region"], "US")

    def test_unrelated_estimation_and_calibration_are_not_promoted(self):
        examples = [
            ("State Estimation Engineer", "Estimate battery state of charge and state of "
             "health for electric vehicles using Kalman filters."),
            ("Software Engineer", "Develop probabilistic estimation of battery health "
             "for autonomous vehicle battery management systems."),
            ("Research Scientist", "Research probabilistic state estimation for power "
             "grid voltage and electrical network monitoring."),
            ("Sensor Calibration Engineer", "Develop sensor calibration algorithms for "
             "laboratory pressure instruments and traceable metrology."),
            ("Calibration Algorithm Engineer", "Implement calibration algorithms for "
             "factory temperature instruments and test fixtures."),
            ("Localization Engineer", "Build translation workflows and linguistic "
             "quality tools for software internationalization."),
            ("Software Engineer", "Develop probabilistic estimation for advertising "
             "conversion and customer demand forecasts."),
        ]
        for title, body in examples:
            with self.subTest(title=title, body=body):
                self.assertIsNone(crawler.classify_job(title, body, "Sunnyvale, CA", "Full-time"))

    def test_role_duties_take_precedence_over_robotics_company_intro(self):
        self.assertIsNone(crawler.classify_job(
            "Sensor Calibration Engineer",
            "We are an autonomous robotics company. Responsibilities: Develop sensor "
            "calibration for laboratory temperature instruments and manufacturing metrology.",
            "Boston, MA", "Full-time"))

    def test_real_motion_estimation_can_mention_battery_or_translation(self):
        for body in (
            "Develop vehicle localization and sensor calibration, including during "
            "battery replacement and sensor maintenance.",
            "Develop robot localization, rotation and translation estimation using cameras.",
        ):
            self.assertIsNotNone(crawler.classify_job(
                "State Estimation Engineer - Localization", body, "Sunnyvale, CA", "Full-time"))

    def test_new_matches_keep_existing_geographic_and_civilian_limits(self):
        self.assertIsNone(crawler.classify_job(
            "Sensor Calibration Engineer", "Calibrate cameras for military autonomous vehicles.",
            "Sunnyvale, CA", "Full-time"))
        self.assertIsNone(crawler.classify_job(
            "Sensor Calibration Engineer", "Calibrate cameras for civilian autonomous vehicles.",
            "Bengaluru, India", "Full-time"))


class StateEstimationDiscoveryTests(unittest.TestCase):
    def test_hyphenated_workday_requisition_ignores_url_copy_suffix(self):
        for path, expected in (
            ("/job/US-CA/State-Estimation-Engineer_JR-202620439-1", "JR-202620439"),
            ("/job/US-CA/Localization_JR2028027", "JR2028027"),
            ("/job/US-CA/Calibration_R-10065402", "R-10065402"),
        ):
            with self.subTest(path=path):
                self.assertEqual(crawler.requisition_from_url(path), expected)

    def test_workday_details_prioritize_estimation_and_calibration_titles(self):
        titles = {
            "/job/US-CA/Aaa-General-Software_JR10001": "Aaa General Software Engineer",
            "/job/US-CA/Sensor-Calibration_JR10002": "Sensor Calibration Engineer",
            "/job/US-CA/State-Estimation_JR10003": "State Estimation Engineer",
        }
        requested = []

        class Client:
            def post_json(self, url, payload):
                return {"total": len(titles), "jobPostings": [
                    {"title": title, "externalPath": path, "locationsText": "Sunnyvale, CA"}
                    for path, title in titles.items()]}

            def get_json(self, url):
                path = url[url.index("/job/"):]
                requested.append(path)
                return {"jobPostingInfo": {
                    "title": titles[path], "jobReqId": crawler.requisition_from_url(path),
                    "location": "Sunnyvale, CA", "canApply": True, "timeType": "Full-time",
                    "jobDescription": "Develop vehicle localization and sensor calibration for autonomous driving.",
                }}

        settings = {
            "id": "example", "name": "Example careers", "company": "Example",
            "type": "workday", "base_url": "https://example.wd5.myworkdayjobs.com",
            "allowed_hosts": ["example.wd5.myworkdayjobs.com"], "tenant": "example",
            "site": "External", "search_url": "https://example.wd5.myworkdayjobs.com/wday/cxs/example/External/jobs",
            "queries": ["state estimation"], "page_size": 20, "max_pages_per_query": 1,
            "max_details": 2,
        }
        with patch.object(crawler, "_rotation_week", return_value=0):
            found = crawler.fetch_source(settings, Client())
        self.assertEqual(set(requested), set(list(titles)[1:]))
        self.assertEqual(len(found.records), 2)

    def test_expanded_global_budget_remains_bounded(self):
        budget = crawler.RunBudget(requests=10000)
        self.assertEqual(budget.remaining, 240)
        for _ in range(240):
            budget.consume()
        with self.assertRaises(crawler.FetchError):
            budget.consume()


if __name__ == "__main__":
    unittest.main()
