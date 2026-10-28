"""Offline checks for the Swedish job-watch additions and their data boundaries."""

import copy
import json
import unittest

from test_job_crawler import EARLIER, NOW, config, crawler, job, source, state


class SwedishMatchingTests(unittest.TestCase):
    def test_country_filter_uses_explicit_location_not_description_or_remote_status(self):
        settings = dict(source("sweden"), country_filter="SE")
        for location, accepted in (
            ("Gothenburg, Sweden", True),
            ("Göteborg, Sverige", True),
            ("Remote / Sweden", True),
            ("Boston, MA, United States", False),
            ("Zurich, Switzerland", False),
            ("Remote", False),
            ("Stockholm", False),
            ("", False),
        ):
            with self.subTest(location=location):
                matched = crawler.make_job(
                    settings, "101", "SLAM Research Engineer",
                    "https://jobs.lever.co/sweden/101", location,
                    "Develop robot localization and sensor fusion. "
                    "Our company also has an office in Sweden.",
                )
                self.assertEqual(matched is not None, accepted)

    def test_provider_iso_country_codes_become_explicit_sweden_locations(self):
        settings = dict(source("sweden"), country_filter="SE")
        for country in ("SE", "SWE", {"name": "SE"}):
            with self.subTest(country=country):
                location = crawler._address_text({"address": {
                    "addressLocality": "Västerås", "addressCountry": country,
                }})
                self.assertEqual(location, "Västerås, Sweden")
                self.assertIsNotNone(crawler.make_job(
                    settings, "101", "SLAM Research Engineer",
                    "https://jobs.lever.co/sweden/101", location,
                    "Develop robot localization and sensor fusion.",
                ))

    def test_jobylon_country_in_postal_address_is_used_without_publishing_street(self):
        location = crawler._address_text({"address": {
            "addressLocality": "Gothenburg", "streetAddress": "Example Street 12, Gothenburg, Sweden",
        }})
        self.assertEqual(location, "Gothenburg, Sweden")
        self.assertNotIn("Example Street", location)
        self.assertEqual(crawler._address_text({"address": {
            "addressLocality": "Gothenburg, Sweden", "addressCountry": "SE",
        }}), "Gothenburg, Sweden")
        self.assertEqual(crawler._address_text({"address": {
            "addressLocality": "Lund", "streetAddress": "12 Sweden Street",
        }}), "Lund")
        self.assertEqual(crawler._address_text({"address": {
            "addressLocality": "Lund", "streetAddress": "Sweden Street 12", "addressCountry": "Canada",
        }}), "Lund, Canada")

    def test_new_technical_titles_need_estimation_evidence(self):
        for title, description in (
            ("Experienced Computer Vision Engineer",
             "Develop visual odometry and camera/LiDAR localization for vehicles."),
            ("Senior Calibration Engineer",
             "Develop sensor fusion and pose estimation for robot calibration."),
            ("Machine Learning Engineer — Autonomy with dual-use application",
             "Develop multimodal sensor fusion and tracking for autonomous vehicles."),
        ):
            with self.subTest(title=title):
                matched = crawler.classify_job(title, description, "Gothenburg, Sweden")
                self.assertIsNotNone(matched)
                self.assertEqual(matched["category"], "robotics-localization")
        self.assertIsNone(crawler.classify_job(
            "Experienced Computer Vision Engineer",
            "Classify shopping catalogue images and improve product thumbnails.",
            "Gothenburg, Sweden",
        ))

    def test_vehicle_calibration_with_explicit_radar_is_a_sensing_match(self):
        matched = crawler.classify_job(
            "Senior Calibration Engineer",
            "Develop intrinsic and extrinsic calibration for vehicle cameras, radar, IMU and GNSS.",
            "Gothenburg, Sweden",
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["category"], "radar-sensing")

    def test_swedish_research_and_clearance_words_remain_neutral_evidence(self):
        matched = crawler.classify_job(
            "Doktorerad analytiker inom sensorfusion",
            "Utveckla målföljning och sensorfusion med Kalmanfilter för autonoma "
            "system. Tjänsten kräver säkerhetsprövning och kan beröra medborgarskap.",
            "Linköping, Sverige", "Heltid",
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched["category"], "robotics-localization")
        self.assertEqual(matched["employment_type"], "Full-time")
        self.assertIn("PhD/graduate study mentioned", matched["requirements"])
        self.assertIn("Citizenship/clearance conditions mentioned", matched["requirements"])
        self.assertNotIn("Current US work authorization required", matched["requirements"])
        self.assertNotIn("Sponsorship restriction mentioned", matched["requirements"])

    def test_student_thesis_exclusion_does_not_remove_phd_internships(self):
        unrestricted = dict(source("sweden"), country_filter="SE")
        filtered = dict(unrestricted, exclude_student_theses=True)
        for title in (
            "Master's Thesis Research Intern, Radio Localization",
            "MSc Thesis Research Intern, Radio Localization",
            "Bachelor Thesis Research Intern, Radio Localization",
            "BSc Thesis Research Intern, Radio Localization",
            "Examensarbete: Research Intern, Radio Localization",
        ):
            with self.subTest(title=title):
                args = ("101", title, "https://jobs.lever.co/sweden/101",
                        "Lund, Sweden", "Research radio SLAM and Bayesian estimation.")
                self.assertIsNotNone(crawler.make_job(unrestricted, *args),
                                     "The fixture must exercise the thesis filter itself")
                self.assertIsNone(crawler.make_job(filtered, *args))
        retained = crawler.make_job(
            filtered, "102", "PhD Research Intern, Radio Localization",
            "https://jobs.lever.co/sweden/102", "Lund, Sweden",
            "Doctoral research on radio SLAM contributing to a PhD thesis.",
        )
        self.assertIsNotNone(retained)
        self.assertEqual(retained["employment_type"], "Internship")


class SwedishSourceTests(unittest.TestCase):
    @staticmethod
    def jobylon_source():
        return {
            "id": "einride", "name": "Example Swedish robotics", "company": "Einride",
            "type": "html", "allowed_hosts": ["cdn.jobylon.com", "emp.jobylon.com"],
            "index_url": "https://cdn.jobylon.com/jobs/companies/3143/embed/v2/",
            "html_discovery": "jobylon", "country_filter": "SE",
            "job_url_patterns": [r"^https://emp\.jobylon\.com/jobs/[0-9]+-[a-z0-9-]+/$"],
            "max_details": 2, "max_requests": 3, "complete_listing": False,
        }

    def test_jobylon_requires_matching_literal_id_and_url_without_executing_scripts(self):
        listing = """
        <script>
        id: '101',
        url: '/jobs/101-example-slam-engineer/',
        id: '102',
        url: '/jobs/103-mismatched-id/',
        id: '104',
        url: 'https://foreign.example/jobs/104-injected/',
        throw new Error('Never execute provider scripts');
        </script>
        """
        self.assertEqual(crawler._jobylon_embed_urls(listing),
                         ["https://emp.jobylon.com/jobs/101-example-slam-engineer/"])

        detail = {"@type": "JobPosting", "identifier": {"value": "101"},
                  "title": "SLAM Research Engineer", "employmentType": "FULL_TIME",
                  "description": "Develop robot localization and sensor fusion.",
                  "jobLocation": {"address": {"addressLocality": "Mölndal",
                                               "addressCountry": "SE"}}}

        class LiteralEmbedClient:
            def get_text(self, url, **kwargs):
                if "cdn.jobylon.com" in url:
                    return listing
                if url != "https://emp.jobylon.com/jobs/101-example-slam-engineer/":
                    raise AssertionError("Invalid discovery URL was requested")
                return '<script type="application/ld+json">' + json.dumps(detail) + '</script>'

        result = crawler.fetch_source(self.jobylon_source(), LiteralEmbedClient())
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.complete)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["location"], "Mölndal, Sweden")
        self.assertEqual(result.records[0]["region"], "Europe")

    def test_empty_jobylon_discovery_preserves_previous_job(self):
        settings = self.jobylon_source()
        old = job("101", "einride",
                  url="https://emp.jobylon.com/jobs/101-example-slam-engineer/",
                  location="Gothenburg, Sweden", region="Europe",
                  status="listed", missing_count=1)

        class ChangedEmbedClient:
            def get_text(self, url, **kwargs):
                if url != settings["index_url"]:
                    raise AssertionError("Invalid index must fail before detail requests")
                return "<p>The job widget format has changed.</p>"

        result = crawler.fetch_source(dict(settings, _known_jobs=[old]), ChangedEmbedClient())
        self.assertEqual(result.status, "error")
        self.assertFalse(result.complete)
        merged = crawler.merge_results(state(old), [result], config(settings), NOW)
        self.assertEqual(len(merged["jobs"]), 1)
        retained = merged["jobs"][0]
        self.assertEqual(retained["id"], old["id"])
        self.assertEqual(retained["status"], "listed")
        self.assertEqual(retained["missing_count"], 1)
        self.assertEqual(retained["last_verified"], EARLIER)
        self.assertEqual(retained["check_status"], "error")

    def test_workday_country_facets_are_validated_and_passed_to_each_query(self):
        facets = {"locationCountry": ["verified-sweden-id"]}
        settings = {
            "id": "sweden", "name": "Swedish careers", "company": "Example",
            "type": "workday", "allowed_hosts": ["example.wd5.myworkdayjobs.com"],
            "base_url": "https://example.wd5.myworkdayjobs.com",
            "tenant": "example", "site": "External",
            "search_url": "https://example.wd5.myworkdayjobs.com/wday/cxs/example/External/jobs",
            "queries": ["SLAM", "sensor fusion"], "page_size": 1,
            "max_pages_per_query": 1, "max_details": 1, "max_requests": 3,
            "country_filter": "SE", "applied_facets": facets,
        }
        crawler.validate_config(config(settings))

        class FacetedClient:
            def __init__(self):
                self.payloads = []

            def post_json(self, url, payload, **kwargs):
                self.payloads.append(copy.deepcopy(payload))
                return {"total": 1, "jobPostings": [{
                    "title": "SLAM Research Engineer", "locationsText": "Sweden",
                    "externalPath": "/job/Sweden/SLAM-Research_R1234",
                }]}

            def get_json(self, url, **kwargs):
                return {"jobPostingInfo": {
                    "title": "SLAM Research Engineer", "jobReqId": "R1234",
                    "location": "Mölndal", "country": {"descriptor": "Sweden"},
                    "jobDescription": "Develop robot localization and sensor fusion.",
                    "posted": True, "canApply": True,
                }}

        client = FacetedClient()
        result = crawler.fetch_source(settings, client)
        self.assertEqual(len(client.payloads), 2)
        self.assertTrue(all(payload["appliedFacets"] == facets for payload in client.payloads))
        self.assertEqual({payload["searchText"] for payload in client.payloads}, set(settings["queries"]))
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["location"], "Mölndal / Sweden")
        with self.assertRaises(ValueError):
            crawler.validate_config(config(dict(settings, applied_facets={"locationCountry": "SE"})))


if __name__ == "__main__":
    unittest.main()
