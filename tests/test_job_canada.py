"""Canadian geography, language requirements and ISAC coverage regressions."""
import copy
import unittest

from test_job_crawler import NOW, config, crawler, source, state
from test_job_page import page, snapshot


class CanadaCoverageTests(unittest.TestCase):
    def make_job(self, location="Toronto, ON, Canada", description="Develop robot localization and sensor fusion.", **changes):
        settings = dict(source("canada"), country_filter="CA", _now=NOW)
        settings.update(changes)
        return crawler.make_job(settings, "101", "SLAM Research Engineer",
                                "https://jobs.lever.co/canada/101", location, description)

    def test_canada_does_not_confuse_california_or_us_city_names(self):
        for location, expected in (
            ("Toronto, ON", "Canada"), ("Calgary, Alberta, Canada", "Canada"),
            ("Ottawa, Canada", "Canada"), ("Vancouver, British Columbia", "Canada"),
            ("Vancouver, Washington, USA", "US"), ("San Francisco, CA", "US"),
            ("Remote / Canada", "Remote"), ("Remote", "Remote"),
        ):
            with self.subTest(location=location):
                self.assertEqual(crawler.region_for(location), expected)
        self.assertIsNone(self.make_job("San Francisco, CA"))
        self.assertIsNone(self.make_job("Remote", "Robot localization for offices in Canada."))
        self.assertEqual(crawler._address_text({"addressLocality": "Calgary", "addressCountry": "CA"}), "Calgary, Canada")

    def test_remote_canada_matches_country_and_multiple_locations_keep_us(self):
        job = self.make_job("Remote / Canada")
        self.assertEqual(job["regions"], ["Remote", "Canada"])
        job = self.make_job("Toronto, Ontario, Canada / San Francisco, CA, United States")
        self.assertEqual(set(job["regions"]), {"Canada", "US"})
        self.assertEqual(job["language_review"], {"status": "no_french_stated", "checked_at": NOW, "method": "posting_text"})

    def test_quebec_and_required_french_are_excluded(self):
        for location in ("Montréal, Canada", "Montreal, QC", "Québec, Canada", "Laval, Quebec"):
            self.assertIsNone(self.make_job(location))
        for requirement in ("French is required.", "Must be fluent in French.",
                            "Bilingual (English/French).", "Proficiency in English and French is essential.",
                            "French: fluent.", "Language requirements: English and French.",
                            "Required: French.", "French is required, English is preferred.",
                            "French fluency required, German is optional.",
                            "French is required and English is preferred."):
            with self.subTest(requirement=requirement):
                self.assertIsNone(self.make_job(description="Robot localization. " + requirement))
        self.assertIsNone(crawler.classify_job("Bilingual SLAM Research Engineer", "Robot localization.", "Toronto, Canada"))

    def test_optional_or_not_required_french_does_not_exclude(self):
        for requirement in ("French is not required.", "French is an asset.",
                            "French proficiency is preferred.", "Bilingualism is optional.",
                            "No French required.", "Optional: French."):
            with self.subTest(requirement=requirement):
                self.assertIsNotNone(self.make_job(description="Robot localization. " + requirement))
        job = self.make_job(description="Robot localization. Fluency in English is required. French is optional.")
        self.assertEqual(job["language_review"]["status"], "english_required")

    def test_full_lever_additional_text_is_screened(self):
        settings = dict(source("canada"), country_filter="CA", _now=NOW)
        class Client:
            def get_json(self, url):
                return [{"id": "101", "text": "SLAM Research Engineer", "hostedUrl": "https://jobs.lever.co/canada/101",
                         "categories": {"location": "Toronto, Canada"},
                         "descriptionPlain": "Develop robot localization and sensor fusion.",
                         "additionalPlain": "Bilingual English and French required."}]
        result = crawler._lever(settings, Client())
        self.assertEqual(result.records, [])
        self.assertEqual(result.rejected_ids, ["canada:101"])

    def test_lever_multilocation_hybrid_is_not_fully_remote(self):
        settings = dict(source("canada"), country_filter="CA", _now=NOW)
        class Client:
            def __init__(self, workplace):
                self.workplace = workplace
            def get_json(self, url):
                return [{"id": "101", "text": "SLAM Research Engineer", "hostedUrl": "https://jobs.lever.co/canada/101",
                         "categories": {"location": "Toronto, ON", "allLocations": [
                             "Toronto, ON", "San Francisco, CA", "Remote, Canada"]},
                         "workplaceType": self.workplace,
                         "descriptionPlain": "Develop robot localization and sensor fusion."}]
        for workplace in ("hybrid", "onsite", "on-site"):
            with self.subTest(workplace=workplace):
                result = crawler._lever(settings, Client(workplace))
                self.assertEqual(result.status, "ok")
                job, = result.records
                self.assertEqual(job["region"], "Canada")
                self.assertEqual(job["regions"], ["Canada", "US"])
                public = crawler._public_job(job, settings)
                self.assertEqual(public["regions"], ["Canada", "US"])
        remote, = crawler._lever(settings, Client("remote")).records
        self.assertEqual(remote["region"], "Remote")
        self.assertEqual(remote["regions"], ["Canada", "US", "Remote"])

    def test_new_rejection_removes_previously_accepted_canadian_job(self):
        settings = dict(source("canada"), country_filter="CA", _now=NOW)
        previous = state(self.make_job())
        result = crawler.SourceResult("canada", "ok", [], complete=True, rejected_ids=["canada:101"])
        merged = crawler.merge_results(previous, [result], config(settings), NOW)
        self.assertEqual(merged["jobs"], [])

    def test_isac_variants_and_technical_test_manager(self):
        for title, description in (
            ("Research Engineer", "Integrated sensing and communication for 6G."),
            ("Research Engineer", "Joint communications and sensing for wireless networks."),
            ("Test Manager ISAC System Verification", "Develop ISAC verification for 6G networks."),
        ):
            match = crawler.classify_job(title, description, "Ottawa, Canada")
            self.assertEqual(match["category"], "wireless-isac")
        self.assertIsNone(crawler.classify_job("Test Manager", "Ordinary software verification.", "Ottawa, Canada"))

    def test_autonomous_vehicle_research_internship_lists_mapping_localization(self):
        match = crawler.classify_job(
            "2027 Intern, PhD Research Scientist",
            "You will... Research algorithms for autonomous vehicles. "
            "Topics include perception, prediction, mapping, localization, and controls. "
            "Pursuing a PhD in robotics; Python and C++.",
            "Toronto, ON", "Intern")
        self.assertEqual(match["category"], "robotics-localization")
        self.assertEqual(match["employment_type"], "Internship")
        unrelated = crawler.classify_job(
            "Research Scientist", "Our company builds autonomous vehicles. You will... "
            "Work on website content mapping and localization for international languages.", "Toronto, ON")
        self.assertIsNone(unrelated)

    def test_language_date_source_and_canada_filter_render_without_promising_language(self):
        data = snapshot()
        data["jobs"][0].update(self.make_job())
        html = page.render_page(data, (page.PAGE / "template.html").read_text())
        self.assertIn('value="Canada"', html)
        self.assertIn('data-regions="Canada"', html)
        self.assertIn("No French requirement stated in the reviewed posting", html)
        self.assertIn("Posting-text check:", html)
        self.assertIn("Source posting</a>", html)
        self.assertNotIn("French is not required", html)
        bad = copy.deepcopy(data)
        bad["jobs"][0]["language_review"]["status"] = "guaranteed_english"
        with self.assertRaises(ValueError):
            page.validate_snapshot(bad)


if __name__ == "__main__":
    unittest.main()
