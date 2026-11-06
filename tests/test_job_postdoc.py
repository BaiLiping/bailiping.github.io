"""Protect postdoctoral appointment labels without broadening technical scope."""

import unittest

from test_job_crawler import crawler, source
from test_job_page import page, snapshot


class PostdocMatchingTests(unittest.TestCase):
    def test_explicit_postdoc_titles_override_full_time_and_intern_metadata(self):
        for title in (
            "Postdoc in robot perception",
            "Postdoctoral Fellow in autonomous systems",
            "Post-doctoral Fellow in sensing",
            "Postdoktor i signalbehandling",
        ):
            for employment in ("Full-time", "Internship", ""):
                with self.subTest(title=title, employment=employment):
                    matched = crawler.classify_job(
                        title, "Develop robot localization, sensor fusion and visual odometry. "
                        "Work with research interns and students.", "Lund, Sweden", employment,
                    )
                    self.assertIsNotNone(matched)
                    self.assertEqual(matched["employment_type"], "Postdoc")
                    self.assertEqual(matched["category"], "robotics-localization")

    def test_research_fellow_needs_explicit_postdoctoral_appointment_evidence(self):
        description = "Develop robot localization and sensor fusion. "
        for detail in (
            "A PhD is required.",
            "Postdoctoral experience is preferred.",
            "Work with postdocs and supervise interns.",
            "This position is not a postdoctoral appointment.",
        ):
            with self.subTest(detail=detail):
                matched = crawler.classify_job(
                    "Research Fellow", description + detail, "Oxford, United Kingdom", "Full-time",
                )
                self.assertEqual(matched["employment_type"], "Full-time")
        for detail in (
            "This position is a postdoctoral appointment.",
            "The fellowship will be a two-year post-doctoral appointment.",
            "Tjänsten är som postdoktor.",
        ):
            with self.subTest(detail=detail):
                matched = crawler.classify_job(
                    "Research Fellow", description + detail, "Lund, Sweden", "Full-time",
                )
                self.assertEqual(matched["employment_type"], "Postdoc")

    def test_explicit_provider_type_is_postdoc(self):
        matched = crawler.classify_job(
            "Research Fellow", "Develop robot localization and sensor fusion.",
            "Oxford, United Kingdom", "Postdoctoral",
        )
        self.assertEqual(matched["employment_type"], "Postdoc")

    def test_postdoc_title_does_not_admit_unrelated_research(self):
        for title, description in (
            ("Postdoctoral Fellow in molecular biology", "Research cell cultures and protein expression."),
            ("Postdoc in battery state estimation", "Estimate battery state of charge and health."),
            ("Postdoc in language localization", "Research linguistic translation of content."),
        ):
            with self.subTest(title=title):
                self.assertIsNone(crawler.classify_job(title, description, "Lund, Sweden", "Full-time"))

    def test_postdoc_and_student_thesis_remain_distinct(self):
        settings = dict(source("sweden"), country_filter="SE", exclude_student_theses=True)
        args = ("102", "Postdoctoral Fellow in SLAM", "https://jobs.lever.co/sweden/102",
                "Lund, Sweden", "Research robot localization; supervise master's thesis students.", "Full-time")
        self.assertEqual(crawler.make_job(settings, *args)["employment_type"], "Postdoc")
        thesis = ("103", "Master's Thesis Research Intern, Radio Localization",
                  "https://jobs.lever.co/sweden/103", "Lund, Sweden",
                  "Research radio SLAM with postdoctoral fellows.", "Internship")
        self.assertIsNone(crawler.make_job(settings, *thesis))

    def test_public_records_retain_postdoc_and_upgrade_older_full_time_labels(self):
        settings = source()
        original = crawler.make_job(
            settings, "1", "Postdoc in SLAM", "https://jobs.lever.co/test/1",
            "Lund, Sweden", "Develop robot localization and sensor fusion.", "Full-time",
        )
        for kind in ("Postdoc", "Full-time", "Unknown"):
            with self.subTest(kind=kind):
                retained = crawler._public_job(dict(original, employment_type=kind), settings)
                self.assertEqual(retained["employment_type"], "Postdoc")


class PostdocRenderingTests(unittest.TestCase):
    def test_postdoc_snapshot_renders_with_filterable_type(self):
        data = snapshot()
        data["jobs"][0].update(title="Postdoctoral Fellow in SLAM", employment_type="Postdoc")
        page.validate_snapshot(data)
        card = page.render_card(data["jobs"][0])
        self.assertIn('data-type="Postdoc"', card)
        self.assertIn("Postdoc", card)


if __name__ == "__main__":
    unittest.main()
