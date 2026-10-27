"""Regression checks for untrusted job content and availability presentation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("job_page", ROOT / "scripts/render-potential-jobs.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def snapshot():
    return {
        "schema_version": 1,
        "generated_at": "2026-10-27T12:00:00Z",
        "sources": [{"id": "test", "name": "Employer", "status": "ok", "checked_at": "2026-10-27T12:00:00Z"}],
        "jobs": [{
            "id": "test:1", "source_id": "test", "company": "Employer",
            "title": "Radio Localization Engineer", "url": "https://example.com/jobs/1",
            "location": "Boston, MA", "region": "US", "employment_type": "Full-time",
            "category": "radio-localization", "status": "open", "check_status": "ok",
            "summary": "Localization and estimation.", "match_reasons": ["UWB positioning"],
            "requirements": [], "last_seen": "2026-10-27T12:00:00Z",
            "last_verified": "2026-10-27T12:00:00Z", "pinned": False, "score": 50,
        }],
    }


class JobPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (ROOT / "career/potential-jobs/template.html").read_text()

    def test_job_html_is_escaped_in_visible_text_and_attributes(self):
        data = snapshot()
        data["jobs"][0]["title"] = '<img src=x onerror="alert(1)">'
        data["jobs"][0]["summary"] = "<script>alert('x')</script>"
        result = page.render_page(data, self.template)
        self.assertNotIn('<img src=x', result)
        self.assertNotIn("<script>alert", result)
        self.assertIn('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;', result)

    def test_failed_check_does_not_display_confirmed_open(self):
        data = snapshot()
        data["jobs"][0]["check_status"] = "error"
        card = page.render_card(data["jobs"][0])
        self.assertIn('data-status="unknown"', card)
        self.assertIn('Needs recheck', card)
        self.assertIn('Latest check incomplete.', card)
        self.assertNotIn('Open at last check', card)
        self.assertIn('0 confirmed open', page.render_snapshot(data))

    def test_closed_role_hidden_by_default_but_still_filterable(self):
        job = snapshot()["jobs"][0]
        job["status"] = "closed"
        card = page.render_card(job)
        self.assertIn('data-status="closed"', card)
        self.assertIn(' hidden>', card)
        self.assertIn('View posting', card)

    def test_saved_reference_is_not_counted_as_verified(self):
        data = snapshot()
        job = data["jobs"][0]
        job.update(status="saved", check_status="manual", pinned=True)
        result = page.render_page(data, self.template)
        self.assertIn('Saved link · unverified', result)
        self.assertIn('0 confirmed open', result)
        self.assertIn('availability is not checked automatically', result)

    def test_eligibility_and_seniority_flags_are_not_hidden_by_skill_list(self):
        job = snapshot()["jobs"][0]
        job["requirements"] = ["Python mentioned", "C++ mentioned", "ROS experience mentioned",
                               "Machine-learning frameworks mentioned", "Senior/experienced role",
                               "Sponsorship restriction mentioned", "Current US work authorization required"]
        card = page.render_card(job)
        visible_flags = card.split('<p class="job-requirements">', 1)[1].split('</p>', 1)[0]
        self.assertIn("Sponsorship restriction mentioned", visible_flags)
        self.assertIn("Senior/experienced role", visible_flags)
        self.assertIn("Current US work authorization required", visible_flags)

    def test_employer_watchlist_link_is_escaped_and_https_only(self):
        data = snapshot()
        data["sources"][0].update(careers_url="https://example.com/careers?x=1&y=2", count=0)
        result = page.render_sources(data)
        self.assertIn('href="https://example.com/careers?x=1&amp;y=2"', result)
        self.assertIn("0 matches in this check", result)
        data["sources"][0]["careers_url"] = "javascript:alert(1)"
        with self.assertRaises(ValueError):
            page.render_page(data, self.template)

    def test_unsafe_links_and_duplicate_ids_fail_before_rendering(self):
        for bad_url in ("javascript:alert(1)", "http://example.com/job", "https://user:pass@example.com/job", "https://example.com:444/job"):
            with self.subTest(url=bad_url):
                data = snapshot()
                data["jobs"][0]["url"] = bad_url
                with self.assertRaises(ValueError):
                    page.render_page(data, self.template)
        data = snapshot()
        data["jobs"].append(deepcopy(data["jobs"][0]))
        with self.assertRaises(ValueError):
            page.render_page(data, self.template)

    def test_generated_page_indexable_template_not_indexable(self):
        result = page.render_page(snapshot(), self.template)
        self.assertIn('name="robots" content="noindex"', self.template)
        self.assertNotIn('name="robots" content="noindex"', result)
        self.assertNotIn('<!-- JOBS -->', result)
        self.assertNotIn('<!-- SNAPSHOT -->', result)
        self.assertIn('data-search="', result)


if __name__ == "__main__":
    unittest.main()
