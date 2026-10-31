"""Regression checks for untrusted job content and availability presentation."""
from copy import deepcopy
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
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

    def test_sweden_uses_only_explicit_location_evidence(self):
        for location in ("Stockholm, Sweden", "Remote / Sverige", "Göteborg", "Linköping, SE",
                         "Lund", "Västerås", "Malmö"):
            with self.subTest(location=location):
                self.assertTrue(page.is_sweden_location(location))
        for location in ("Zurich, Switzerland", "Boston, MA, United States", "Remote",
                         "Remote, Europe", "Stockholm, Wisconsin, US", "Lund, BC, Canada",
                         "Gothenburg, NE", "Stockholmer Straße, Berlin, Germany"):
            with self.subTest(location=location):
                self.assertFalse(page.is_sweden_location(location))
        job = snapshot()["jobs"][0]
        job["summary"] = "The team collaborates with colleagues in Sweden."
        self.assertIn('data-sweden="false"', page.render_card(job))
        job["location"] = 'Stockholm, Sweden" onmouseover="alert(1)'
        card = page.render_card(job)
        attributes = []
        parser = HTMLParser()
        parser.handle_starttag = lambda tag, attrs: attributes.append(dict(attrs)) if tag == "article" else None
        parser.feed(card)
        self.assertEqual(attributes[0]["data-sweden"], "true")
        self.assertNotIn("onmouseover", attributes[0])
        self.assertIn('Sweden&quot; onmouseover=&quot;alert(1)', card)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to exercise the browser filter script")
    def test_sweden_and_europe_filters_include_the_right_rendered_cards(self):
        examples = [
            ("sweden", "Stockholm, Sweden", "Europe"),
            ("switzerland", "Zurich, Switzerland", "Europe"),
            ("us", "Boston, MA, United States", "US"),
            ("remote-sweden", "Remote / Sweden", "Remote"),
            ("remote-unspecified", "Remote", "Remote"),
            ("swedish-city", "Göteborg", "Unknown"),
        ]
        cards = []
        for identity, location, region in examples:
            job = snapshot()["jobs"][0]
            job.update(id=identity, location=location, region=region)
            parser = HTMLParser()
            parser.handle_starttag = lambda tag, attrs: cards.append(
                {key.removeprefix("data-"): value for key, value in attrs if key.startswith("data-")}
            ) if tag == "article" else None
            parser.feed(page.render_card(job))
        # Exercise the actual progressive-enhancement script with its small DOM
        # interface, using data attributes produced by the renderer above.
        harness = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const cards = JSON.parse(fs.readFileSync(0, 'utf8')).map(dataset => ({dataset, hidden: false}));
const handlers = {};
const nodes = {
  'job-filters': {addEventListener: (name, callback) => {handlers[name] = callback;}},
  'job-search': {value: ''}, 'job-category': {value: 'all'},
  'job-type': {value: 'all'}, 'job-region': {value: 'all'},
  'job-status': {value: 'current'}, 'results-count': {}, 'no-results': {},
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  document: {
    getElementById: id => nodes[id],
    querySelectorAll: () => cards,
    querySelector: () => null,
  },
  requestAnimationFrame: callback => callback(),
});
const results = {};
for (const region of ['Sweden', 'Europe', 'US', 'Remote', 'all']) {
  nodes['job-region'].value = region;
  handlers.change();
  results[region] = cards.filter(card => !card.hidden).map(card => card.dataset.job);
}
process.stdout.write(JSON.stringify(results));
'''
        result = subprocess.run(
            [shutil.which("node"), "-e", harness, str(ROOT / "career/potential-jobs/jobs.js")],
            input=json.dumps(cards), text=True, capture_output=True, check=True,
        )
        visible = json.loads(result.stdout)
        self.assertEqual(visible["Sweden"], ["sweden", "remote-sweden", "swedish-city"])
        self.assertEqual(visible["Europe"], ["sweden", "switzerland", "remote-sweden", "swedish-city"])
        self.assertEqual(visible["US"], ["us"])
        self.assertEqual(visible["Remote"], ["remote-sweden", "remote-unspecified"])
        self.assertEqual(len(visible["all"]), len(examples))
        self.assertIn('<option value="Sweden">Sweden</option>', self.template)
        self.assertIn('<option value="Europe">Europe</option>', self.template)

    def test_failed_check_does_not_display_confirmed_open(self):
        data = snapshot()
        data["jobs"][0]["check_status"] = "error"
        card = page.render_card(data["jobs"][0])
        self.assertIn('data-status="unknown"', card)
        self.assertIn('Needs recheck', card)
        self.assertIn('Latest check incomplete.', card)
        self.assertNotIn('Open at last check', card)
        self.assertIn('0 confirmed open', page.render_snapshot(data))

    def test_public_wording_preserves_source_titles_links_and_snapshot(self):
        data = snapshot()
        data["jobs"][0].update(
            title="Senior RADAR Perception Engineer",
            url="https://example.com/jobs/Senior-Radar-Perception-Engineer",
            category="radar-sensing",
            summary="Role text includes radar sensing, signal processing, or radar-based perception.",
            match_reasons=["Radar is named in the role's title"],
            requirements=["Radar mentioned", "Python mentioned"],
        )
        original = deepcopy(data)
        result = page.render_page(data, self.template)
        text = []
        parser = HTMLParser()
        parser.handle_data = text.append
        parser.feed(result)
        self.assertNotRegex(" ".join(text), r"(?i)\bradar\b")
        self.assertIn("Senior Perception Engineer", result)
        self.assertIn("Role text includes sensing, signal processing, or perception.", result)
        self.assertIn("Sensing mentioned", result)
        self.assertIn('value="radar-sensing">Sensing</option>', result)
        self.assertIn('href="https://example.com/jobs/Senior-Radar-Perception-Engineer"', result)
        self.assertEqual(data, original)

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
