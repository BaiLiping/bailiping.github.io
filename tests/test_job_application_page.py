"""Public application labels must remain evidence-based and filter correctly."""
from copy import deepcopy
from html.parser import HTMLParser
import json
import shutil
import subprocess
import unittest

from test_job_page import ROOT, page, snapshot


def reviewed(sponsorship="not_stated", access="no_restriction_stated",
             work_authorization="not_stated"):
    return {
        "sponsorship": sponsorship, "access": access,
        "work_authorization": work_authorization,
        "evidence": [], "checked_at": "2026-11-05T15:30:00Z",
        "method": "manual_review",
    }


class ApplicationPageTests(unittest.TestCase):
    def test_missing_review_is_unknown_not_a_positive_target(self):
        job = snapshot()["jobs"][0]
        card = page.render_card(job)
        self.assertIn('data-application="unknown"', card)
        self.assertIn('data-sponsorship="unknown"', card)
        self.assertNotIn("Potential target", card)
        self.assertIn("not yet been verified", card)

    def test_access_and_current_authorization_override_sponsorship_offer(self):
        for key, value, expected in (
            ("access", "us_person", "restricted"),
            ("access", "citizenship_clearance", "restricted"),
            ("access", "export_review", "export_review"),
            ("work_authorization", "required_now", "work_authorization"),
        ):
            with self.subTest(key=key, value=value):
                job = snapshot()["jobs"][0]
                job["eligibility"] = reviewed("offered")
                job["eligibility"][key] = value
                card = page.render_card(job)
                self.assertIn(f'data-application="{expected}"', card)
                self.assertIn("Explicitly offered for this role", card)
                self.assertNotIn("Potential target", card)

    def test_conflicting_reviews_are_not_shown_as_unqualified_targets(self):
        job = snapshot()["jobs"][0]
        job["eligibility"] = reviewed("offered")
        job["eligibility_conflict"] = "A newer posting differs from a previous manual review."
        card = page.render_card(job)
        self.assertIn('data-application="unknown"', card)
        self.assertIn("Requirements differ between checks", card)
        self.assertNotIn("Potential target", card)

    def test_evidence_notes_are_escaped_and_preserve_their_review_date(self):
        job = snapshot()["jobs"][0]
        job["eligibility"] = reviewed("not_offered")
        job["eligibility"]["evidence"] = [
            {"topic": "sponsorship", "text": 'No sponsorship. <img src=x onerror="alert(1)">'}
        ]
        job["fit_note"] = "<script>unsafe</script>"
        original = deepcopy(job)
        card = page.render_card(job)
        self.assertNotIn("<img src=", card)
        self.assertNotIn("<script>unsafe", card)
        self.assertIn("&lt;img", card)
        self.assertIn("05 Nov 2026, 16:30 CET", card)
        self.assertEqual(job, original)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for actual filter-script checks")
    def test_application_and_sponsorship_filters_intersect_with_location(self):
        examples = [
            ("offered", "US", reviewed("offered")),
            ("export", "US", reviewed("offered", "export_review")),
            ("current-auth", "US", reviewed(work_authorization="required_now")),
            ("no-sponsor", "US", reviewed("not_offered")),
            ("silent", "US", reviewed()),
            ("foreign", "Europe", reviewed("offered")),
            ("unknown", "US", None),
        ]
        datasets = []
        for identity, region, eligibility in examples:
            job = snapshot()["jobs"][0]
            job.update(id=identity, region=region)
            if eligibility:
                job["eligibility"] = eligibility
            parser = HTMLParser()
            parser.handle_starttag = lambda tag, attrs: datasets.append({
                k[5:].replace("-", "_"): v for k, v in attrs if k.startswith("data-")
            }) if tag == "article" else None
            parser.feed(page.render_card(job))
        harness = r"""
const fs = require('node:fs'), vm = require('node:vm');
const cards = JSON.parse(fs.readFileSync(0, 'utf8')).map(dataset => ({dataset}));
const handlers = {}, nodes = {
  'job-filters': {addEventListener: (name, fn) => {handlers[name] = fn;}},
  'job-search': {value: ''}, 'job-category': {value: 'all'},
  'job-type': {value: 'all'}, 'job-region': {value: 'US'},
  'job-status': {value: 'current'}, 'job-sponsorship': {value: 'all'},
  'job-access': {value: 'all'}, 'results-count': {}, 'no-results': {},
  'job-list': {appendChild: () => {}}
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  document: {getElementById: id => nodes[id], querySelectorAll: () => cards, querySelector: () => null},
  requestAnimationFrame: fn => fn()
});
const visible = () => cards.filter(c => !c.hidden).map(c => c.dataset.job);
const results = {};
nodes['job-sponsorship'].value = 'offered'; handlers.change(); results.offered = visible();
nodes['job-access'].value = 'potential'; handlers.change(); results.potentialOffered = visible();
nodes['job-sponsorship'].value = 'all'; handlers.change(); results.potential = visible();
nodes['job-access'].value = 'work_authorization'; handlers.change(); results.work = visible();
nodes['job-access'].value = 'unknown'; handlers.change(); results.unknown = visible();
process.stdout.write(JSON.stringify(results));
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", harness, str(ROOT / "career/potential-jobs/jobs.js")],
            input=json.dumps(datasets), text=True, capture_output=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout), {
            "offered": ["offered", "export"], "potentialOffered": ["offered"],
            "potential": ["offered", "silent"],
            "work": ["current-auth", "no-sponsor"], "unknown": ["unknown"],
        })


if __name__ == "__main__":
    unittest.main()
