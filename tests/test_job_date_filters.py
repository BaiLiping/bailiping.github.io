"""Date filtering/sorting checks using rendered cards and the real page script."""
from html.parser import HTMLParser
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("job_date_page", ROOT / "scripts/render-potential-jobs.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def job(identity, added=None, **changes):
    record = {
        "id": identity, "source_id": "test", "company": "Employer",
        "title": "Localization Engineer", "url": "https://example.com/jobs/" + identity,
        "location": "Sunnyvale, California, United States", "region": "US",
        "employment_type": "Full-time", "category": "robotics-localization",
        "status": "open", "check_status": "ok", "pinned": False, "score": 75,
        "summary": "Physical localization and estimation.", "match_reasons": [],
        "requirements": [], "last_seen": "2026-11-20T12:00:00Z",
        "last_verified": "2026-11-20T12:00:00Z", "posted_at": "2026-09-01",
    }
    if added is not None:
        record["first_seen"] = added
    record.update(changes)
    return record


def rendered_datasets(jobs):
    data = {
        "schema_version": 1, "generated_at": "2026-11-20T12:00:00Z",
        "sources": [], "jobs": jobs,
    }
    rendered = page.render_page(data, (ROOT / "career/potential-jobs/template.html").read_text())
    datasets = []

    def starttag(tag, attributes):
        attributes = dict(attributes)
        if tag != "article" or "data-job" not in attributes:
            return
        datasets.append({
            re.sub(r"-([a-z])", lambda match: match.group(1).upper(), key[5:]): value
            for key, value in attributes.items() if key.startswith("data-")
        })

    parser = HTMLParser()
    parser.handle_starttag = starttag
    parser.feed(rendered)
    return datasets


HARNESS = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const cards = input.cards.map(dataset => ({dataset, hidden: false}));
const handlers = {};
const animationFrames = [];
const defaults = {
  'job-search': '', 'job-category': 'all', 'job-type': 'all',
  'job-region': 'all', 'job-status': 'current',
  'job-added-from': '', 'job-added-through': '', 'job-sort': 'recommended',
};
function element(value = '') {
  return {
    value, textContent: '', hidden: true, attributes: {}, validationMessage: '',
    setCustomValidity(message) { this.validationMessage = message; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    removeAttribute(name) { delete this.attributes[name]; },
    getAttribute(name) { return this.attributes[name] ?? null; },
  };
}
const nodes = Object.fromEntries(Object.entries(defaults).map(([id, value]) => [id, element(value)]));
nodes['job-filters'] = {hidden: true, addEventListener: (name, callback) => {handlers[name] = callback;}};
nodes['results-count'] = element();
nodes['no-results'] = element();
nodes['job-date-error'] = element();
nodes['job-date-error'].textContent = input.errorText;
nodes['job-list'] = {
  children: [...cards],
  appendChild(card) {
    this.children = this.children.filter(existing => existing !== card);
    this.children.push(card);
    return card;
  },
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  document: {
    getElementById: id => nodes[id],
    querySelectorAll: () => [...nodes['job-list'].children],
    querySelector: () => null,
  },
  requestAnimationFrame: callback => animationFrames.push(callback),
});
function flushFrames() {
  while (animationFrames.length) animationFrames.shift()();
}
function state() {
  return {
    visible: nodes['job-list'].children.filter(card => !card.hidden).map(card => card.dataset.job),
    order: nodes['job-list'].children.map(card => card.dataset.job),
    count: nodes['results-count'].textContent,
    emptyHidden: nodes['no-results'].hidden,
    errorHidden: nodes['job-date-error'].hidden,
    error: nodes['job-date-error'].textContent,
    values: Object.fromEntries(Object.keys(defaults).map(id => [id, nodes[id].value])),
  };
}
flushFrames();
const results = [state()];
for (const action of input.actions) {
  if (action.reset) {
    // A native reset event occurs before the controls regain default values.
    handlers.reset();
    for (const [id, value] of Object.entries(defaults)) nodes[id].value = value;
  } else {
    for (const [id, value] of Object.entries(action.values || {})) nodes[id].value = value;
    handlers[action.event || 'change']();
  }
  flushFrames();
  results.push(state());
}
process.stdout.write(JSON.stringify(results));
'''


@unittest.skipUnless(shutil.which("node"), "Node is needed to exercise the actual filter script")
class JobDateFilterTests(unittest.TestCase):
    def run_filters(self, jobs, actions):
        error_text = []
        capture_depth = 0

        def starttag(tag, attributes):
            nonlocal capture_depth
            if capture_depth or dict(attributes).get("id") == "job-date-error":
                capture_depth += 1

        def endtag(tag):
            nonlocal capture_depth
            capture_depth = max(0, capture_depth - 1)

        parser = HTMLParser()
        parser.handle_starttag = starttag
        parser.handle_endtag = endtag
        parser.handle_data = lambda text: error_text.append(text) if capture_depth else None
        parser.feed((ROOT / "career/potential-jobs/template.html").read_text())
        result = subprocess.run(
            [shutil.which("node"), "-e", HARNESS, str(ROOT / "career/potential-jobs/jobs.js")],
            input=json.dumps({"cards": rendered_datasets(jobs), "actions": actions,
                              "errorText": "".join(error_text)}),
            text=True, capture_output=True, check=True,
            # Filters must follow the supplied Stockholm calendar date, even if
            # the visitor/browser runs in a different timezone.
            env={**os.environ, "TZ": "America/Los_Angeles"},
        )
        return json.loads(result.stdout)

    def test_date_range_is_inclusive_and_uses_stockholm_day(self):
        rows = [
            job("before", "2026-11-04T12:00:00Z"),
            job("lower", "2026-11-04T23:30:00Z"),
            job("upper", "2026-11-07T22:59:00Z"),
            job("after", "2026-11-07T23:00:00Z"),
            job("unknown"),
        ]
        result = self.run_filters(rows, [
            {"values": {"job-added-from": "2026-11-05", "job-added-through": "2026-11-07"}},
            {"values": {"job-added-through": "2026-11-05"}},
        ])
        self.assertEqual(result[1]["visible"], ["lower", "upper"])
        self.assertEqual(result[1]["count"], "2 of 5 listings")
        self.assertEqual(result[2]["visible"], ["lower"])
        self.assertTrue(result[2]["errorHidden"])

    def test_one_sided_ranges_exclude_missing_or_malformed_added_dates(self):
        rows = [
            job("early", "2026-11-01T12:00:00Z"),
            job("late", "2026-11-05T12:00:00Z"),
            job("unknown"), job("malformed", "not-a-date"),
        ]
        result = self.run_filters(rows, [
            {"values": {"job-added-from": "2026-11-05"}},
            {"values": {"job-added-from": "", "job-added-through": "2026-11-01"}},
            {"values": {"job-added-through": ""}},
        ])
        self.assertEqual(result[1]["visible"], ["late"])
        self.assertEqual(result[2]["visible"], ["early"])
        self.assertEqual(set(result[3]["visible"]), {"early", "late", "unknown", "malformed"})

    def test_date_filter_uses_first_seen_not_posting_or_refresh_date(self):
        row = job("saved", "2026-11-05T10:00:00Z")
        result = self.run_filters([row], [
            {"values": {"job-added-from": "2026-09-01", "job-added-through": "2026-09-01"}},
            {"values": {"job-added-from": "2026-11-20", "job-added-through": "2026-11-20"}},
            {"values": {"job-added-from": "2026-11-05", "job-added-through": "2026-11-05"}},
        ])
        self.assertEqual(result[1]["visible"], [])
        self.assertEqual(result[2]["visible"], [])
        self.assertEqual(result[3]["visible"], ["saved"])
        self.assertEqual(result[3]["count"], "1 of 1 listing")

    def test_added_range_combines_with_search_region_and_status(self):
        rows = [
            job("gm", "2026-11-05T09:00:00Z", company="General Motors"),
            job("gm-older", "2026-10-10T09:00:00Z", company="General Motors"),
            job("gm-sweden", "2026-11-05T09:00:00Z", company="General Motors",
                location="Gothenburg, Sweden", region="Europe"),
            job("gm-closed", "2026-11-05T09:00:00Z", company="General Motors", status="closed"),
            job("other", "2026-11-05T09:00:00Z", company="Other Company"),
        ]
        result = self.run_filters(rows, [
            {"values": {"job-search": "General Motors", "job-region": "US",
                        "job-added-from": "2026-11-01"}, "event": "input"},
            {"values": {"job-status": "all"}},
            {"values": {"job-region": "Sweden"}},
        ])
        self.assertEqual(result[1]["visible"], ["gm"])
        self.assertEqual(result[2]["visible"], ["gm", "gm-closed"])
        self.assertEqual(result[3]["visible"], ["gm-sweden"])

    def test_date_sort_is_chronological_and_recommended_order_is_restored(self):
        rows = [
            job("pinned-old", "2026-11-01T09:00:00Z", pinned=True),
            job("newest", "2026-11-05T11:00:00Z", score=10),
            job("same-day-earlier", "2026-11-05T08:00:00Z", score=90),
            job("unknown", score=100),
            job("malformed", "not-a-date", score=95),
        ]
        result = self.run_filters(rows, [
            {"values": {"job-sort": "newest"}},
            {"values": {"job-sort": "oldest"}},
            {"values": {"job-sort": "recommended"}},
        ])
        self.assertEqual(result[0]["visible"],
                         ["pinned-old", "unknown", "malformed", "same-day-earlier", "newest"])
        self.assertEqual(result[1]["visible"],
                         ["newest", "same-day-earlier", "pinned-old", "unknown", "malformed"])
        self.assertEqual(result[2]["visible"],
                         ["pinned-old", "same-day-earlier", "newest", "unknown", "malformed"])
        self.assertEqual(result[3]["order"], result[0]["order"])

    def test_reversed_range_has_actionable_error_and_reset_restores_results(self):
        rows = [job("one", "2026-11-01T09:00:00Z"), job("two", "2026-11-05T09:00:00Z")]
        result = self.run_filters(rows, [
            {"values": {"job-added-from": "2026-11-07", "job-added-through": "2026-11-01",
                        "job-sort": "newest", "job-search": "unmatched"}},
            {"reset": True},
        ])
        self.assertEqual(result[1]["visible"], [])
        self.assertFalse(result[1]["errorHidden"])
        self.assertTrue(result[1]["error"].strip())
        self.assertTrue(result[1]["emptyHidden"])
        self.assertEqual(result[2]["visible"], result[0]["visible"])
        self.assertEqual(result[2]["order"], result[0]["order"])
        self.assertTrue(result[2]["errorHidden"])
        self.assertTrue(result[2]["emptyHidden"])
        self.assertEqual(result[2]["values"]["job-added-from"], "")
        self.assertEqual(result[2]["values"]["job-added-through"], "")
        self.assertEqual(result[2]["values"]["job-sort"], "recommended")


if __name__ == "__main__":
    unittest.main()
