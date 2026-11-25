"""Ashby's public v1 feed: visibility, geographic scope, and safe history."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ashby_crawler', ROOT / 'scripts/crawl-potential-jobs.py')
crawler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = crawler
SPEC.loader.exec_module(crawler)

ID = '250e9c69-11d9-4d43-adde-dadf725eef13'
NOW = '2026-11-05T10:00:00Z'
LATER = '2026-11-12T10:00:00Z'


def source():
    return {'id': 'wayve', 'name': 'Wayve official careers', 'company': 'Wayve', 'type': 'ashby',
            'allowed_hosts': ['api.ashbyhq.com', 'jobs.ashbyhq.com', 'wayve.ai'],
            'api_url': 'https://api.ashbyhq.com/posting-api/job-board/wayve',
            'index_url': 'https://jobs.ashbyhq.com/wayve'}


def posting(**changes):
    # Synthetic content with the documented v1 shape and verified location pattern.
    result = {'id': ID, 'title': 'Staff Robotics Engineer, Localization',
              'jobUrl': f'https://jobs.ashbyhq.com/wayve/{ID}',
              'applyUrl': f'https://jobs.ashbyhq.com/wayve/{ID}/application',
              'location': 'Sunnyvale, California USA', 'address': None,
              'secondaryLocations': [{'location': 'London, United Kingdom', 'address': None}],
              'isListed': True, 'isRemote': True, 'workplaceType': 'Hybrid',
              'descriptionHtml': '<p>Develop vehicle localization, sensor fusion, and state estimation in C++ and Python.</p>',
              'employmentType': 'FullTime', 'department': 'AV Engineering', 'team': 'Sensor Calibration and Fusion',
              'publishedAt': '2026-10-13T18:57:00.101+00:00'}
    result.update(changes)
    return result


class Client:
    requests = 1
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
    def get_json(self, url):
        if self.error:
            raise self.error
        return deepcopy(self.payload)


def fetch(*items, payload=None):
    return crawler.fetch_source(source(), Client(payload if payload is not None else {'apiVersion': '1', 'jobs': list(items)}))


def config():
    return {'schema_version': 1, 'sources': [source()], 'manual_jobs': [], 'pinned_jobs': []}


def saved_state():
    return crawler.merge_results({'jobs': [], 'sources': []}, [fetch(posting())], config(), NOW)


class AshbyTests(unittest.TestCase):
    def test_live_shape_keeps_both_hiring_regions_and_hybrid_is_not_remote(self):
        result = fetch(posting())
        self.assertEqual(result.status, 'ok')
        self.assertTrue(result.complete)
        job, = result.records
        self.assertEqual(job['id'], 'wayve:' + ID)
        self.assertEqual(job['employment_type'], 'Full-time')
        self.assertEqual(job['status'], 'open')
        self.assertEqual(job['regions'], ['US', 'Europe'])
        self.assertNotIn('Remote', job['location'])
        self.assertIn('Hybrid', job['location'])
        self.assertIn('London', job['location'])
        self.assertNotIn('posted_at', job)  # Ashby supplies last-publication time.
        hybrid_label = fetch(posting(location='Remote / Sunnyvale, California USA'))
        self.assertEqual(hybrid_label.records[0]['region'], 'US')
        self.assertEqual(hybrid_label.records[0]['regions'], ['US', 'Europe'])
        merged = crawler.merge_results({'jobs': [], 'sources': []}, [result], config(), NOW)
        self.assertEqual(merged['jobs'][0]['regions'], ['US', 'Europe'])
        self.assertEqual(merged['jobs'][0]['first_seen'], NOW)

    def test_optional_id_uses_the_exact_board_url_uuid(self):
        item = posting()
        del item['id']
        result = fetch(item)
        self.assertTrue(result.complete)
        self.assertEqual(result.records[0]['id'], 'wayve:' + ID)
        mismatch = fetch(posting(id='other'))
        self.assertEqual(mismatch.status, 'partial')
        self.assertFalse(mismatch.complete)
        self.assertEqual(mismatch.records, [])

    def test_structured_addresses_normalize_country_codes_and_remote_type(self):
        result = fetch(posting(location='Stockholm', address={'postalAddress': {'addressCountry': 'SWE'}},
                               secondaryLocations=[{'location': 'London', 'address': {'addressCountry': 'GBR'}}],
                               workplaceType='Remote'))
        job, = result.records
        self.assertEqual(job['regions'], ['Europe', 'Remote'])
        self.assertEqual(job['region'], 'Remote')
        self.assertIn('Sweden', job['location'])
        self.assertIn('United Kingdom', job['location'])
        onsite = fetch(posting(workplaceType='OnSite', isRemote=False, secondaryLocations=[]))
        self.assertEqual(onsite.records[0]['regions'], ['US'])
        unknown = fetch(posting(location='', address=None, secondaryLocations=[], workplaceType=None))
        self.assertEqual(unknown.records[0]['regions'], ['Unknown'])
        self.assertNotIn('Remote', unknown.records[0]['location'])

    def test_remote_does_not_override_explicit_out_of_scope_country(self):
        outside = fetch(posting(location='Remote in Yokohama', address={'postalAddress': {'addressCountry': 'JPN'}},
                                secondaryLocations=[], workplaceType='Remote'))
        self.assertEqual(outside.records, [])
        self.assertTrue(outside.complete)
        mixed = fetch(posting(location='Yokohama, Japan', secondaryLocations=[{'location': 'London', 'address': {'addressCountry': 'GBR'}}]))
        self.assertEqual(mixed.records[0]['regions'], ['Other', 'Europe'])

    def test_unlisted_is_not_discovered_or_treated_as_closed(self):
        result = fetch(posting(isListed=False))
        self.assertTrue(result.complete)
        self.assertEqual(result.records, [])
        self.assertIn('wayve:' + ID, result.seen_ids)
        previous = saved_state()
        for date in (LATER, '2026-11-19T10:00:00Z'):
            previous = crawler.merge_results(previous, [result], config(), date)
        job, = previous['jobs']
        self.assertNotEqual(job['status'], 'closed')
        self.assertEqual(job['missing_count'], 0)
        self.assertEqual(job['check_status'], 'partial')
        self.assertEqual(job['first_seen'], NOW)

    def test_malformed_entries_never_make_a_complete_inventory(self):
        for changes in ({'isListed': None}, {'secondaryLocations': 'London'}, {'descriptionHtml': []},
                        {'workplaceType': 'Somewhere'}, {'isRemote': 'yes'}, {'address': []},
                        {'title': ''}, {'jobUrl': 'https://evil.example/wayve/' + ID},
                        {'jobUrl': 'https://jobs.ashbyhq.com/other/' + ID}):
            with self.subTest(changes=changes):
                result = fetch(posting(**changes))
                self.assertFalse(result.complete)
                self.assertEqual(result.records, [])
                merged = crawler.merge_results(saved_state(), [result], config(), LATER)
                self.assertNotEqual(merged['jobs'][0]['status'], 'closed')
                self.assertEqual(merged['jobs'][0]['missing_count'], 0)
        duplicate = fetch(posting(), posting())
        self.assertFalse(duplicate.complete)
        self.assertEqual(len(duplicate.records), 1)

    def test_wrong_schema_and_network_failures_preserve_existing_records(self):
        results = [fetch(payload={'jobs': []}), fetch(payload={'apiVersion': '2', 'jobs': []}),
                   fetch(payload={'apiVersion': '1', 'jobs': None}),
                   crawler.fetch_source(source(), Client(error=HTTPError(source()['api_url'], 403, 'Forbidden', {}, None)))]
        for result in results:
            with self.subTest(error=result.error):
                self.assertEqual(result.status, 'error')
                self.assertFalse(result.complete)
                merged = crawler.merge_results(saved_state(), [result], config(), LATER)
                self.assertEqual(len(merged['jobs']), 1)
                self.assertEqual(merged['jobs'][0]['first_seen'], NOW)
                self.assertEqual(merged['jobs'][0]['check_status'], 'error')
                self.assertNotEqual(merged['jobs'][0]['status'], 'closed')

    def test_only_repeated_complete_absence_can_close_a_prior_job(self):
        first = crawler.merge_results(saved_state(), [fetch()], config(), LATER)
        self.assertNotEqual(first['jobs'][0]['status'], 'closed')
        second = crawler.merge_results(first, [fetch()], config(), '2026-11-19T10:00:00Z')
        self.assertEqual(second['jobs'][0]['status'], 'closed')

    def test_civilian_and_technical_matching_still_applies(self):
        for changes in ({'department': 'Defence Engineering'}, {'team': 'Military Systems'},
                        {'title': 'Senior Recruiter', 'descriptionHtml': '<p>Hire robotics engineers.</p>'}):
            result = fetch(posting(**changes))
            self.assertEqual(result.records, [])
            self.assertTrue(result.complete)
            self.assertIn('wayve:' + ID, result.seen_ids)

    def test_source_endpoint_and_optional_regions_are_validated(self):
        self.assertEqual(crawler.validate_config(config()), config())
        for url in ('https://api.ashbyhq.com/job.list', 'https://jobs.ashbyhq.com/wayve'):
            cfg = config()
            cfg['sources'][0]['api_url'] = url
            with self.assertRaises(ValueError):
                crawler.validate_config(cfg)
        good = fetch(posting()).records[0]
        duplicate = dict(good, regions=['US', 'Europe', 'US'])
        self.assertEqual(crawler._public_job(duplicate, source())['regions'], ['US', 'Europe'])
        for bad in ('US', [], ['Invalid'], [None]):
            with self.subTest(regions=bad), self.assertRaises(ValueError):
                crawler._public_job(dict(good, regions=bad), source())
        legacy = dict(good)
        del legacy['regions']
        self.assertNotIn('regions', crawler._public_job(legacy, source()))


if __name__ == '__main__':
    unittest.main()
