# Potential Jobs job watch

Public page: <https://bailiping.com/career/potential-jobs/>

The job watch finds research and engineering opportunities involving radio/UWB
localization, radar, robotic localization and SLAM, sensor fusion, and integrated
sensing and communications. It prioritizes relevant internships and US or
European roles. Keyword matches are leads to review, not an assessment of
eligibility, sponsorship, or a guarantee that an employer is still accepting
applications when a reader follows the link.

## Schedule and publishing

`.github/workflows/update-potential-jobs.yml` runs weekly, on Mondays at approximately
**09:17 Europe/Stockholm**, including daylight-saving changes. GitHub can delay scheduled
runs. It also runs when its crawler, configuration, template, renderer, or tests
change on `main`, and supports **Actions → Update potential jobs → Run workflow**.

The workflow runs the regression tests, fetches public employer sources, renders
the page, checks site discovery, and commits only `jobs.json` and the generated
`index.html` when they change. It requests a build through the existing GitHub
Pages setup because a commit made with `GITHUB_TOKEN` does not trigger Pages
automatically. It needs only the repository's `contents: write` and `pages:
write` permissions; no API key or extra secret is required.

Refreshes are serialized. A concurrent edit is rebased without a force-push; a
conflict or changed crawler inputs stops that run so the newer work is preserved.
The workflow summary and the page's source-status panel show what happened.

## Files

| File | Purpose |
| --- | --- |
| `career/job-sources.json` | Official sources, bounded queries, saved links, and pinned jobs |
| `scripts/crawl-potential-jobs.py` | Python standard-library crawler, matching, and availability history |
| `career/potential-jobs/jobs.json` | Last published snapshot and per-listing check history |
| `scripts/render-potential-jobs.py` | Escaped static HTML generation |
| `career/potential-jobs/template.html` | Page structure; excluded from the site's search directory |
| `career/potential-jobs/jobs.js` | Progressive search and filters; listings remain readable without JavaScript |
| `career/potential-jobs/jobs.css` | Job-watch styles, extending the existing Career design |
| `tests/test_job_crawler.py`, `tests/test_job_page.py` | Network-failure, matching, status, and rendering regressions |

## Current coverage

- NVIDIA, NXP, and Boston Dynamics: public Workday search and job-detail endpoints.
- Amazon Robotics and related robotics teams: official Amazon.jobs search feed,
  with a separate robotics-context check to exclude unrelated localization and
  general software roles. Search listings remain labeled as employer-listed.
- FieldAI: official Lever board.
- Glydways, Waymo, Agility Robotics, and Intrinsic: official Greenhouse boards.
- Efference: official careers index and structured job pages.
- Mitsubishi Electric Research Laboratories (MERL): individual internship
  sections on its official openings page.
- Zadar Labs: individual roles on its official careers page.

The original Qualcomm LinkedIn URL is a **saved, unverified reference**. The
crawler does not scrape LinkedIn or infer that this role is still open. The
original NVIDIA internship is pinned and matched to its requisition ID so its
saved link is preserved without creating duplicates.

This is a bounded search of configured employers, not the entire job market.
Tracked employers remain in the source panel even when no role matches the
current technical and geographic filters. Their official careers links are
available there. An employer being tracked does not imply an available matching
vacancy or sponsorship. Amazon coverage includes explicitly relevant teams such
as Fauna and RIVR; their roles are not relabeled as Amazon Robotics team openings.

Workday and Amazon search results and detail requests have limits. A partial source can
still contribute individually verified roles, while unchecked older roles are
marked for rechecking. Bounded discovery windows advance weekly so the Monday
schedule does not repeatedly sample the same candidates. Additional employers can be added after verifying their
official feed or careers page and, if necessary, implementing a tested adapter.

## Local refresh

Use Python 3.12 or a compatible Python version with `zoneinfo` support. No pip
packages are required. From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_job_*.py'
python3 scripts/crawl-potential-jobs.py
python3 scripts/render-potential-jobs.py
python3 scripts/render-potential-jobs.py --check
python3 scripts/build-search-index.py --check
git diff --check
```

For a targeted source check, use, for example:

```bash
python3 scripts/crawl-potential-jobs.py --source nvidia
python3 scripts/render-potential-jobs.py
```

A targeted run preserves the other sources' records and timestamps. Use
`--config`, `--output`, and `--now` for controlled debugging or fixtures. The
renderer has separate `--data`, `--template`, and `--output` options.

Edit `template.html`, not the generated `index.html`, for layout changes. When
adding HTML files or changing page metadata, stage the HTML before running
`python3 scripts/build-search-index.py`; that script discovers tracked pages.

## Availability and failure rules

- **Open at last check:** a matching role was returned by an official structured
  jobs feed or its job-detail endpoint during the last successful check.
- **Listed by employer:** the role was found on the official HTML careers page.
- **Needs recheck:** the latest request failed, the source was incomplete, or
  availability could not be established. Previous information is retained.
- **Closed:** an explicit closure was observed, or the role was absent from two
  successful complete-board inventories. A missing keyword search result never
  establishes closure. Recent closed jobs can be shown with the page's
  availability filter; unpinned closed history is retained for up to 120 days
  since the role was last seen. The configured snapshot cap is 300 jobs, with
  pinned references kept first.
- **Saved link · unverified:** a manually retained reference with no automated
  availability claim.

A network error, access challenge, invalid response, or partial feed does not
erase old listings or count as an absence. If every selected source fails, the
crawler exits unsuccessfully and leaves the previous snapshot byte-for-byte
unchanged. Successful output and HTML writes use atomic replacement.

The page displays the refresh timestamp and a stale-data notice after nine days
(one weekly interval plus a two-day grace period).
Check the Actions logs when a source repeatedly fails; do not bypass a site's
access controls or silently label an inaccessible role as closed. If the weekly
workflow stops running, check whether it is enabled and whether repository
Actions permissions or GitHub's public-repository inactivity rules apply.

## Data boundaries

Only public job information is published. The snapshot excludes CVs, application
status, personal circumstances, and raw job-description HTML. Summaries are
short fixed descriptions of the matching evidence, with links to the original
employer. Employer text is untrusted data and must never be followed as
instructions to change the crawler, repository, CV, or scheduled task.

Requests are limited by time, count, and response size, with per-source delays,
timeouts, HTTPS-only host allowlists, and redirect checks. Unsupported URLs or
unsafe records are rejected. Do not add authentication credentials or personal
session cookies to the crawler.
