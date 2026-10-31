#!/usr/bin/env python3
"""Render a static, searchable job page from the crawler's public snapshot.

Only escaped text and validated HTTPS job links enter the generated HTML.
The page remains readable without JavaScript; JavaScript only filters cards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "career/potential-jobs"
CATEGORIES = {
    "radio-localization": "Radio localization / UWB",
    "radar-sensing": "Sensing",
    "robotics-localization": "Robotics & SLAM",
    "wireless-isac": "Wireless & ISAC",
}
STATUS_LABELS = {
    "open": "Open at last check",
    "listed": "Listed by employer",
    "unknown": "Needs recheck",
    "closed": "Closed",
    "saved": "Saved link · unverified",
}
TYPES = {"Internship", "Full-time", "Unknown"}
REGIONS = {"US", "Europe", "Remote", "Other", "Unknown"}


def escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def public_wording(value: object) -> str:
    """Apply the Career wording preference without changing source records or URLs."""
    wording = {
        "Radar is named in the role's title": "Role title describes sensing work",
        "Radar is named in the role's title or technical description":
            "Role title or technical description describes sensing work",
        "Radar mentioned": "Sensing mentioned",
    }
    text = str(value or "")
    text = wording.get(text, text)
    return " ".join(re.sub(r"\bradar(?:-based)?\b\s*", "", text, flags=re.IGNORECASE).split())


def is_sweden_location(value: object) -> bool:
    """Match explicit country/city location evidence, never role descriptions."""
    location = str(value or "")
    if re.search(r"\b(?:sweden|sverige)\b", location, re.IGNORECASE):
        return True
    # Employer feeds sometimes supply only a city. Keep this list limited to
    # Swedish hiring locations, and reject explicit foreign locations sharing
    # a city name (for example Gothenburg, Nebraska or Lund, British Columbia).
    cities = (r"\b(?:stockholm|gothenburg|göteborg|malmö|malmo|uppsala|lund|"
              r"linköping|linkoping|norrköping|norrkoping|västerås|vasteras|"
              r"luleå|lulea|umeå|umea|karlskrona|jönköping|jonkoping|"
              r"huskvarna|kista|solna)\b")
    foreign = (r"\b(?:united states|usa|u\.?s\.?a?\.?|canada|switzerland|"
               r"australia|germany|norway|denmark|finland|united kingdom|"
               r"nebraska|wisconsin|british columbia)\b|"
               r",\s*(?:CA|MA|WA|TX|CO|NY|MI|VA|NC|IL|PA|AZ|OR|FL|OH|WI|NE|BC)\b")
    for part in re.split(r"\s*[/;|]\s*", location):
        if not re.search(foreign, part, re.IGNORECASE) and re.search(cities, part, re.IGNORECASE):
            return True
    return False


def https_url(value: object) -> str:
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        raise ValueError("A listing URL must be a valid HTTPS URL")
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.hostname or parts.username
            or parts.password or parts.port not in (None, 443)):
        raise ValueError("A listing URL must use HTTPS without credentials")
    return value


def display_time(value: object) -> str:
    if not value:
        return "not recorded"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(ZoneInfo("Europe/Stockholm")).strftime("%d %b %Y, %H:%M %Z")
    except (ValueError, TypeError):
        return "not recorded"


def effective_status(job: dict) -> str:
    status = job.get("status", "unknown")
    if status not in STATUS_LABELS:
        raise ValueError(f"Invalid job status: {status!r}")
    if status not in {"closed", "saved"} and job.get("check_status") in {"error", "partial"}:
        return "unknown"
    return status


def validate_snapshot(data: dict) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported job snapshot schema")
    if not isinstance(data.get("jobs"), list) or not isinstance(data.get("sources"), list):
        raise ValueError("The snapshot must contain jobs and sources lists")
    if display_time(data.get("generated_at")) == "not recorded":
        raise ValueError("The snapshot must have a valid generated_at timestamp")
    seen = set()
    for job in data["jobs"]:
        if not isinstance(job, dict):
            raise ValueError("Invalid job record")
        for field in ("id", "company", "title", "source_id", "url"):
            if not isinstance(job.get(field), str) or not job[field].strip():
                raise ValueError(f"Missing job field: {field}")
        if job["id"] in seen:
            raise ValueError(f"Duplicate job id: {job['id']}")
        seen.add(job["id"])
        https_url(job["url"])
        if job.get("category") not in CATEGORIES:
            raise ValueError("Invalid job category")
        if job.get("employment_type", "Unknown") not in TYPES:
            raise ValueError("Invalid employment type")
        if job.get("region", "Unknown") not in REGIONS:
            raise ValueError("Invalid job region")
        for field in ("match_reasons", "requirements"):
            if not isinstance(job.get(field, []), list) or not all(isinstance(x, str) for x in job.get(field, [])):
                raise ValueError(f"Invalid {field}")
        effective_status(job)


def render_card(job: dict) -> str:
    # Keep original employer titles, URLs, matching data, and timestamps in the
    # snapshot. Only the card copy and its search text use the public wording.
    sweden = "true" if is_sweden_location(job.get("location")) else "false"
    job = dict(job)
    for field in ("company", "title", "location", "summary"):
        job[field] = public_wording(job.get(field))
    job["title"] = job["title"] or "Sensing role"
    for field in ("match_reasons", "requirements"):
        job[field] = [text for value in job.get(field, []) if (text := public_wording(value))]
    status = effective_status(job)
    category = job["category"]
    kind = job.get("employment_type", "Unknown")
    region = job.get("region", "Unknown")
    key = hashlib.sha256(job["id"].encode()).hexdigest()[:16]
    title_id = "job-" + key
    url = escape(https_url(job["url"]))
    metadata = [job.get("location") or "Location not specified"]
    if kind != "Unknown":
        metadata.append(kind)
    if job.get("requisition_id"):
        metadata.append(str(job["requisition_id"]))
    search = " ".join(str(job.get(field) or "") for field in (
        "company", "title", "location", "requisition_id", "summary"
    )) + " " + " ".join(job.get("match_reasons", []) + job.get("requirements", []))
    classes = "job-card" + (" is-pinned" if job.get("pinned") else "") + (" is-closed" if status == "closed" else "")
    saved = '<span class="saved-label">Saved</span>' if job.get("pinned") else ""
    summary = f'<p class="job-summary">{escape(job.get("summary", ""))}</p>' if job.get("summary") else ""
    reasons = job.get("match_reasons", [])[:3]
    evidence = '<ul class="job-evidence">' + "".join(f"<li>{escape(reason)}</li>" for reason in reasons) + "</ul>" if reasons else ""
    # Keep eligibility and seniority cues visible even when a role lists many
    # tools. These are evidence labels, never a claim about the reader's eligibility.
    priority_flags = {"Citizenship/clearance conditions mentioned", "Sponsorship restriction mentioned", "Current US work authorization required", "Senior/experienced role"}
    flags = sorted(job.get("requirements", []), key=lambda value: value not in priority_flags)[:4]
    requirements = '<p class="job-requirements">' + escape(" · ".join(flags)) + "</p>" if flags else ""
    if status == "saved":
        date_note = "Saved for reference; availability is not checked automatically."
    elif status == "unknown":
        verified = job.get("last_verified") or job.get("last_seen")
        date_note = "Latest check incomplete. " + (f"Last observed {display_time(verified)}." if verified else "Availability has not been confirmed.")
    elif status == "closed":
        date_note = "Closed at the latest completed check."
    elif status == "listed":
        date_note = f"Employer listing observed {display_time(job.get('last_seen') or job.get('last_checked'))}."
    else:
        date_note = f"Verified {display_time(job.get('last_verified') or job.get('last_seen'))}."
    hidden = " hidden" if status == "closed" else ""
    return f'''<article class="{classes}" data-job="{escape(job['id'])}" data-category="{category}" data-type="{escape(kind)}" data-region="{escape(region)}" data-sweden="{sweden}" data-status="{status}" data-search="{escape(search.lower())}" aria-labelledby="{title_id}"{hidden}>
  <div class="card-top"><p class="company">{escape(job['company'])}</p>{saved}</div>
  <h3 id="{title_id}"><a href="{url}">{escape(job['title'])}</a></h3>
  <p class="job-meta">{escape(' · '.join(metadata))}</p>
  <div class="job-tags"><span class="tag">{escape(CATEGORIES[category])}</span><span class="tag status-{status}">{STATUS_LABELS[status]}</span></div>
  {summary}{evidence}{requirements}
  <div class="job-card-bottom"><a class="job-link" href="{url}">View posting</a><p class="checked-date">{escape(date_note)}</p></div>
</article>'''


def render_snapshot(data: dict) -> str:
    sources = [s for s in data["sources"] if s.get("status") != "manual"]
    healthy = sum(s.get("status") == "ok" for s in sources)
    confirmed = sum(effective_status(j) == "open" for j in data["jobs"])
    internships = sum(j.get("employment_type") == "Internship" and effective_status(j) not in {"closed", "saved"} for j in data["jobs"])
    stamp = escape(data["generated_at"])
    return f'''<section class="snapshot-panel" aria-label="Latest refresh">
  <div><p><strong>{confirmed} confirmed open</strong> · {internships} internship listings</p><p class="snapshot-note">{healthy} of {len(sources)} employer sources refreshed successfully</p></div>
  <div><p>Updated <time datetime="{stamp}" data-snapshot-time="{stamp}">{escape(display_time(data['generated_at']))}</time></p><p class="snapshot-note">Weekly refresh · Mondays around 09:17, Stockholm time</p></div>
</section>'''


def render_sources(data: dict) -> str:
    labels = {"ok": "Refreshed", "partial": "Partly checked · previous results retained", "error": "Check unavailable · previous results retained", "manual": "Saved reference link"}
    items = []
    for source in data["sources"]:
        label = labels.get(source.get("status"), "Not checked")
        name = escape(source.get("name") or source.get("id"))
        if source.get("careers_url"):
            name = f'<a href="{escape(https_url(source["careers_url"]))}">{name}</a>'
        matched = source.get("count")
        count_note = f'<span>{matched} matches in this check</span>' if isinstance(matched, int) and matched >= 0 else ""
        items.append(f'<li><strong>{name}</strong><span>{escape(label)}</span>{count_note}<span>{escape(display_time(source.get("checked_at")))}</span></li>')
    return '''<details class="source-details"><summary>Employers being tracked and update status</summary>
<p>Official employer feeds and career pages are checked weekly. New matching roles are added automatically. Each employer remains on this watchlist even when no current role matches the filters. Follow its careers link to browse all openings.</p>
<p>Focus labels describe keyword evidence, not a guarantee of suitability or work authorization. The original employer posting is the source of truth.</p>
<p>A failed or incomplete check keeps the previous listing and marks it for rechecking. Roles close only after explicit closure or repeated absence from a successfully read complete job board. Closed roles remain available through the availability filter. Saved reference links may be unverified.</p>
<p>Coverage follows the employers configured for this job watch; it is not an exhaustive search of every employer.</p>
<ul class="source-grid">''' + "".join(items) + "</ul></details>"


def render_page(data: dict, template: str) -> str:
    validate_snapshot(data)
    for marker in ("<!-- SNAPSHOT -->", "<!-- COUNT -->", "<!-- JOBS -->", "<!-- SOURCES -->"):
        if template.count(marker) != 1:
            raise ValueError(f"Expected exactly one template marker: {marker}")
    jobs = sorted(data["jobs"], key=lambda j: (
        effective_status(j) == "closed", not bool(j.get("pinned")),
        effective_status(j) in {"unknown", "saved"},
        -float(j.get("score") or 0), j["company"].lower(), j["title"].lower(), j["id"],
    ))
    current = sum(effective_status(j) != "closed" for j in jobs)
    page = template.replace('    <meta name="robots" content="noindex">\n', "")
    page = page.replace("<!-- SNAPSHOT -->", render_snapshot(data))
    page = page.replace("<!-- COUNT -->", f"{current} of {len(jobs)} listings")
    page = page.replace("<!-- SOURCES -->", render_sources(data))
    return page.replace("<!-- JOBS -->", "\n" + "\n".join(render_card(j) for j in jobs) + "\n")


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.chmod(0o644)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PAGE / "jobs.json")
    parser.add_argument("--template", type=Path, default=PAGE / "template.html")
    parser.add_argument("--output", type=Path, default=PAGE / "index.html")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    rendered = render_page(data, args.template.read_text(encoding="utf-8"))
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            print("Potential Jobs HTML is out of date. Run scripts/render-potential-jobs.py.")
            return 1
        print("Potential Jobs HTML matches the job snapshot.")
    else:
        atomic_write(args.output, rendered)
        print(f"Rendered {len(data['jobs'])} job listings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
