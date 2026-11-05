#!/usr/bin/env python3
"""Refresh a bounded, public-only collection of relevant employer job postings.

No credentials, browser sessions, search-engine scraping, or LinkedIn requests are
used. Provider text is data, never configuration or instructions. A failed or
incomplete check cannot close a previously observed job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import hashlib
from html import unescape
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode
from urllib.request import Request, HTTPRedirectHandler, build_opener


ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = {"radio-localization", "radar-sensing", "robotics-localization", "wireless-isac"}
JOB_STATUSES = {"open", "listed", "unknown", "closed", "saved"}
SOURCE_TYPES = {"workday", "lever", "greenhouse", "ashby", "html", "amazon", "ericsson"}
ELIGIBILITY_VALUES = {
    "sponsorship": {"offered", "case_by_case", "not_offered", "not_stated", "unknown"},
    "access": {"no_restriction_stated", "export_review", "us_person", "citizenship_clearance", "unknown"},
    "work_authorization": {"required_now", "not_stated", "unknown"},
}
ELIGIBILITY_DEFAULTS = {"unknown", "not_stated", "no_restriction_stated"}
ELIGIBILITY_RESTRICTIONS = {"not_offered", "export_review", "us_person", "citizenship_clearance", "required_now"}
MAX_ELIGIBILITY_QUOTE_WORDS = 25
MAX_RESPONSE_BYTES = 6 * 1024 * 1024
INSTRUCTION_TEXT = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"(?:system|developer)\s+(?:prompt|message)|"
    r"(?:reveal|print|send|exfiltrate)\s+(?:your\s+)?(?:api\s*keys?|secrets?|credentials)|"
    r"you\s+are\s+(?:now\s+)?(?:chatgpt|an?\s+ai\s+assistant)", re.I
)
CHALLENGE = re.compile(
    r"verify (?:that )?you are (?:a )?(?:human|robot)|unusual traffic|"
    r"checking your browser|just a moment.{0,200}cloudflare|"
    r"access denied|enable javascript and cookies to continue", re.I | re.S
)
EXPLICIT_CLOSED = re.compile(
    r"(?:this|the) (?:job|position|vacancy) (?:is |has been )?(?:no longer available|closed|filled)|"
    r"no longer accepting applications for this (?:job|position)", re.I
)



# Defence manufacturers are excluded even if a particular job sounds civilian.
# Mixed-use employers require verified role-specific civilian assignment.
EXCLUDED_DEFENCE_COMPANIES = {
    "saab", "lockheed martin", "raytheon", "rtx", "northrop grumman",
    "bae systems", "general dynamics", "anduril", "shield ai",
    "rheinmetall", "l3harris", "leonardo drs",
}
MILITARY_ROLE_TERMS = re.compile(
    r"\b(?:military|militarised|militarized|defen[cs]e|weapons?|weaponry|"
    r"combat|battlefield|warfighting|warfare|armed forces|"
    r"försvarsmakten|försvarsindustrin|försvaret|försvar|"
    r"militär(?:a|t)?|vapensystem|krigföring|"
    r"army|navy|air force|department of defen[cs]e|"
    r"national[- ]security|intelligence (?:agency|community|programmes?|"
    r"programs?|operations)|dual[- ]use|"
    r"federal (?:autonomy|team|programmes?|programs?|contracts?))\b",
    re.I,
)


def is_military_role(title, description="", department=""):
    """Check role duties and ATS department, not generic company marketing.

    Historical DARPA affiliations, civilian background checks, veteran EEO
    notices, generic sensing and export controls are not military evidence.
    """
    title = clean_text(title, 260)
    department = clean_text(department, 300)
    if re.search(r"\bfederal\b", department, re.I):
        return True  # mixed-use federal team needs civilian review
    detail = clean_text(description, 30000)
    start = re.search(
        r"\b(?:about (?:the )?(?:role|job)|we(?: are|'re|’re) looking for|"
        r"what you(?:'|’)ll do|what you will do|your responsibilities|"
        r"the role|job responsibilities|responsibilities)\b", detail, re.I
    )
    if start:
        detail = detail[start.start():]
    end = re.search(
        r"\b(?:why join|about the company|equal opportunity employer|"
        r"we are an equal opportunity|equal employment opportunity|"
        r"(?:do|does) not discriminate|non[- ]discrimination|benefits and perks|"
        r"be part of the next robotics revolution)\b", detail, re.I
    )
    if end:
        detail = detail[:end.start()]
    # Lists of previous industries describe candidate experience, not the
    # current role's customers. Keep direct defence duties and ATS teams visible.
    detail = re.sub(
        r"\b(?:background|experience)\s+from\s+(?:related\s+)?industries\s+such\s+as\b"
        r"[^.!?]{0,320}(?:[.!?]|$)",
        " ", detail, flags=re.I,
    )
    return bool(MILITARY_ROLE_TERMS.search(" ".join((title, department, detail))))


@dataclass
class SourceResult:
    source_id: str
    status: str
    records: list[dict]
    complete: bool = False
    error: str | None = None
    checked_at: str | None = None
    closed_ids: list[str] = field(default_factory=list)
    requests: int = 0
    # All provider IDs, including irrelevant jobs, prevent false absence closures.
    seen_ids: list[str] = field(default_factory=list)
    attempted_ids: list[str] = field(default_factory=list)
    # Fresh role data was successfully screened and excluded. Search-card
    # sightings and failed detail requests are not rejection evidence.
    rejected_ids: list[str] = field(default_factory=list)


class FetchError(RuntimeError):
    pass


def safe_url(url: str, allowed_hosts: list[str] | set[str]) -> str:
    """Validate an absolute HTTPS URL against exact, trusted public host names."""
    if not isinstance(url, str) or not url or len(url) > 3000:
        raise ValueError("Invalid URL")
    if any(ord(c) < 33 for c in url) or "\\" in url:
        raise ValueError("URL contains whitespace or control characters")
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid URL port") from exc
    if parts.scheme != "https" or parts.username is not None or parts.password is not None:
        raise ValueError("Only public HTTPS URLs without credentials are allowed")
    if port not in (None, 443) or not host or host.endswith("."):
        raise ValueError("Invalid HTTPS host or port")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP-address destinations are not allowed")
    if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
        raise ValueError("Private host names are not allowed")
    if host not in {str(h).lower() for h in allowed_hosts}:
        raise ValueError("URL host is not in the configured allowlist")
    return url


class SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(urljoin(req.full_url, newurl), self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class RunBudget:
    def __init__(self, seconds=600, requests=240):
        self.deadline = time.monotonic() + min(max(seconds, 10), 600)
        self.remaining = min(max(requests, 1), 240)

    def consume(self):
        if self.remaining <= 0 or time.monotonic() >= self.deadline:
            raise FetchError("Run request/time budget exhausted")
        self.remaining -= 1


class HttpClient:
    def __init__(self, allowed_hosts, max_requests=30, budget=None, timeout=12, delay=0.15):
        self.allowed_hosts = allowed_hosts
        self.max_requests = min(max(int(max_requests), 1), 40)
        self.budget = budget or RunBudget()
        self.timeout = min(max(float(timeout), 1), 20)
        self.delay = min(max(float(delay), 0.1), 2)
        self.requests = 0
        self.last_request = 0.0
        self.opener = build_opener(SafeRedirectHandler(allowed_hosts))

    def _read(self, url, payload=None):
        safe_url(url, self.allowed_hosts)
        if self.requests >= self.max_requests:
            raise FetchError("Source request budget exhausted")
        self.budget.consume()
        wait = self.delay - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        self.requests += 1
        self.last_request = time.monotonic()
        headers = {"User-Agent": "BaiLiping-CareerDiscovery/1.0 (public employer postings)",
                   "Accept": "application/json,text/html;q=0.9"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers)
        remaining = max(0.1, self.budget.deadline - time.monotonic())
        with self.opener.open(req, timeout=min(self.timeout, remaining)) as response:
            safe_url(response.geturl(), self.allowed_hosts)
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise FetchError("Response exceeds the safe size limit")
            return raw.decode("utf-8", errors="replace")

    def get_json(self, url, **_kwargs):
        return json.loads(self._read(url))

    def post_json(self, url, payload, **_kwargs):
        return json.loads(self._read(url, payload))

    def get_text(self, url, **_kwargs):
        text = self._read(url)
        if CHALLENGE.search(text):
            raise FetchError("Employer returned an access challenge, not a job listing")
        return text


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "iframe", "svg", "noscript", "template"}:
            self.hidden += 1
        if not self.hidden and tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "iframe", "svg", "noscript", "template"} and self.hidden:
            self.hidden -= 1
        if not self.hidden and tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value, limit=1000):
    if value is None:
        return ""
    parser = TextOnly()
    parser.feed(unescape(str(value))[:150000])
    lines = []
    for line in "".join(parser.parts).splitlines():
        if INSTRUCTION_TEXT.search(line):
            continue
        line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)[:limit].strip()


def _unknown_eligibility():
    return {**{key: "unknown" for key in ELIGIBILITY_VALUES}, "evidence": [],
            "checked_at": None, "method": "posting_text"}


def _eligibility_timestamp(value):
    """Validate an observation date, never substitute the crawl/merge date."""
    if value is None:
        return None
    if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}T", value) or len(value) > 40:
        raise ValueError("Eligibility checked_at must be an ISO timestamp or null")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("Eligibility checked_at needs a timezone")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _eligibility_clauses(description):
    """Keep visible sentence/bullet boundaries; never read application fields.

    U.S. and similar abbreviations are protected only for boundary detection;
    returned evidence retains the employer's spelling and punctuation.
    """
    if not isinstance(description, str) or not description.strip() or len(description) > 150000:
        return []
    parser = TextOnly()
    parser.feed(unescape(description)[:150000])
    clauses, remaining = [], 60000
    for line in "".join(parser.parts).splitlines():
        line = clean_text(line, remaining + 1)
        if len(line) > remaining:
            return []  # An incomplete description cannot establish eligibility.
        remaining -= len(line)
        if not line:
            continue
        protected = re.sub(r"\b(?:U\.S\.(?:A\.)?|U\.K\.|E\.U\.|e\.g\.|i\.e\.)",
                           lambda m: m.group().replace(".", "\ue000"), line, flags=re.I)
        start = 0
        for boundary in re.finditer(r"[.!?;](?:\s+|$)", protected):
            clauses.append(line[start:boundary.end()].strip())
            start = boundary.end()
        if line[start:].strip():
            clauses.append(line[start:].strip())
    return clauses


def _eligibility_quote(clause, match, *, whole=False):
    """Quote a short exact clause without removing a material qualification."""
    quote = clause if whole else match.group().strip()
    # A short regex match must not omit a nearby exception or visa-type limit.
    qualifier = re.compile(r"\b(?:unless|except|only|subject to|depending on|"
                           r"case[- ]by[- ]case|may|might|could|H[- ]?1B1?|O[- ]?1|L[- ]?1|"
                           r"(?:existing|current) (?:visa|sponsorship)|transfers? only)\b", re.I)
    if any(not (match.start() <= item.start() and item.end() <= match.end())
           for item in qualifier.finditer(clause)):
        quote = clause
    if len(quote.split()) > MAX_ELIGIBILITY_QUOTE_WORDS:
        return None  # Abstain rather than publish a misleading truncated quote.
    return quote


def extract_eligibility(description, checked_at=None):
    """Extract explicit posting statements, not a candidate eligibility verdict.

    Called only with a freshly fetched role description by make_job. Empty or
    unavailable text has unknown fields and no observation date. Reading an
    employer's application questions, EEO text, or a bare ITAR mention does not
    establish sponsorship or a citizenship restriction.
    """
    clauses = _eligibility_clauses(description)
    if not clauses:
        return _unknown_eligibility()
    observed = _eligibility_timestamp(checked_at) if checked_at else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assessment = {"sponsorship": "not_stated", "access": "no_restriction_stated",
                  "work_authorization": "not_stated", "evidence": [],
                  "checked_at": observed, "method": "posting_text"}
    findings = {key: [] for key in ELIGIBILITY_VALUES}
    visa_type = r"(?:H[- ]?1B1?|O[- ]?1|L[- ]?1)"
    sponsor_noun = r"(?:(?:visa|immigration|employment(?:[- ]visa)?|work[- ](?:visa|permit)|H[- ]?1B1?|O[- ]?1|L[- ]?1)\s+)?sponsorship"
    sponsor_verb = r"sponsor(?:ing)?\s+(?:(?:employment|work)\s+)?(?:visas?|candidates?|applicants?|H[- ]?1B1?|O[- ]?1|L[- ]?1)"
    employer_name = r"(?:(?!(?:but|or|and|unless|except|not)\b)[A-Za-z0-9&.'’_-]{1,40}\s+){0,3}"
    no_sponsor = re.compile(
        r"\b(?:do(?:es)? not|don(?:'t|’t)|doesn(?:'t|’t)|will not|can(?:not|'t|’t)|won(?:'t|’t)|"
        r"(?:aren|isn)(?:'t|’t) able to|unable to|not able to|"
        r"not currently able to|not|no longer)\s+(?:currently\s+)?(?:be able to\s+)?"
        r"(?:(?:offer|provide|support|facilitate)\s+" + sponsor_noun + r"|" + sponsor_verb + r")|"
        r"\b" + sponsor_noun + r"\s+(?:(?:is|will be)\s+)?(?:not (?:available|offered|provided|supported)|unavailable)|"
        r"\bno\s+" + sponsor_noun + r"(?:\s+(?:is\s+)?(?:available|offered|provided|supported)|\s+for\s+" + visa_type + r"|[.!;]|$)|"
        r"\b(?:this (?:role|position)|candidates?|applicants?)\s+(?:is|are)\s+not\s+eligible\s+for\s+" + employer_name + sponsor_noun + r"|"
        r"\bmust\s+(?:not\s+require|be\s+(?:able|authori[sz]ed|eligible)\s+to\s+work[^!?;]{0,100}\bwithout)"
        r"\s+(?:(?:current|future|now|or|and|any|employer|requiring)\s+){0,5}" + sponsor_noun,
        re.I)
    conditional_sponsor = re.compile(
        r"\b(?:may|might|could|can)\s+(?:(?:be able to|be eligible to)\s+)?"
        r"(?:(?:offer|provide|support|consider)\s+" + sponsor_noun + r"|" + sponsor_verb + r")|"
        r"\b(?:consider(?:ing)?|open to)\s+(?:" + sponsor_noun + r"|" + sponsor_verb + r")|"
        r"\b" + sponsor_noun + r"\s+(?:(?:is|will be|may be|might be|could be)\s+)?"
        r"(?:considered|available on a case[- ]by[- ]case basis)|"
        r"\b" + sponsor_noun + r"\s+(?:may|might|could)\s+be\s+(?:available|offered|provided)|"
        r"\b(?:may|might)\s+be\s+eligible\s+for\s+" + sponsor_noun,
        re.I)
    offered_sponsor = re.compile(
        r"\b(?:offer(?:s)?|provide(?:s)?|support(?:s)?)\s+" + sponsor_noun + r"|"
        r"\b(?:we|the company)\s+(?:will\s+)?" + sponsor_verb + r"|"
        r"\b" + sponsor_noun + r"\s+(?:(?:is|will be)\s+)?(?:available|offered|provided|supported)|"
        r"\b(?:this (?:role|position)|candidates?|applicants?)\s+(?:is|are)\s+eligible\s+for\s+" + sponsor_noun,
        re.I)
    us = r"(?:U\.?S\.?|United States)"
    citizenship = r"(?:" + us + r"|Swedish|British|U\.?K\.?|EU|European Union)\s+(?:citizens?(?:hip)?|nationals?)"
    person = us + r"\s*[- ]\s*persons?"
    obligation = r"(?:must\s+(?:be|qualify as)|(?:is|are)\s+required\s+to\s+be|only\s+(?:consider|hire|accept)\s+(?:applicants?|candidates?)\s+(?:who\s+are|with))"
    person_required = re.compile(
        r"\b" + obligation + r"\s+(?:an?\s+)?[\"'‘’“”]?" + person + r"|"
        r"\b" + person + r"(?:\s+status)?\s+(?:(?:is|will be)\s+)?required|"
        r"\b" + person + r"s?\s+only\b", re.I)
    citizen_required = re.compile(
        r"\b" + obligation + r"\s+(?:an?\s+)?" + citizenship + r"|"
        r"\b" + citizenship + r"\s+(?:(?:is|will be)\s+)?required|"
        r"\bonly\s+" + citizenship + r"|"
        r"\brequires?\s+" + citizenship, re.I)
    clearance_required = re.compile(
        r"\b(?:must|(?:is|are|will be) required to)\s+(?:(?:currently|already)\s+)?"
        r"(?:hold|have|possess|obtain|maintain|be (?:able|eligible) to (?:obtain|maintain)|be eligible for)"
        r"\s+(?:an?\s+)?(?:(?:active|current|U\.?S\.?|United States|government|federal|secret|top[- ]secret)\s+){0,4}security clearance|"
        r"\b(?:active\s+)?(?:(?:U\.?S\.?|United States|government|federal|secret|top[- ]secret)\s+){0,3}"
        r"security clearance\s+(?:(?:is|will be)\s+)?required|"
        r"\b(?:ability|eligibility) to (?:obtain|maintain)\s+(?:an?\s+)?"
        r"(?:(?:U\.?S\.?|United States|government|federal)\s+)?security clearance\s+(?:is\s+)?required|"
        r"\b(?:active|current)\s+(?:secret|top[- ]secret|TS/SCI)\s+(?:clearance\s+)?(?:is\s+)?required", re.I)
    export_review = re.compile(
        r"\b(?:subject to|requires?|may require|contingent (?:on|upon)|dependent (?:on|upon))\s+"
        r"(?:(?:an?|the|obtaining|receiving|successful|required|necessary|applicable)\s+){0,4}"
        r"(?:export[- ]control\s+(?:review|approval|authorization)|(?:deemed[- ])?export\s+(?:licen[cs]e|authorization|approval))|"
        r"\b(?:eligible|eligibility|ability|able)\s+(?:to\s+(?:obtain|receive)|for)\s+(?:(?:an?|the|required|necessary)\s+){0,3}"
        r"export\s+(?:licen[cs]e|authorization|approval)|"
        r"\bexport\s+(?:licen[cs]e|authorization|approval)\s+(?:may be|is|will be)\s+required|"
        r"\b(?:must|will need to)\s+(?:obtain|receive)\s+(?:(?:an?|the|required|necessary)\s+){0,3}"
        r"export\s+(?:licen[cs]e|authorization|approval)", re.I)
    work_now = re.compile(
        r"\b(?:must|(?:are|is) required to)\s+(?:already|currently)\s+be\s+(?:legally\s+)?authori[sz]ed\s+to\s+work|"
        r"\bmust\s+be\s+(?:already|currently)\s+(?:legally\s+)?authori[sz]ed\s+to\s+work|"
        r"\bmust\s+(?:already\s+)?(?:have|hold|possess)\s+(?:current|existing)\s+(?:valid\s+)?(?:work\s+authori[sz]ation|authori[sz]ation\s+to\s+work|work\s+permit)|"
        r"\b(?:current|existing)\s+(?:work\s+authori[sz]ation|authori[sz]ation\s+to\s+work|work\s+permit)\s+(?:is\s+)?required",
        re.I)

    for clause in clauses:
        if "?" in clause or re.match(r"^(?:will you|do you|are you|would you|have you|please (?:indicate|select|confirm)|if yes)\b", clause, re.I):
            continue
        for label, pattern in (("not_offered", no_sponsor), ("case_by_case", conditional_sponsor), ("offered", offered_sponsor)):
            match = pattern.search(clause)
            if not match:
                continue
            exception = re.search(r"\b(?:unless|except|case[- ]by[- ]case|depending on|subject to)\b", clause, re.I)
            # A refusal limited to one visa type does not establish either a
            # blanket refusal or willingness to consider other visas.
            limited_negative = label == "not_offered" and (
                re.search(r"\b" + visa_type + r"\b", match.group(), re.I)
                or re.match(r"\s+for\s+" + visa_type + r"\b", clause[match.end():], re.I))
            uncertain_negative = label == "not_offered" and re.search(r"\b(?:may|might|could)\s*$", clause[:match.start()], re.I)
            value = "case_by_case" if exception else "unknown" if limited_negative or uncertain_negative else label
            findings["sponsorship"].append((value, _eligibility_quote(clause, match, whole=bool(exception or limited_negative))))
            break
        export_match = export_review.search(clause)
        person_match, citizen_match = person_required.search(clause), citizen_required.search(clause)
        clearance_match = clearance_required.search(clause)
        alternatives = re.search(r"\bor\s+(?:(?:otherwise|be|is|are|with|an?|the)\s+){0,3}(?:eligible|able|qualify|obtain|receive|export|authorization)", clause, re.I)
        protected = re.search(r"\b(?:permanent residents?|green[- ]card holders?|protected individuals?|asylees?|refugees?)\b", clause, re.I)
        if clearance_match:
            findings["access"].append(("citizenship_clearance", _eligibility_quote(clause, clearance_match)))
        elif (person_match or citizen_match) and not (export_match and alternatives):
            match = person_match or citizen_match
            value = "us_person" if person_match or protected else "citizenship_clearance"
            findings["access"].append((value, _eligibility_quote(clause, match, whole=bool(protected and not person_match))))
        elif export_match:
            findings["access"].append(("export_review", _eligibility_quote(clause, export_match, whole=bool(alternatives))))
        if match := work_now.search(clause):
            findings["work_authorization"].append(("required_now", _eligibility_quote(clause, match)))

    # Conflicting sponsorship clauses mean a qualified policy, never an
    # unconditional yes or no. Keep both short clauses when the quote budget fits.
    sponsor_values = {value for value, _quote in findings["sponsorship"]}
    if len(sponsor_values) > 1:
        findings["sponsorship"] = [("case_by_case", quote) for _value, quote in findings["sponsorship"]]
    rank = {"citizenship_clearance": 5, "us_person": 4, "export_review": 3,
            "not_offered": 5, "case_by_case": 3, "offered": 1, "required_now": 4, "unknown": 0}
    selected = {}
    for topic, items in findings.items():
        if not items:
            continue
        value = max(items, key=lambda item: rank[item[0]])[0]
        quotes = list(dict.fromkeys(quote for item_value, quote in items if item_value == value and quote))
        # Preserve both sides of a split sponsorship policy. A single short
        # statement suffices for an otherwise consistent assessment.
        if topic == "sponsorship" and len(sponsor_values) > 1:
            quote = " … ".join(quotes) if quotes and all(q for _v, q in items) else None
        else:
            quote = min(quotes, key=lambda q: len(q.split())) if quotes else None
        selected[topic] = (value, quote)
    remaining = MAX_ELIGIBILITY_QUOTE_WORDS
    for topic, (value, quote) in sorted(selected.items(), key=lambda item: -rank[item[1][0]]):
        if quote and len(quote.split()) <= remaining:
            assessment[topic] = value
            assessment["evidence"].append({"topic": topic, "text": quote})
            remaining -= len(quote.split())
        else:
            assessment[topic] = "unknown"  # Explicit but not safely expressible in a short excerpt.
    return assessment


def _normalize_eligibility(value, *, strict=False):
    """Whitelist stored/manual metadata without refreshing its observation date."""
    unknown = _unknown_eligibility()
    try:
        if not isinstance(value, dict):
            raise ValueError("Eligibility must be an object")
        if any(value.get(key) not in allowed for key, allowed in ELIGIBILITY_VALUES.items()):
            raise ValueError("Invalid eligibility label")
        if value.get("method") not in {"posting_text", "manual_review"}:
            raise ValueError("Invalid eligibility method")
        checked = _eligibility_timestamp(value.get("checked_at"))
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > 6:
            raise ValueError("Eligibility evidence must be a bounded array")
        excerpts = []
        for item in evidence:
            if not isinstance(item, dict) or item.get("topic") not in ELIGIBILITY_VALUES or not isinstance(item.get("text"), str):
                raise ValueError("Invalid eligibility evidence")
            text = clean_text(item["text"], 1000)
            if not text or len(text) > 600:
                raise ValueError("Eligibility evidence must be a short visible clause")
            excerpts.append({"topic": item["topic"], "text": text})
        if sum(len(item["text"].split()) for item in excerpts) > MAX_ELIGIBILITY_QUOTE_WORDS:
            raise ValueError("Eligibility evidence exceeds the 25-word quote budget")
        topics = {item["topic"] for item in excerpts}
        if any(value[key] not in ELIGIBILITY_DEFAULTS and key not in topics for key in ELIGIBILITY_VALUES):
            raise ValueError("Explicit eligibility labels require matching evidence")
        return {**{key: value[key] for key in ELIGIBILITY_VALUES}, "evidence": excerpts,
                "checked_at": checked, "method": value["method"]}
    except (ValueError, TypeError):
        if strict:
            raise
        return unknown


def _normalize_eligibility_review(value, *, strict=False):
    try:
        if not isinstance(value, dict):
            raise ValueError("Eligibility review must be an object")
        reviewed = _normalize_eligibility(value.get("eligibility"), strict=True)
        if reviewed["method"] != "manual_review":
            raise ValueError("Configured reviews require method manual_review")
        result = {"eligibility": reviewed}
        for key in ("fit_note", "compensation_note"):
            if key in value:
                note = value[key]
                if not isinstance(note, str) or len(note) > 800:
                    raise ValueError(f"{key} must be a short plain-text note")
                result[key] = clean_text(note, 800)
        return result
    except (ValueError, TypeError):
        if strict:
            raise
        return None


def _newer_eligibility(first, second):
    """An unread/undated description cannot overwrite a dated observation."""
    if not first or (second and (second.get("checked_at") or "") >= (first.get("checked_at") or "")):
        return second or _unknown_eligibility()
    return first


def _set_job_eligibility(job, posting, review=None):
    """Choose one dated observation, retain both sources, and flag conflicts."""
    posting = _normalize_eligibility(posting)
    job["eligibility_posting"] = posting
    job.pop("eligibility_conflict", None)
    if not review:
        job["eligibility"] = posting
        job.pop("eligibility_review", None)
        return
    review = _normalize_eligibility_review(review, strict=True)
    manual = review["eligibility"]
    job["eligibility_review"] = review
    for key in ("fit_note", "compensation_note"):
        if key in review:
            job[key] = review[key]
    if not posting.get("checked_at"):
        job["eligibility"] = manual
        return
    if (manual.get("checked_at") or "") > (posting.get("checked_at") or ""):
        omitted = [key for key in ELIGIBILITY_VALUES
                   if manual[key] in ELIGIBILITY_DEFAULTS and posting[key] not in ELIGIBILITY_DEFAULTS]
        if omitted:
            job["eligibility"] = posting
            if any(manual[key] not in ELIGIBILITY_DEFAULTS and manual[key] != posting[key] for key in ELIGIBILITY_VALUES):
                job["eligibility_conflict"] = "The review leaves earlier posting requirements unresolved; confirm both with recruiting."
        else:
            job["eligibility"] = manual
        return
    disagreements = [key for key in ELIGIBILITY_VALUES
                     if manual[key] not in ELIGIBILITY_DEFAULTS and posting[key] not in ELIGIBILITY_DEFAULTS
                     and manual[key] != posting[key]]
    new_restrictions = [key for key in ELIGIBILITY_VALUES
                        if posting[key] in ELIGIBILITY_RESTRICTIONS and posting[key] != manual[key]]
    manual_restrictions = [key for key in ELIGIBILITY_VALUES
                           if manual[key] in ELIGIBILITY_RESTRICTIONS and manual[key] != posting[key]]
    if new_restrictions:
        selected = posting
    elif manual_restrictions or (any(manual[key] not in ELIGIBILITY_DEFAULTS for key in ELIGIBILITY_VALUES)
                                and not any(posting[key] not in ELIGIBILITY_DEFAULTS for key in ELIGIBILITY_VALUES)):
        selected = manual
    else:
        selected = posting
    unresolved = any(posting[item["topic"]] == "unknown" and manual[item["topic"]] not in ELIGIBILITY_DEFAULTS
                     for item in posting["evidence"])
    if disagreements or unresolved or (new_restrictions and manual_restrictions):
        job["eligibility_conflict"] = "The posting and manual review contain different requirements; confirm both with recruiting."
    job["eligibility"] = selected


def _carry_eligibility(current, existing):
    posting = _newer_eligibility(existing.get("eligibility_posting"), current.get("eligibility_posting"))
    old_review, new_review = existing.get("eligibility_review"), current.get("eligibility_review")
    if old_review and (not new_review or (old_review["eligibility"].get("checked_at") or "") > (new_review["eligibility"].get("checked_at") or "")):
        new_review = old_review
    for key in ("fit_note", "compensation_note"):
        if key not in current and key in existing:
            current[key] = existing[key]
    _set_job_eligibility(current, posting, new_review)


def region_for(location):
    text = clean_text(location, 1000).lower()
    if re.search(r"\bremote\b|work from home", text):
        return "Remote"
    if re.search(r"\b(?:united states|usa|u\.s\.?a?\.?|us)\b", text):
        return "US"
    if re.search(r"\b(?:california|massachusetts|washington|texas|colorado|michigan|new york|"
                 r"virginia|north carolina|illinois|pennsylvania|arizona|oregon|florida|ohio|"
                 r"santa clara|mountain view|san francisco|san jose|irvine|boston|waltham|"
                 r"north reading|seattle|pittsburgh|denver|austin|foster city|menlo park|"
                 r"redwood city|sunnyvale|cambridge,? ma)\b", text):
        return "US"
    if re.search(r",\s*(?:ca|ma|wa|tx|co|ny|mi|va|nc|il|pa|az|or|fl|oh)(?:\b|,)", text):
        return "US"
    if re.search(r"\b(?:europe|germany|france|sweden|sverige|finland|norway|denmark|switzerland|austria|"
                 r"netherlands|belgium|ireland|united kingdom|uk|england|scotland|spain|italy|"
                 r"portugal|poland|czechia|czech republic|romania|hungary|greece|luxembourg|"
                 r"lithuania|latvia|estonia|iceland|zurich|zürich|berlin|munich|münchen|"
                 r"london|paris|stockholm|gothenburg|göteborg|lund|västerås|vasteras|linköping|linkoping|"
                 r"malmö|malmo|mölndal|molndal|helsinki|eindhoven|hamburg|bristol|"
                 r"leuven|copenhagen|cambridge,? (?:uk|united kingdom))\b", text):
        return "Europe"
    if re.search(r"\b(?:china|india|taiwan|singapore|japan|korea|israel|australia|canada|"
                 r"brazil|mexico|malaysia|vietnam|hong kong|shanghai|beijing|shenzhen|"
                 r"bangalore|bengaluru|hyderabad|taipei|tel aviv|toronto|vancouver)\b", text):
        return "Other"
    return "Unknown"


def classify_job(title, description, location="", employment_type=""):
    """Conservative relevance classification; returned text is fixed evidence labels."""
    title = clean_text(title, 260)
    body = clean_text(description, 30000)
    t, b = title.lower(), body.lower()
    if not t or re.search(r"\b(?:recruiter|recruiting|sales|marketing|accountant|payroll|"
                          r"human resources|legal counsel|business development|product manager|"
                          r"program manager|project manager|executive assistant)\b", t):
        return None
    if not re.search(r"engineer|scientist|research|intern|developer|student|algorithm|ph\.?d|postdoc|"
                     r"ingenjör|utvecklare|forskare|analytiker|doktor(?:and|erad)|technical coordinator", t):
        return None
    if re.search(r"(?:anomaly|defect|fault) locali[sz]ation|predictive maintenance", t):
        return None
    region = region_for(location)
    if region == "Other" or is_military_role(title, description):
        return None
    # Prefer role-specific sections over company introductions wherever available.
    boundary = re.search(r"what you(?:'|’)ll (?:do|get to do)|what you will|about the job|"
                         r"key job responsibilities|responsibilities|you will:", b)
    if boundary:
        b = b[boundary.start():]
    footer = re.search(r"why join (?:us|field)|equal opportunity employer|compensation and benefits", b)
    if footer:
        b = b[:footer.start()]
    text = t + ". " + b
    motion_context = re.search(
        r"\brobot(?:ic(?:s)?|s)?\b|autonom(?:ous|y)|automated driving|self[- ]driving|"
        r"\badas\b|\bvehicles?\b|navigation|positioning|\bpose\b|\bradar\b|"
        r"\blidar\b|inertial|odometry|\bslam\b", text)
    spatial_estimation = re.search(
        r"\bslam\b|odometry|pose estimation|(?:robot|vehicle|visual|inertial|lidar|camera)"
        r"[^.!?]{0,80}locali[sz]ation|locali[sz]ation[^.!?]{0,80}(?:robot|vehicle)", text)
    # Battery/grid state estimation and language localization share search
    # terms with pose estimation but are different assignments. Company-level
    # vehicle/robotics context alone must not promote them into this watch.
    if (re.search(r"state[- ]estimation|probabilistic estimation|sensor[- ]fusion|calibration", text) and not spatial_estimation
            and re.search(r"\bbatter(?:y|ies)\b|state[- ]of[- ](?:charge|health)|"
                          r"\b(?:soc|soh)\b|power[- ](?:grid|system)|electrical grid", text)):
        return None
    if (re.search(r"locali[sz]ation", t) and not spatial_estimation
            and re.search(r"translation|translators?|linguistic|internationali[sz]ation|"
                          r"language locali[sz]ation|locali[sz]ation (?:of |for )?(?:content|language)", text)):
        return None
    radio = r"\b(?:uwb|ultra[- ]wideband|wi[- ]?fi|bluetooth|gnss|gps|rf|radio|wireless)\b"
    position = r"locali[sz]ation|positioning|ranging|time[- ]of[- ]flight|angle[- ]of[- ]arrival|\brtt\b"
    radio_pair = re.search(radio + r"[^.!?]{0,200}(?:" + position + r")|(?:" + position + r")[^.!?]{0,200}" + radio, text)
    if re.search(position, t) and re.search(r"\buwb\b|ultra[- ]wideband|wi[- ]?fi|bluetooth", body, re.I):
        radio_pair = True
    isac = re.search(r"\bisac\b|integrated sensing and communication|joint sensing and communication", text)
    slam_title = re.search(r"\bslam\b|locali[sz]ation|odometry|sensor[- ]fusion|state[- ]estimation|calibration.*mapping", t)
    if re.search(r"sensor[- ]fusion|state[- ]estimation", t) and not motion_context:
        slam_title = None
    calibration_work = re.search(
        r"(?:sensor|camera|lidar|imu|inertial)[^.!?]{0,60}calibration|"
        r"calibration[^.!?]{0,60}(?:sensor|camera|lidar|imu|inertial)|"
        r"extrinsic calibration|sensor data alignment", text)
    radar_title = re.search(r"\bradar\b", t)
    category = None
    reasons = []
    if radio_pair and not (radar_title and not re.search(position, t)):
        category = "radio-localization"
        reasons = ["Role text connects radio/RF technology with localization or ranging"]
    elif isac:
        category = "wireless-isac"
        reasons = ["Integrated sensing and communications (ISAC) is explicitly mentioned"]
    elif radar_title:
        category = "radar-sensing"
        reasons = ["Radar is named in the role's title"]
    elif slam_title:
        category = "robotics-localization"
        reasons = ["Role title names localization, SLAM, sensor fusion, or state estimation"]
    elif (re.search(r"\bpositioning\b", t)
          and re.search(r"\b(?:autonomy|autonomous|robotics?|robotic|navigation)\b", t)
          and re.search(r"\bkalman\b|inertial (?:measurement|navigation)|\bodometry\b|pose estimation", b)):
        category = "robotics-localization"
        reasons = ["Autonomy or robotics positioning role describes filtering or inertial navigation"]
    elif radar_title or (re.search(r"perception|sensing|sensor|signal processing|calibration|autonomous|robotics|robotic", t)
                         and re.search(r"\bradar\b", b)):
        category = "radar-sensing"
        reasons = ["Radar is named in the role's title or technical description"]
    elif motion_context and calibration_work and re.search(r"calibration|software|algorithm|research|perception|robot", t):
        category = "robotics-localization"
        reasons = ["Role describes multi-sensor calibration or alignment for robotics"]
    elif motion_context and re.search(r"software|algorithm|research|scientist|robot", t) and re.search(
            r"state[- ]estimation|geometric optimi[sz]ation|probabilistic estimation|bundle adjustment|factor[- ]graph", b):
        category = "robotics-localization"
        reasons = ["Role describes geometric or probabilistic estimation for robotic motion"]
    elif (re.search(r"robotics|robotic|autonom(?:ous|y)|perception|research|mapping|tracking|computer vision|"
                   r"calibration|analytiker|sensorfusion|signalbehandling", t)
          or (re.search(r"software.*(?:engineer|developer)|(?:algorithm|development) engineer|scientist", t)
              and re.search(r"\brobot(?:ic(?:s)?|s)?\b", b))
          or (re.search(r"(?:deep|machine) learning engineer", t)
              and re.search(r"multi[- ]modal perception", b) and re.search(r"online mapping", b))) and re.search(
            r"\bslam\b|visual[- ]inertial|lidar[- ]inertial|probabilistic state estimation|"
            r"sensor fusion|robot locali[sz]ation|multi[- ](?:target|object) tracking|pose estimation|"
            r"visual odometry|lidar odometry|målföljning|sensorfusion|online mapping|"
            r"sensor calibration|multi[- ]sensor calibration|calibration algorithms|"
            r"sensor data alignment", b):
        category = "robotics-localization"
        reasons = ["Technical description mentions robotic estimation, tracking, or SLAM"]
        if calibration_work or re.search(r"calibration algorithms|sensor data alignment", b):
            if not motion_context:
                return None
            reasons = ["Role describes multi-sensor calibration or alignment for robotics"]
    if category is None:
        return None
    kind = "Internship" if re.search(r"intern|co[- ]?op|placement|working student", t + " " + str(employment_type).lower()) else (
        "Full-time" if re.search(r"full[-_ ]?time|regular|heltid", str(employment_type), re.I) else "Unknown")
    requirements = []
    for pattern, label in [
        (r"\bpython\b", "Python mentioned"), (r"\bc\+\+", "C++ mentioned"),
        (r"\bcuda\b|\bcupy\b", "GPU/CUDA experience mentioned"),
        (r"pytorch|tensorflow|\bjax\b", "Machine-learning frameworks mentioned"),
        (r"\bros\b|\bros2\b", "ROS experience mentioned"),
        (r"ph\.?\s?d|doctoral|graduate (?:student|program)|doktor(?:and|erad|sexamen)|disputerad", "PhD/graduate study mentioned"),
        (r"bayesian|kalman|factor graph|belief propagation", "Bayesian estimation/filtering mentioned"),
        (r"\b(?:uwb|ultra[- ]wideband)\b", "UWB mentioned"),
        (r"\bradar\b", "Radar mentioned"),
        (r"security clearance|u\.?s\.? citizen|united states citizen|us person|u\.s\. person|"
         r"säkerhetsprövning|säkerhetsklass|medborgarskap", "Citizenship/clearance conditions mentioned"),
        (r"\bmust\s+(?:have\s+current\s+(?:work\s+)?authori[sz]ation\s+to\s+work|"
         r"be\s+currently\s+authori[sz]ed\s+to\s+work)\s+in\s+(?:the\s+)?(?:united states|u\.?s\.?a?\.?)\b",
         "Current US work authorization required"),
        (r"(?:not|no|unable to|cannot|can't|do not)\s+(?:offer\s+|provide\s+|support\s+)?(?:visa\s+)?sponsor", "Sponsorship restriction mentioned"),
    ]:
        if re.search(pattern, text, re.I):
            requirements.append(label)
    senior = bool(re.search(r"\b(?:senior|sr\.?|staff|principal|lead|manager|director|distinguished|experienced|erfaren)\b", t))
    if senior:
        requirements.append("Senior/experienced role")
    score = {"radio-localization": 100, "wireless-isac": 90, "radar-sensing": 85, "robotics-localization": 75}[category]
    score += 20 if kind == "Internship" else 0
    score -= 20 if senior else 0
    summaries = {
        "radio-localization": "Role text links radio or RF measurements with positioning, ranging, or localization.",
        "radar-sensing": "Role text includes radar sensing, signal processing, or radar-based perception.",
        "robotics-localization": "Role text includes robotic localization, mapping, tracking, or sensor fusion.",
        "wireless-isac": "Role text connects wireless systems with integrated sensing and communications.",
    }
    return {"category": category, "region": region, "employment_type": kind,
            "match_reasons": reasons, "summary": summaries[category],
            "requirements": requirements, "score": score}


def canonical_url(url):
    p = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"source", "sourceid", "ref", "gh_src"}]
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/", urlencode(query), p.fragment))


def stable_job_id(source_id, native_id, url):
    native = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(native_id or "")).strip("-").lower()[:130]
    if not native:
        native = hashlib.sha256(canonical_url(url).encode()).hexdigest()[:20]
    return f"{source_id}:{native}"


def requisition_from_url(url):
    m = re.search(r"_(JR-?\d+|R-\d+|R\d{4,}|REQ_\d+)(?:\b|/)", url, re.I)
    return m.group(1) if m else ""


def make_job(source, native_id, title, url, location="", description="", employment_type="",
             status="listed", requisition_id="", posted_at=None, department=""):
    url = safe_url(url, source["allowed_hosts"])
    title, location = clean_text(title, 260), clean_text(location, 260)
    identity = stable_job_id(source["id"], native_id or requisition_id, url)
    if (str(source["company"]).strip().casefold() in EXCLUDED_DEFENCE_COMPANIES
            or is_military_role(title, description, department)):
        return None
    if (source.get("civilian_review_required")
            and identity not in set(source.get("civilian_verified_job_ids", []))):
        return None
    # Dedicated country sources use explicit provider location data, never a
    # mention in the description or an unspecified remote-work arrangement.
    if source.get("country_filter") == "SE" and not re.search(r"\b(?:sweden|sverige)\b", location, re.I):
        return None
    if source.get("exclude_student_theses") and re.search(
            r"\b(?:master(?:['’]s)?|bachelor(?:['’]s)?|msc|bsc|m\.sc\.?|b\.sc\.?)\b.*\bthes[ie]s\b|"
            r"\b(?:examensarbete|exjobb)\b", title, re.I):
        return None
    matched = classify_job(title, description, location, employment_type)
    if matched is None:
        return None
    job = {"id": stable_job_id(source["id"], native_id or requisition_id, url),
           "source_id": source["id"], "company": clean_text(source["company"], 120),
           "title": title, "url": url, "location": location, **matched,
           "status": status, "pinned": False,
           "eligibility": extract_eligibility(description, source.get("_now"))}
    if requisition_id:
        job["requisition_id"] = clean_text(requisition_id, 100)
    if posted_at and re.match(r"^\d{4}-\d{2}-\d{2}", str(posted_at)):
        job["posted_at"] = str(posted_at)[:25]
    return job


def brief_error(exc):
    if isinstance(exc, HTTPError):
        return f"Employer HTTP {exc.code}; availability not established"
    if isinstance(exc, json.JSONDecodeError):
        return "Employer returned invalid JSON or an access challenge"
    return clean_text(str(exc), 200) or type(exc).__name__


def _rotation_week(source):
    """Advance discovery windows once per UTC Monday, including manual reruns."""
    check_time = source.get("_now") or datetime.now(timezone.utc).isoformat()
    moment = datetime.fromisoformat(check_time.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (moment.astimezone(timezone.utc).date().toordinal() - 1) // 7


def _workday(source, client):
    result = SourceResult(source["id"], "ok", [], complete=False)
    errors, candidates, seen = [], {}, set()
    queries = source.get("queries", [""])
    page_size = source.get("page_size", 20)
    all_complete = queries == [""]
    detail_root = source["base_url"].rstrip("/") + f"/wday/cxs/{source['tenant']}/{source['site']}"
    known_order = {}
    for path in source.get("seed_paths", []):
        candidates[path] = (0, "")
    for job in source.get("_known_jobs", []):
        path = urlsplit(job.get("url", "")).path
        if "/job/" in path and urlsplit(job.get("url", "")).hostname == urlsplit(source["base_url"]).hostname:
            path = path[path.index("/job/"):]
            candidates.setdefault(path, (1, job.get("title", "")))
            known_order[path] = job.get("last_detail_attempt") or job.get("last_verified") or job.get("last_seen", "")
    for query in queries:
        offset, expected, query_complete = 0, None, False
        for _page in range(source.get("max_pages_per_query", 2)):
            try:
                data = client.post_json(source["search_url"], {"appliedFacets": source.get("applied_facets", {}), "limit": page_size,
                                                              "offset": offset, "searchText": query})
                if not isinstance(data, dict) or not isinstance(data.get("jobPostings"), list) or not isinstance(data.get("total"), int):
                    raise FetchError("Workday search shape changed")
                total, entries = data["total"], data["jobPostings"]
                if expected is not None and total != expected:
                    raise FetchError("Workday results changed during pagination")
                expected = total
                if not entries and offset < total:
                    raise FetchError("Workday pagination returned an incomplete page")
                for entry in entries:
                    path, title = entry.get("externalPath", ""), entry.get("title", "")
                    if not isinstance(path, str) or not path.startswith("/job/") or not title:
                        raise FetchError("Workday job entry shape changed")
                    safe_url(detail_root + path, source["allowed_hosts"])
                    req = requisition_from_url(path)
                    if req:
                        seen.add(stable_job_id(source["id"], req, detail_root + path))
                    # Query matches often come from company boilerplate. Spend
                    # scarce detail requests on relevant titles and geography.
                    if region_for(entry.get("locationsText", "")) == "Other":
                        continue
                    core_title = bool(re.search(r"\b(?:uwb|isac|rf|radio|radar|slam)\b|"
                                                r"locali[sz]ation|positioning|sensing|sensor[- ]fusion|"
                                                r"state[- ]estimation|sensor[- ]calibration|"
                                                r"calibration.*(?:camera|lidar|imu)|(?:camera|lidar|imu).*calibration", title, re.I))
                    adjacent_title = bool(re.search(r"robotic|autonomous|perception|mapping|wireless", title, re.I))
                    intern = bool(re.search(r"intern|student", title, re.I))
                    priority = (2 if core_title else 4 if adjacent_title else 6) + (0 if intern else 1)
                    candidates.setdefault(path, (priority, title))
                offset += len(entries)
                if offset >= total:
                    query_complete = True
                    break
            except Exception as exc:
                errors.append(brief_error(exc))
                break
        if not query_complete:
            all_complete = False
            errors.append(f"Query {query or '(all)'} was only partially checked")
    ordered = sorted(candidates.items(), key=lambda item: (item[1][0], item[1][1].casefold(), item[0]))
    limit = source.get("max_details", 12)
    if len(ordered) > limit:
        all_complete = False
        errors.append("Detail-check limit reached; additional matches remain unchecked")
    # Reserve discovery capacity rather than repeatedly refreshing the same first
    # twelve jobs. Previously seen records rotate by oldest attempted check.
    seeds = [item for item in ordered if item[1][0] == 0]
    known = sorted((item for item in ordered if item[1][0] == 1), key=lambda item: (known_order.get(item[0], ""), item[0]))
    fresh = [item for item in ordered if item[1][0] >= 2]
    slots = max(0, limit - len(seeds))
    known_slots = min(len(known), slots // 2 if fresh else slots)
    fresh_slots = slots - known_slots
    # Irrelevant details are not retained as jobs. Rotate the unpersisted
    # candidates too, so they cannot permanently consume the same detail slots.
    if fresh and len(fresh) > fresh_slots and fresh_slots > 0:
        offset = (_rotation_week(source) * fresh_slots) % len(fresh)
        fresh = fresh[offset:] + fresh[:offset]
    selected = (seeds + known[:known_slots] + fresh[:fresh_slots])[:limit]
    if len(selected) < limit:
        selected.extend(known[known_slots:known_slots + limit - len(selected)])
    for path, _metadata in selected:
        requested_req = requisition_from_url(path)
        if requested_req:
            result.attempted_ids.append(stable_job_id(source["id"], requested_req, detail_root + path))
        try:
            data = client.get_json(detail_root + path)
            if not isinstance(data, dict) or not isinstance(data.get("jobPostingInfo"), dict):
                raise FetchError("Workday detail shape changed")
            d = data["jobPostingInfo"]
            if not d.get("title") or not d.get("jobReqId"):
                raise FetchError("Workday detail lacks title or requisition")
            url = d.get("externalUrl") or source["base_url"].rstrip("/") + "/" + source["site"] + path
            job_id = stable_job_id(source["id"], d["jobReqId"], url)
            if job_id not in result.attempted_ids:
                result.attempted_ids.append(job_id)
            seen.add(job_id)
            if d.get("posted") is False and d.get("canApply") is False:
                result.closed_ids.append(job_id)
                continue
            locations = [d.get("location", "")]
            locations.extend(x if isinstance(x, str) else x.get("descriptor", "") for x in d.get("additionalLocations", []))
            country = d.get("country") or {}
            if isinstance(country, dict) and country.get("descriptor"):
                locations.append(country["descriptor"])
            job = make_job(source, d["jobReqId"], d["title"], url, " / ".join(filter(None, locations)),
                           d.get("jobDescription", ""), d.get("timeType", ""),
                           "open" if d.get("canApply") is True and d.get("posted") is True else "listed",
                           d["jobReqId"], d.get("startDate"),
                           department=" / ".join(str(d.get(key) or "") for key in (
                               "department", "jobFamily", "jobFamilyGroup", "businessUnit",
                               "supervisoryOrganization")))
            if job:
                result.records.append(job)
            else:
                result.rejected_ids.append(job_id)
        except Exception as exc:
            errors.append(brief_error(exc))
    result.complete = all_complete and not errors
    result.seen_ids = sorted(seen)
    if errors:
        result.status = "partial" if result.records or seen else "error"
        result.error = "; ".join(dict.fromkeys(errors))[:400]
    return result


def _amazon_location(entry):
    """Use this posting's location fields, including Amazon's ISO country codes."""
    countries = {
        "US": "United States", "USA": "United States", "CH": "Switzerland", "CHE": "Switzerland",
        "DE": "Germany", "DEU": "Germany", "GB": "United Kingdom", "GBR": "United Kingdom",
        "FR": "France", "FRA": "France", "SE": "Sweden", "SWE": "Sweden",
        "NL": "Netherlands", "NLD": "Netherlands", "AT": "Austria", "AUT": "Austria",
        "BE": "Belgium", "BEL": "Belgium", "IE": "Ireland", "IRL": "Ireland",
        "IT": "Italy", "ITA": "Italy", "ES": "Spain", "ESP": "Spain",
        "FI": "Finland", "FIN": "Finland", "NO": "Norway", "NOR": "Norway",
        "DK": "Denmark", "DNK": "Denmark", "PL": "Poland", "POL": "Poland",
        "PT": "Portugal", "PRT": "Portugal", "CZ": "Czechia", "CZE": "Czechia",
        "RO": "Romania", "ROU": "Romania", "LU": "Luxembourg", "LUX": "Luxembourg",
        "CA": "Canada", "CAN": "Canada", "IN": "India", "IND": "India",
        "CN": "China", "CHN": "China", "JP": "Japan", "JPN": "Japan",
        "SG": "Singapore", "SGP": "Singapore", "AU": "Australia", "AUS": "Australia",
        "IL": "Israel", "ISR": "Israel", "TW": "Taiwan", "TWN": "Taiwan",
        "BR": "Brazil", "BRA": "Brazil", "MX": "Mexico", "MEX": "Mexico",
    }
    def display_location(label, country, code):
        label = clean_text(label, 180)
        if country and code:
            label = re.sub(r"\b" + re.escape(code) + r"\b", str(country), label)
        if country and str(country).casefold() not in label.casefold():
            label = ", ".join(x for x in (label, str(country)) if x)
        return label

    locations = []
    for raw in entry.get("locations") or []:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                locations.append(clean_text(raw, 180))
                continue
        if isinstance(raw, dict):
            label = raw.get("normalizedLocation") or raw.get("locationNonStemming") or raw.get("location") or raw.get("city") or ""
            code = str(raw.get("countryIso3a") or raw.get("normalizedCountryCode") or raw.get("countryIso2a") or "").upper()
            country = raw.get("normalizedCountryName") or countries.get(code, "")
            locations.append(display_location(label, country, code))
    if not locations:
        label = entry.get("normalized_location") or entry.get("location") or ""
        code = str(entry.get("country_code") or "").upper()
        country = countries.get(code, "")
        locations.append(display_location(label, country, code))
    return " / ".join(dict.fromkeys(clean_text(x, 200) for x in locations if x))


def _amazon_robotics_context(entry, description):
    """Exclude Amazon-wide language localization and unrelated software roles."""
    team = entry.get("team") or {}
    label = team.get("label", "") if isinstance(team, dict) else ""
    text = clean_text(str(entry.get("title", "")) + " " + description, 35000).lower()
    robotics = "robotics" in str(label).lower() or re.search(r"\brobot(?:ic(?:s)?|s)?\b", text)
    physical_estimation = re.search(
        r"\bslam\b|\bvio\b|visual[- ]inertial|lidar[- ]inertial|sensor fusion|state estimation|"
        r"multi[- ](?:target|object) tracking|robot locali[sz]ation|"
        r"locali[sz]ation[^.!?]{0,120}(?:mapping|navigation|odometry|ranging)|"
        r"(?:mapping|navigation|odometry|ranging)[^.!?]{0,120}locali[sz]ation|"
        r"\bradar\b|\buwb\b|ultra[- ]wideband|\bisac\b|"
        r"(?:radio|wireless|\brf\b)[^.!?]{0,120}(?:positioning|ranging|locali[sz]ation)", text)
    return bool(robotics and physical_estimation)


def _amazon(source, client):
    """Read Amazon's public, paginated search feed; it is never a full board."""
    result = SourceResult(source["id"], "ok", [], complete=False)
    errors, seen = [], set()
    valid_entries, empty_queries = 0, 0
    page_size = source.get("page_size", 50)
    api = urlsplit(source["api_url"])
    for query in source.get("queries", ["SLAM", "localization"]):
        offset, expected, query_complete = 0, None, False
        query_ids = set()
        for _page in range(source.get("max_pages_per_query", 3)):
            try:
                params = dict(parse_qsl(api.query, keep_blank_values=True))
                params.update(base_query=query, offset=offset, result_limit=page_size, sort="recent")
                url = urlunsplit((api.scheme, api.netloc, api.path, urlencode(params), ""))
                data = client.get_json(url)
                if (not isinstance(data, dict) or data.get("error")
                        or not isinstance(data.get("jobs"), list)
                        or type(data.get("hits")) is not int or data["hits"] < 0):
                    raise FetchError("Amazon search shape changed or returned an error")
                total, entries = data["hits"], data["jobs"]
                if expected is not None and total != expected:
                    raise FetchError("Amazon results changed during pagination")
                expected = total
                if (not entries and offset < total) or offset + len(entries) > total:
                    raise FetchError("Amazon pagination returned an inconsistent page")
                page_ids = set()
                for entry in entries:
                    try:
                        if not isinstance(entry, dict):
                            raise FetchError("Amazon job entry shape changed")
                        native, title, path = str(entry.get("id_icims") or ""), entry.get("title"), entry.get("job_path")
                        if (not re.fullmatch(r"\d{1,12}", native) or not isinstance(title, str) or not title.strip()
                                or not isinstance(path, str)
                                or not re.match(r"^/(?:[a-z-]+/)?jobs/" + re.escape(native) + r"(?:/|$)", path)):
                            raise FetchError("Amazon job lacks a matching ID, title, or official job path")
                        target = safe_url(urljoin(source["api_url"], path), source["allowed_hosts"])
                        fields = [entry.get(key) or "" for key in ("description", "basic_qualifications", "preferred_qualifications")]
                        if not fields[0] or any(not isinstance(x, str) for x in fields):
                            raise FetchError("Amazon full job description is missing or changed")
                        # Wider-team boilerplate can mention SLAM even when this
                        # role only builds generic infrastructure. Keep this
                        # role's duties and its separate qualifications instead.
                        role_text = re.split(r"\babout the team\b|\babout amazon\b",
                                             clean_text(fields[0], 30000), maxsplit=1, flags=re.I)[0]
                        description = "\n".join([role_text, *fields[1:]])
                        page_ids.add(native)
                        valid_entries += 1
                        identity = stable_job_id(source["id"], native, target)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        if not _amazon_robotics_context(entry, description):
                            result.rejected_ids.append(identity)
                            continue
                        employment = entry.get("job_schedule_type") or ""
                        if entry.get("is_intern") is True or str(entry.get("is_intern")).lower() in {"true", "1"}:
                            employment = "Internship"
                        posted = entry.get("posted_date")
                        if isinstance(posted, str) and not re.match(r"^\d{4}-\d{2}-\d{2}", posted):
                            try:
                                posted = datetime.strptime(posted, "%B %d, %Y").strftime("%Y-%m-%d")
                            except ValueError:
                                posted = None
                        job = make_job(source, native, title, target, _amazon_location(entry), description,
                                       employment, "listed", native, posted,
                                       department=str(entry.get("team") or ""))
                        if job:
                            result.records.append(job)
                        else:
                            result.rejected_ids.append(identity)
                    except Exception as exc:
                        errors.append(brief_error(exc))
                if entries and page_ids and page_ids <= query_ids:
                    raise FetchError("Amazon pagination repeated a previously returned page")
                query_ids.update(page_ids)
                offset += len(entries)
                if offset >= total:
                    query_complete = True
                    empty_queries += total == 0
                    break
            except Exception as exc:
                errors.append(brief_error(exc))
                break
        if not query_complete:
            errors.append(f"Query {query} was only partially checked")
    result.seen_ids = sorted(seen)
    if errors:
        result.status = "partial" if valid_entries or empty_queries else "error"
        result.error = "; ".join(dict.fromkeys(errors))[:400]
    return result


def _ericsson_endpoint(base, parameters, source):
    parts = urlsplit(safe_url(base, source["allowed_hosts"]))
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(parameters)
    return safe_url(urlunsplit((parts.scheme, parts.netloc, parts.path,
                               urlencode(query), "")), source["allowed_hosts"])


def _ericsson_payload(response, label):
    if not isinstance(response, dict) or response.get("status") != 200:
        raise FetchError(f"Ericsson {label} did not return a successful JSON response")
    if not isinstance(response.get("data"), dict):
        raise FetchError(f"Ericsson {label} data object is missing")
    error = response.get("error")
    if isinstance(error, dict) and (error.get("message") or error.get("body")):
        raise FetchError(f"Ericsson {label} returned an error payload")
    if error and not isinstance(error, dict):
        raise FetchError(f"Ericsson {label} returned an error payload")
    if isinstance(response.get("metadata"), dict) and response["metadata"].get("isFallback"):
        raise FetchError(f"Ericsson {label} returned fallback data")
    return response["data"]


def _ericsson_position_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise FetchError("Ericsson position id is missing or malformed")
    value = str(value)
    if not re.fullmatch(r"[0-9]{1,24}", value):
        raise FetchError("Ericsson position id is missing or malformed")
    return value


def _ericsson_native_id(entry, position_id):
    value = entry.get("displayJobId") or entry.get("atsJobId") or position_id
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise FetchError("Ericsson requisition id is malformed")
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", value):
        raise FetchError("Ericsson requisition id is malformed")
    return value


def _ericsson_location(entry):
    locations = entry.get("locations")
    if locations is not None:
        if not isinstance(locations, list) or any(not isinstance(x, str) for x in locations):
            raise FetchError("Ericsson locations array is malformed")
        location = " / ".join(x.strip() for x in locations if x.strip())
    else:
        location = entry.get("location", "")
        if not isinstance(location, str):
            raise FetchError("Ericsson location is malformed")
    if not clean_text(location, 500):
        raise FetchError("Ericsson job location is missing")
    return location


def _ericsson_job_url(source, position_id, entry):
    value = entry.get("publicUrl") or entry.get("positionUrl") or f"/careers/job/{position_id}"
    if not isinstance(value, str):
        raise FetchError("Ericsson job URL is malformed")
    url = safe_url(urljoin(source["api_url"], value), source["allowed_hosts"])
    parts = urlsplit(url)
    if (parts.hostname != urlsplit(source["api_url"]).hostname
            or not re.match(r"^/careers/job/" + re.escape(position_id) + r"(?:$|[-/])", parts.path)):
        raise FetchError("Ericsson job URL does not identify the requested posting")
    return url


def _ericsson(source, client):
    """Discover bounded search results and verify descriptions individually.

    Even completed keyword searches are not the complete employer inventory;
    this adapter never uses absence to close an earlier posting. HTTP errors,
    malformed data, and expired detail URLs preserve previous observations.
    """
    result = SourceResult(source["id"], "ok", [], complete=False)
    queries = source.get("queries", ["ISAC", "sensing", "radar", "positioning"])
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 10
            or any(not isinstance(q, str) or len(q) > 80 for q in queries)):
        raise FetchError("Ericsson queries must be a bounded string array")
    max_pages = min(max(int(source.get("max_pages_per_query", 2)), 1), 5)
    max_details = min(max(int(source.get("max_details", 12)), 1), 30)
    domain = source.get("domain", "ericsson.com")
    location = source.get("search_location", "Sweden")
    if not isinstance(domain, str) or not re.fullmatch(r"[a-z0-9.-]{1,100}", domain):
        raise FetchError("Ericsson domain setting is invalid")
    if not isinstance(location, str) or len(location) > 120:
        raise FetchError("Ericsson search location setting is invalid")
    safe_url(source["api_url"], source["allowed_hosts"])
    safe_url(source["detail_url"], source["allowed_hosts"])
    errors, candidates, seen, known_order = [], {}, set(), {}

    for value in source.get("seed_ids", [])[:max_details]:
        pid = _ericsson_position_id(value)
        candidates[pid] = {"id": pid, "_priority": 0}
    for old in source.get("_known_jobs", []):
        parts = urlsplit(old.get("url", ""))
        match = re.match(r"^/careers/job/([0-9]{1,24})(?:$|[-/])", parts.path)
        if parts.hostname != urlsplit(source["api_url"]).hostname or not match:
            continue
        pid = match.group(1)
        candidates.setdefault(pid, {"id": pid, "name": old.get("title", ""),
                                    "displayJobId": old.get("requisition_id", ""), "_priority": 1})
        known_order[pid] = old.get("last_detail_attempt") or old.get("last_verified") or old.get("last_seen", "")

    for query in queries:
        offset, expected, query_done, query_ids = 0, None, False, set()
        for _page in range(max_pages):
            try:
                endpoint = _ericsson_endpoint(source["api_url"], {
                    "domain": domain, "query": query, "location": location,
                    "start": offset, "sort_by": "relevance"}, source)
                data = _ericsson_payload(client.get_json(endpoint), "search")
                count, entries = data.get("count"), data.get("positions")
                if (not isinstance(count, int) or isinstance(count, bool) or count < 0
                        or not isinstance(entries, list) or len(entries) > 100):
                    raise FetchError("Ericsson search count or positions array is malformed")
                if expected is not None and count != expected:
                    raise FetchError("Ericsson results changed during pagination")
                expected = count
                if (not entries and offset < count) or offset + len(entries) > count:
                    raise FetchError("Ericsson pagination returned an inconsistent page")
                for entry in entries:
                    if not isinstance(entry, dict):
                        raise FetchError("Ericsson search job entry is malformed")
                    pid = _ericsson_position_id(entry.get("id"))
                    if pid in query_ids:
                        raise FetchError("Ericsson pagination repeated a posting")
                    query_ids.add(pid)
                    title = entry.get("name")
                    if not isinstance(title, str) or not clean_text(title, 260):
                        raise FetchError("Ericsson search job title is missing")
                    _ericsson_location(entry)
                    job_url = _ericsson_job_url(source, pid, entry)
                    native = _ericsson_native_id(entry, pid)
                    seen.add(stable_job_id(source["id"], native, job_url))
                    priority = 2 if re.search(r"\bisac\b|\bradar\b|\bslam\b|locali[sz]ation|"
                                              r"positioning|sensor fusion|sensing", title, re.I) else 4
                    if pid in candidates:
                        priority = candidates[pid]["_priority"]
                    candidates[pid] = dict(entry, _priority=priority)
                offset += len(entries)
                if offset >= count:
                    query_done = True
                    break
            except Exception as exc:
                errors.append(brief_error(exc))
                break
        if not query_done:
            errors.append(f"Query {query or '(all)'} was only partially checked")

    ordered = sorted(candidates.items(), key=lambda item: (
        item[1]["_priority"], str(item[1].get("name", "")).casefold(), item[0]))
    seeds = [x for x in ordered if x[1]["_priority"] == 0]
    known = sorted((x for x in ordered if x[1]["_priority"] == 1),
                   key=lambda x: (known_order.get(x[0], ""), x[0]))
    fresh = [x for x in ordered if x[1]["_priority"] >= 2]
    slots = max(0, max_details - len(seeds))
    known_slots = min(len(known), slots // 2 if fresh else slots)
    fresh_slots = slots - known_slots
    if fresh_slots and len(fresh) > fresh_slots:
        offset = (_rotation_week(source) * fresh_slots) % len(fresh)
        fresh = fresh[offset:] + fresh[:offset]
    selected = (seeds + known[:known_slots] + fresh[:fresh_slots])[:max_details]
    if len(selected) < max_details:
        selected.extend(known[known_slots:known_slots + max_details - len(selected)])
    if len(ordered) > len(selected):
        errors.append("Detail-check limit reached; additional matches remain unchecked")

    for pid, entry in selected:
        try:
            old_native = _ericsson_native_id(entry, pid)
            old_url = _ericsson_job_url(source, pid, entry)
            old_identity = stable_job_id(source["id"], old_native, old_url)
            result.attempted_ids.append(old_identity)
            endpoint = _ericsson_endpoint(source["detail_url"], {
                "position_id": pid, "domain": domain, "hl": "en"}, source)
            detail = _ericsson_payload(client.get_json(endpoint), "detail")
            if _ericsson_position_id(detail.get("id")) != pid:
                raise FetchError("Ericsson returned details for a different posting")
            title, description = detail.get("name"), detail.get("jobDescription")
            if not isinstance(title, str) or not clean_text(title, 260):
                raise FetchError("Ericsson detail job title is missing")
            if (not isinstance(description, str) or len(clean_text(description, 30000)) < 80
                    or CHALLENGE.search(description)):
                raise FetchError("Ericsson full job description is missing or blocked")
            job_location = _ericsson_location(detail)
            url = _ericsson_job_url(source, pid, detail)
            native = _ericsson_native_id(detail, pid)
            if native == pid and old_native != pid:
                native = old_native
            elif old_native != pid and native != old_native:
                raise FetchError("Ericsson requisition identity changed between search and detail")
            identity = stable_job_id(source["id"], native, url)
            if identity not in result.attempted_ids:
                result.attempted_ids.append(identity)
            seen.add(identity)
            status = "listed"
            actions = detail.get("positionUserActions") or {}
            apply_action = actions.get("applyAction") if isinstance(actions, dict) else None
            if isinstance(apply_action, dict) and apply_action.get("status") == "link_off":
                # Explicit live application link supplied by the official detail
                # payload. Validate it, but never request or submit to it.
                try:
                    safe_url(apply_action.get("applyUrl", ""), source["allowed_hosts"])
                except ValueError:
                    errors.append("Ericsson application destination is not in the configured allowlist")
                else:
                    status = "open"
            posted = None
            if isinstance(detail.get("postedTs"), (int, float)) and not isinstance(detail["postedTs"], bool):
                try:
                    posted = datetime.fromtimestamp(detail["postedTs"], timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, OverflowError, OSError):
                    pass
            job = make_job(source, native, title, url, job_location, description,
                           detail.get("employmentType", ""), status, native, posted,
                           department=" / ".join(str(detail.get(key) or "") for key in (
                               "department", "departmentName", "jobFamily", "businessUnit")))
            if job:
                result.records.append(job)
            else:
                result.rejected_ids.append(identity)
        except Exception as exc:
            errors.append(brief_error(exc))

    result.seen_ids = sorted(seen)
    if errors:
        result.status = "partial" if seen or result.records else "error"
        result.error = "; ".join(dict.fromkeys(errors))[:400]
    return result


def _lever(source, client):
    data = client.get_json(source["api_url"])
    if not isinstance(data, list):
        raise FetchError("Lever did not return its complete postings array")
    result = SourceResult(source["id"], "ok", [], complete=True)
    for d in data:
        try:
            if not isinstance(d, dict) or not d.get("id") or not d.get("text") or not d.get("hostedUrl"):
                raise FetchError("Lever job entry shape changed")
            native, url = str(d["id"]), d["hostedUrl"]
            result.seen_ids.append(stable_job_id(source["id"], native, url))
            categories = d.get("categories") or {}
            location = " / ".join(categories.get("allLocations") or [categories.get("location", "")])
            if d.get("workplaceType") == "remote":
                location = "Remote / " + location
            desc = " ".join([d.get("descriptionPlain") or d.get("description", ""),
                             *[str(x.get("text", "")) + " " + str(x.get("content", "")) for x in d.get("lists", [])]])
            posted = None
            if isinstance(d.get("createdAt"), (int, float)):
                posted = datetime.fromtimestamp(d["createdAt"] / 1000, timezone.utc).strftime("%Y-%m-%d")
            job = make_job(source, native, d["text"], url, location, desc,
                           categories.get("commitment", ""), "open", posted_at=posted,
                           department=" / ".join(str(categories.get(key) or "") for key in ("department", "team")))
            if job:
                result.records.append(job)
            else:
                result.rejected_ids.append(stable_job_id(source["id"], native, url))
        except Exception as exc:
            result.status, result.complete, result.error = "partial", False, brief_error(exc)
    return result


def _ashby_location(label, address, *, primary=False):
    """Normalize only provider location fields; descriptions are not geography."""
    if label is not None and not isinstance(label, str):
        raise FetchError("Ashby location is malformed")
    if address is None:
        address = {}
    if not isinstance(address, dict):
        raise FetchError("Ashby location address is malformed")
    if primary:
        address = address.get("postalAddress") or {}
        if not isinstance(address, dict):
            raise FetchError("Ashby postal address is malformed")
    countries = {"US": "United States", "USA": "United States", "GB": "United Kingdom", "GBR": "United Kingdom",
                 "SE": "Sweden", "SWE": "Sweden", "DE": "Germany", "DEU": "Germany",
                 "FR": "France", "FRA": "France", "CH": "Switzerland", "CHE": "Switzerland",
                 "NL": "Netherlands", "NLD": "Netherlands", "IE": "Ireland", "IRL": "Ireland",
                 "FI": "Finland", "FIN": "Finland", "NO": "Norway", "NOR": "Norway",
                 "DK": "Denmark", "DNK": "Denmark", "AT": "Austria", "AUT": "Austria",
                 "BE": "Belgium", "BEL": "Belgium", "ES": "Spain", "ESP": "Spain",
                 "IT": "Italy", "ITA": "Italy", "PL": "Poland", "POL": "Poland",
                 "PT": "Portugal", "PRT": "Portugal", "CA": "Canada", "CAN": "Canada",
                 "JP": "Japan", "JPN": "Japan", "IL": "Israel", "ISR": "Israel",
                 "IN": "India", "IND": "India", "CN": "China", "CHN": "China",
                 "SG": "Singapore", "SGP": "Singapore", "AU": "Australia", "AUS": "Australia"}
    parts = [clean_text(label, 260)] if label else []
    for key in ("addressLocality", "addressRegion", "addressCountry"):
        value = address.get(key)
        if value is not None and not isinstance(value, str):
            raise FetchError("Ashby address component is malformed")
        value = clean_text(value, 150)
        if key == "addressCountry":
            value = countries.get(value.upper(), value)
        if value and not any(value.casefold() in part.casefold() for part in parts):
            parts.append(value)
    return ", ".join(parts)


def _ashby(source, client):
    # Official v1 documentation describes one complete published-postings array,
    # not a paginated search: https://developers.ashbyhq.com/docs/public-job-posting-api
    data = client.get_json(source["api_url"])
    if not isinstance(data, dict) or data.get("apiVersion") != "1" or not isinstance(data.get("jobs"), list):
        raise FetchError("Ashby did not return its v1 published-postings array")
    board = urlsplit(source["api_url"]).path.rstrip("/").rsplit("/", 1)[-1]
    result = SourceResult(source["id"], "ok", [], complete=True)
    seen = set()
    for d in data["jobs"]:
        try:
            if not isinstance(d, dict) or not isinstance(d.get("title"), str) or not d["title"].strip():
                raise FetchError("Ashby job title is missing or malformed")
            url = safe_url(d.get("jobUrl"), source["allowed_hosts"])
            parts = urlsplit(url)
            path = re.fullmatch(r"/" + re.escape(board) + r"/([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})/?", parts.path)
            if parts.hostname != "jobs.ashbyhq.com" or not path:
                raise FetchError("Ashby job URL does not identify a posting on the configured board")
            native = path.group(1).lower()
            if d.get("id") is not None and d["id"] != native:
                raise FetchError("Ashby job ID differs from its posting URL")
            identity = stable_job_id(source["id"], native, url)
            if identity in seen:
                raise FetchError("Ashby returned duplicate posting identities")
            seen.add(identity)
            if not isinstance(d.get("isListed"), bool):
                raise FetchError("Ashby listing visibility is missing or malformed")
            if not d["isListed"]:
                # Unlisted means direct-link-only, not closed. Retain its ID as
                # seen so an older saved record cannot be closed by absence.
                continue
            locations = [_ashby_location(d.get("location"), d.get("address"), primary=True)]
            secondary = d.get("secondaryLocations") or []
            if not isinstance(secondary, list):
                raise FetchError("Ashby secondary locations are malformed")
            for item in secondary:
                if not isinstance(item, dict):
                    raise FetchError("Ashby secondary location is malformed")
                locations.append(_ashby_location(item.get("location"), item.get("address")))
            locations = list(dict.fromkeys(x for x in locations if x))
            regions = list(dict.fromkeys(region_for(re.sub(r"\bremote\b|work from home", "", loc, flags=re.I)) for loc in locations)) or ["Unknown"]
            if all(region == "Other" for region in regions):
                continue
            workplace = d.get("workplaceType")
            if workplace not in (None, "OnSite", "Remote", "Hybrid"):
                raise FetchError("Ashby workplace type is malformed")
            if d.get("isRemote") is not None and not isinstance(d["isRemote"], bool):
                raise FetchError("Ashby remote flag is malformed")
            # isRemote also covers Hybrid in Ashby. Only Remote is fully remote.
            location = " / ".join(locations)
            if workplace == "Remote":
                location = "Remote / " + location if location else "Remote"
                if "Remote" not in regions:
                    regions.append("Remote")
            elif workplace == "Hybrid":
                location = "Hybrid / " + location if location else "Hybrid"
            for key in ("descriptionHtml", "descriptionPlain", "employmentType", "department", "team"):
                if d.get(key) is not None and not isinstance(d[key], str):
                    raise FetchError("Ashby job text is malformed")
            job = make_job(source, native, d["title"], url, location,
                           d.get("descriptionHtml") or d.get("descriptionPlain") or "", d.get("employmentType") or "",
                           "open", department=" / ".join(filter(None, (d.get("department"), d.get("team")))))
            # publishedAt is the last publication, not original posting or
            # discovery time. Do not substitute it for either date.
            if job:
                job["regions"] = regions
                if workplace in {"OnSite", "Hybrid"}:
                    # Location labels can contain "Remote" too; the provider's
                    # explicit workplace enum governs the fully-remote filter.
                    job["region"] = region_for(re.sub(r"\bremote\b|work from home", "", " / ".join(locations), flags=re.I))
                result.records.append(job)
            else:
                result.rejected_ids.append(identity)
        except Exception as exc:
            result.status, result.complete, result.error = "partial", False, brief_error(exc)
    result.seen_ids = sorted(seen)
    return result


def _greenhouse(source, client):
    data = client.get_json(source["api_url"])
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise FetchError("Greenhouse jobs array is missing")
    result = SourceResult(source["id"], "ok", [], complete=True)
    total = (data.get("meta") or {}).get("total")
    if total is not None and total != len(data["jobs"]):
        result.status, result.complete, result.error = "partial", False, "Greenhouse total does not match returned jobs"
    for d in data["jobs"]:
        try:
            if not isinstance(d, dict) or not d.get("id") or not d.get("title") or not d.get("absolute_url"):
                raise FetchError("Greenhouse job entry shape changed")
            native, url = str(d["id"]), d["absolute_url"]
            result.seen_ids.append(stable_job_id(source["id"], native, url))
            location = (d.get("location") or {}).get("name", "")
            job = make_job(source, native, d["title"], url, location, d.get("content", ""),
                           "", "open", posted_at=d.get("first_published"),
                           department=" / ".join(str(dep.get("name", "")) for dep in (d.get("departments") or []) if isinstance(dep, dict)))
            if job:
                result.records.append(job)
            else:
                result.rejected_ids.append(stable_job_id(source["id"], native, url))
        except Exception as exc:
            result.status, result.complete, result.error = "partial", False, brief_error(exc)
    return result


class PageMetadata(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.jsonld = [], []
        self.in_json = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append((attrs["href"]))
        if tag == "script" and attrs.get("type", "").lower() == "application/ld+json":
            self.in_json, self.parts = True, []

    def handle_endtag(self, tag):
        if tag == "script" and self.in_json:
            self.jsonld.append("".join(self.parts))
            self.in_json = False

    def handle_data(self, data):
        if self.in_json:
            self.parts.append(data)


def _job_objects(value):
    if isinstance(value, list):
        for child in value:
            yield from _job_objects(child)
    elif isinstance(value, dict):
        kind = value.get("@type", "")
        if kind == "JobPosting" or (isinstance(kind, list) and "JobPosting" in kind):
            yield value
        for key in ("@graph", "mainEntity", "itemListElement", "item"):
            if key in value:
                yield from _job_objects(value[key])


def _address_text(value):
    if isinstance(value, list):
        return " / ".join(filter(None, (_address_text(x) for x in value)))
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    if "address" in value:
        return _address_text(value["address"])
    country = value.get("addressCountry", "")
    if isinstance(country, dict):
        country = country.get("name", "")
    # Jobylon sometimes places an explicit country in streetAddress while
    # omitting addressCountry. Keep the country evidence without publishing
    # an unnecessary street address or inferring it from company boilerplate.
    if not country and re.search(r"(?:^|,)\s*(?:sweden|sverige)\s*$", str(value.get("streetAddress", "")), re.I):
        country = "Sweden"
    country = {"SE": "Sweden", "SWE": "Sweden"}.get(str(country).upper(), country)
    parts = list(dict.fromkeys(str(x) for x in [value.get("addressLocality", ""), value.get("addressRegion", "")] if x))
    location = ", ".join(parts)
    if country and not re.search(r"(?:^|,\s*)" + re.escape(str(country)) + r"(?:,|$)", location, re.I):
        parts.append(str(country))
    return ", ".join(parts)


def _inline_html(source, text):
    mode = source.get("inline_discovery")
    if mode == "merl":
        headings = list(re.finditer(r'<h2\b[^>]*\bid=["\']([^"\']+)["\'][^>]*>(.*?)</h2>', text, re.I | re.S))
    elif mode == "zadar":
        headings = list(re.finditer(r'<span\b[^>]*class=["\'][^"\']*accordion-item__title[^"\']*["\'][^>]*>(.*?)</span>', text, re.I | re.S))
    else:
        raise FetchError("HTML source has no supported inline parser")
    if not headings:
        raise FetchError("HTML job headings disappeared; listing completeness is unknown")
    result = SourceResult(source["id"], "ok", [], complete=source.get("complete_listing", False))
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        title = clean_text(match.group(2 if mode == "merl" else 1), 260)
        native = match.group(1) if mode == "merl" else re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if mode == "merl":
            title = re.sub(r"^[A-Z]+\d+:\s*", "", title)
        else:
            title = re.sub(r"^zada(?:r)? labs\s*\|\s*", "", title, flags=re.I)
            native = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        url = source["index_url"] + ("#" + native if mode == "merl" else "")
        result.seen_ids.append(stable_job_id(source["id"], native, url))
        description = clean_text(text[match.end():end], 25000)
        # Geography and eligibility remain within this one role's isolated block.
        location_match = re.search(r"\b(?:based in|located in|location\s*:|office in)\s+([^.;\n]{3,90})", description, re.I)
        candidate_location = location_match.group(1) if location_match else ""
        location = candidate_location if region_for(candidate_location) != "Unknown" else ""
        job = make_job(source, native, title, url, location, description,
                       "Internship" if mode == "merl" else "", "listed", native if mode == "merl" else "")
        if job:
            result.records.append(job)
        else:
            result.rejected_ids.append(stable_job_id(source["id"], native, url))
    return result


def _jobylon_embed_urls(page):
    """Read literal ID/URL pairs from the public embed; never run its scripts."""
    pair = re.compile(
        r"(?m)^\s*id:\s*['\"](?P<job_id>[0-9]+)['\"],\s*\n"
        r"\s*url:\s*['\"](?P<path>/jobs/(?P=job_id)-[a-z0-9-]+/)['\"],\s*$"
    )
    return sorted({"https://emp.jobylon.com" + match.group("path")
                   for match in pair.finditer(page)})


def _html(source, client):
    text = client.get_text(source["index_url"])
    if CHALLENGE.search(text):
        raise FetchError("Employer returned an access challenge")
    if source.get("inline_discovery"):
        return _inline_html(source, text)
    parser = PageMetadata()
    parser.feed(text)
    patterns = [re.compile(x) for x in source.get("job_url_patterns", [])]
    links = set()
    for href in parser.links:
        url = urljoin(source["index_url"], href)
        if any(pattern.fullmatch(url) for pattern in patterns):
            links.add(safe_url(url, source["allowed_hosts"]))
    if source.get("html_discovery") == "jobylon":
        for url in _jobylon_embed_urls(text):
            if any(pattern.fullmatch(url) for pattern in patterns):
                links.add(safe_url(url, source["allowed_hosts"]))
    if not links:
        raise FetchError("Employer job links disappeared; listing completeness is unknown")
    seeds = list(dict.fromkeys(source.get("seed_urls", [])))
    known_jobs = sorted((j for j in source.get("_known_jobs", []) if j.get("url")),
                        key=lambda j: (j.get("last_detail_attempt") or j.get("last_verified") or j.get("last_seen", ""), j["url"]))
    known = list(dict.fromkeys(j["url"] for j in known_jobs if j["url"] not in seeds))
    fresh = sorted(links - set(seeds) - set(known), key=lambda u: (
        0 if re.search(r"uwb|radar|slam|localization|wireless|sensing", u, re.I) else 1, u))
    urls = seeds + known + fresh
    result = SourceResult(source["id"], "ok", [], complete=source.get("complete_listing", False))
    errors = []
    limit = source.get("max_details", 12)
    if len(urls) > limit:
        result.complete = False
        errors.append("Detail-check limit reached; additional listings remain unchecked")
    slots = max(0, limit - len(seeds))
    known_slots = min(len(known), slots // 2 if fresh else slots)
    fresh_slots = slots - known_slots
    # Unknown/irrelevant pages are intentionally not persisted as jobs. A weekly
    # rotating window still gives their URLs coverage instead of rechecking the
    # same first twelve pages forever.
    if fresh and len(fresh) > fresh_slots and fresh_slots > 0:
        offset = (_rotation_week(source) * fresh_slots) % len(fresh)
        fresh = fresh[offset:] + fresh[:offset]
    selected = (seeds + known[:known_slots] + fresh[:fresh_slots])[:limit]
    if len(selected) < limit:
        selected.extend(known[known_slots:known_slots + limit - len(selected)])
    for url in selected:
        for old in known_jobs:
            if canonical_url(old["url"]) == canonical_url(url):
                result.attempted_ids.append(old["id"])
        try:
            safe_url(url, source["allowed_hosts"])
            detail = client.get_text(url)
            if CHALLENGE.search(detail):
                raise FetchError("Employer returned an access challenge")
            p = PageMetadata()
            p.feed(detail)
            objects = []
            for item in p.jsonld:
                objects.extend(_job_objects(json.loads(item)))
            if not objects:
                if EXPLICIT_CLOSED.search(clean_text(detail, 20000)):
                    for old in source.get("_known_jobs", []):
                        if canonical_url(old.get("url", "")) == canonical_url(url):
                            result.closed_ids.append(old["id"])
                    continue
                raise FetchError("HTML detail no longer contains JobPosting data")
            for d in objects:
                target = d.get("url") or url
                identifier = d.get("identifier") or ""
                native = identifier.get("value", "") if isinstance(identifier, dict) else str(identifier)
                if not native:
                    native = urlsplit(url).path.rstrip("/").split("/")[-1]
                identity = stable_job_id(source["id"], native, target)
                result.seen_ids.append(identity)
                if identity not in result.attempted_ids:
                    result.attempted_ids.append(identity)
                location = _address_text(d.get("jobLocation", ""))
                if d.get("jobLocationType") == "TELECOMMUTE":
                    location = "Remote / " + location
                employment = d.get("employmentType", "")
                if isinstance(employment, list):
                    employment = " ".join(str(x) for x in employment)
                job = make_job(source, native, d.get("title", ""), target, location,
                               d.get("description", ""), employment, "listed", posted_at=d.get("datePosted"),
                               department=str(d.get("occupationalCategory") or d.get("industry") or ""))
                if job:
                    result.records.append(job)
                else:
                    result.rejected_ids.append(identity)
        except Exception as exc:
            errors.append(brief_error(exc))
    if errors:
        result.status = "partial" if result.records or result.seen_ids else "error"
        result.complete, result.error = False, "; ".join(dict.fromkeys(errors))[:400]
    return result


def fetch_source(source, client):
    try:
        result = {"workday": _workday, "lever": _lever, "greenhouse": _greenhouse, "ashby": _ashby,
                  "html": _html, "amazon": _amazon, "ericsson": _ericsson}[source["type"]](source, client)
    except Exception as exc:
        result = SourceResult(source["id"], "error", [], error=brief_error(exc))
    result.requests = getattr(client, "requests", 0)
    return result


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1 or not isinstance(config.get("sources"), list):
        raise ValueError("Configuration must have schema_version 1 and a sources array")
    seen = set()
    for source in config["sources"]:
        if not isinstance(source, dict) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,60}", source.get("id", "")):
            raise ValueError("Every source needs a safe, stable id")
        if source["id"] in seen or source["id"] == "manual":
            raise ValueError("Duplicate or reserved source id")
        seen.add(source["id"])
        if source.get("type") not in SOURCE_TYPES or not source.get("name") or not source.get("company"):
            raise ValueError("Source type, name, and company are required")
        if str(source["company"]).strip().casefold() in EXCLUDED_DEFENCE_COMPANIES:
            raise ValueError("Defence employer is excluded from civilian job watch")
        if "civilian_review_required" in source and not isinstance(source["civilian_review_required"], bool):
            raise ValueError("civilian_review_required must be boolean")
        verified = source.get("civilian_verified_job_ids", [])
        if (not isinstance(verified, list) or len(verified) > 100
                or len(verified) != len(set(str(x) for x in verified))
                or any(not isinstance(x, str) or not x.startswith(source["id"] + ":")
                       or not re.fullmatch(r"[a-z0-9][a-z0-9._:-]{1,150}", x) for x in verified)):
            raise ValueError("Invalid civilian_verified_job_ids")
        if verified and not source.get("civilian_review_required"):
            raise ValueError("Civilian verified IDs require role review")
        hosts = source.get("allowed_hosts")
        if not isinstance(hosts, list) or not hosts or any(not re.fullmatch(r"[a-z0-9.-]+", h) for h in hosts):
            raise ValueError("Source requires exact allowed_hosts")
        for key in ("base_url", "search_url", "api_url", "detail_url", "index_url", "careers_url"):
            if key in source:
                safe_url(source[key], hosts)
        if source.get("country_filter") not in (None, "SE"):
            raise ValueError("Unsupported country filter")
        if "exclude_student_theses" in source and not isinstance(source["exclude_student_theses"], bool):
            raise ValueError("exclude_student_theses must be boolean")
        for key in ("seed_urls",):
            for url in source.get(key, []):
                safe_url(url, hosts)
        if source["type"] == "workday":
            for key in ("base_url", "search_url", "tenant", "site"):
                if not source.get(key):
                    raise ValueError(f"Workday source missing {key}")
            if not all(re.fullmatch(r"[A-Za-z0-9_-]+", source[k]) for k in ("tenant", "site")):
                raise ValueError("Invalid Workday tenant/site")
            facets = source.get("applied_facets", {})
            if (not isinstance(facets, dict) or len(facets) > 5
                    or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(key))
                           or not isinstance(values, list) or not 1 <= len(values) <= 10
                           or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value)
                                  for value in values) for key, values in facets.items())):
                raise ValueError("Invalid Workday applied facets")
            queries = source.get("queries", [""])
            if not isinstance(queries, list) or not 1 <= len(queries) <= 10 or any(not isinstance(q, str) or len(q) > 80 for q in queries):
                raise ValueError("Workday queries must be a bounded string array")
            for path in source.get("seed_paths", []):
                if not path.startswith("/job/") or ".." in path:
                    raise ValueError("Invalid Workday seed path")
        elif source["type"] in {"lever", "greenhouse", "ashby", "amazon", "ericsson"} and not source.get("api_url"):
            raise ValueError("API source missing api_url")
        if source["type"] == "ashby":
            endpoint = urlsplit(source["api_url"])
            if endpoint.hostname != "api.ashbyhq.com" or not re.fullmatch(r"/posting-api/job-board/[A-Za-z0-9_-]+/?", endpoint.path):
                raise ValueError("Ashby source requires an official public posting API endpoint")
        if source["type"] in {"amazon", "ericsson"}:
            queries = source.get("queries", ["SLAM", "localization"])
            if (not isinstance(queries, list) or not 1 <= len(queries) <= 10
                    or any(not isinstance(q, str) or not q.strip() or len(q) > 80 for q in queries)):
                raise ValueError("Search queries must be a bounded nonempty string array")
            if source["type"] == "ericsson":
                if not source.get("detail_url"):
                    raise ValueError("Ericsson source missing detail_url")
                if not re.fullmatch(r"[a-z0-9.-]{1,100}", source.get("domain", "ericsson.com")):
                    raise ValueError("Invalid Ericsson search domain")
                location = source.get("search_location", "Sweden")
                if not isinstance(location, str) or len(location) > 120:
                    raise ValueError("Invalid Ericsson search location")
                seeds = source.get("seed_ids", [])
                if (not isinstance(seeds, list) or len(seeds) > 30
                        or any(not isinstance(seed, str) or not re.fullmatch(r"[0-9]{1,24}", seed)
                               for seed in seeds)):
                    raise ValueError("Invalid Ericsson seed IDs")
        elif source["type"] == "html":
            if not source.get("index_url"):
                raise ValueError("HTML source missing index_url")
            if source.get("inline_discovery") not in (None, "merl", "zadar"):
                raise ValueError("Unsupported inline HTML parser")
            if source.get("html_discovery") not in (None, "jobylon"):
                raise ValueError("Unsupported HTML discovery mode")
            for pattern in source.get("job_url_patterns", []):
                if len(pattern) > 250:
                    raise ValueError("Overlong job URL pattern")
                re.compile(pattern)
        for key, maximum in (("page_size", 50), ("max_pages_per_query", 5), ("max_details", 30), ("max_requests", 40)):
            if key in source and (not isinstance(source[key], int) or not 1 <= source[key] <= maximum):
                raise ValueError(f"Invalid bound for {key}")
    for section in ("manual_jobs", "pinned_jobs"):
        if not isinstance(config.get(section, []), list):
            raise ValueError(f"{section} must be an array")
        for job in config.get(section, []):
            if not isinstance(job, dict) or not job.get("title") or not job.get("url"):
                raise ValueError("Saved jobs require title and URL")
            if section == "pinned_jobs" and job.get("source_id") not in seen:
                raise ValueError("Pinned job references an unknown source")
            hosts = [urlsplit(job["url"]).hostname] if section == "manual_jobs" else next(s["allowed_hosts"] for s in config["sources"] if s["id"] == job["source_id"])
            safe_url(job["url"], hosts)
    reviews = config.get("eligibility_reviews", [])
    if not isinstance(reviews, list) or len(reviews) > 600:
        raise ValueError("eligibility_reviews must be a bounded array")
    reviewed_ids = set()
    for review in reviews:
        identity = review.get("job_id") if isinstance(review, dict) else None
        if (not isinstance(identity, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,60}:[a-z0-9._-]{1,130}", identity)
                or identity.split(":", 1)[0] not in seen | {"manual"} or identity in reviewed_ids):
            raise ValueError("Eligibility review requires a unique full job_id from a configured source")
        reviewed_ids.add(identity)
        _normalize_eligibility_review(review, strict=True)
    return config


def _public_job(record, source):
    """Whitelist output fields; raw employer descriptions never enter public JSON."""
    url = safe_url(record["url"], source["allowed_hosts"])
    title = clean_text(record.get("title", ""), 260)
    if not title:
        raise ValueError("Job title is empty after sanitization")
    company = clean_text(record.get("company") or source["company"], 120)
    if company.casefold() in EXCLUDED_DEFENCE_COMPANIES or is_military_role(
            title, record.get("description", ""), record.get("department", "")):
        raise ValueError("Non-civilian record excluded")
    if (source.get("civilian_review_required")
            and record.get("id") not in set(source.get("civilian_verified_job_ids", []))):
        raise ValueError("Mixed-use role lacks verified civilian assignment")
    classification = classify_job(title, record.get("description", ""), record.get("location", ""), record.get("employment_type", ""))
    category = record.get("category") if record.get("category") in CATEGORIES else (classification or {}).get("category")
    if not category:
        raise ValueError("Job has no supported relevance category")
    job = {"id": str(record.get("id") or stable_job_id(source["id"], record.get("requisition_id"), url)),
           "source_id": source["id"], "company": clean_text(record.get("company") or source["company"], 120),
           "title": title, "url": url, "location": clean_text(record.get("location", ""), 260),
           "region": record.get("region") if record.get("region") in {"US", "Europe", "Remote", "Other", "Unknown"} else region_for(record.get("location", "")),
           "employment_type": record.get("employment_type") if record.get("employment_type") in {"Internship", "Full-time", "Unknown"} else (classification or {}).get("employment_type", "Unknown"),
           "category": category, "status": record.get("status") if record.get("status") in JOB_STATUSES else "unknown",
           "pinned": bool(record.get("pinned", False)), "score": max(0, min(int(record.get("score", 70)), 150))}
    if "regions" in record:
        regions = record["regions"]
        if (not isinstance(regions, list) or not 1 <= len(regions) <= 5
                or any(not isinstance(region, str) or region not in {"US", "Europe", "Remote", "Other", "Unknown"} for region in regions)):
            raise ValueError("Invalid job regions")
        job["regions"] = list(dict.fromkeys(regions))
    for key in ("match_reasons", "requirements"):
        items = record.get(key, (classification or {}).get(key, []))
        job[key] = [clean_text(x, 150) for x in items[:12] if clean_text(x, 150)] if isinstance(items, list) else []
    templates = {
        "radio-localization": "Role text links radio or RF measurements with positioning, ranging, or localization.",
        "radar-sensing": "Role text includes radar sensing, signal processing, or radar-based perception.",
        "robotics-localization": "Role text includes robotic localization, mapping, tracking, or sensor fusion.",
        "wireless-isac": "Role text connects wireless systems with integrated sensing and communications.",
    }
    job["summary"] = templates[category]
    for key in ("requisition_id", "posted_at", "first_seen", "last_seen", "last_checked", "last_verified", "last_detail_attempt", "check_status", "last_check_error"):
        if record.get(key):
            job[key] = clean_text(record[key], 220 if key == "last_check_error" else 100)
    for key in ("fit_note", "compensation_note"):
        if isinstance(record.get(key), str):
            job[key] = clean_text(record[key], 800)
    observed = _normalize_eligibility(record.get("eligibility"))
    posting = _normalize_eligibility(record.get("eligibility_posting")) if "eligibility_posting" in record else (
        observed if observed["method"] == "posting_text" else _unknown_eligibility())
    review = _normalize_eligibility_review(record.get("eligibility_review"))
    if not review and observed["method"] == "manual_review":
        review = {"eligibility": observed, **{key: job[key] for key in ("fit_note", "compensation_note") if key in job}}
    _set_job_eligibility(job, posting, review)
    job["missing_count"] = max(0, int(record.get("missing_count", 0)))
    return job


def _same_job(a, b):
    if a["id"] == b["id"]:
        return True
    ar, br = a.get("requisition_id"), b.get("requisition_id")
    if ar and br and a["company"].casefold() == b["company"].casefold() and ar.casefold() == br.casefold():
        return True
    return canonical_url(a["url"]) == canonical_url(b["url"]) and a["title"].casefold() == b["title"].casefold()


def _careers_url(source):
    url = source.get("careers_url") or source.get("index_url")
    if not url and source.get("base_url"):
        url = source["base_url"].rstrip("/")
        if source.get("type") == "workday" and source.get("site"):
            url += "/" + source["site"]
    if url:
        try:
            return safe_url(url, source["allowed_hosts"])
        except (ValueError, TypeError, KeyError):
            pass
    return ""


def merge_results(previous, results, config, now):
    sources = {s["id"]: s for s in config["sources"]}
    manual_hosts = {urlsplit(j["url"]).hostname for j in config.get("manual_jobs", [])}
    sources["manual"] = {"id": "manual", "name": "Manually saved references", "company": "", "allowed_hosts": list(manual_hosts)}
    jobs = {}
    for old in previous.get("jobs", []):
        if old.get("source_id") not in sources:
            continue
        try:
            job = _public_job(old, sources[old["source_id"]])
            jobs[job["id"]] = job
        except (ValueError, TypeError, KeyError):
            continue
    # Seed only explicitly configured saved opportunities, never scraped social links.
    for key in ("pinned_jobs", "manual_jobs"):
        for seed in config.get(key, []):
            sid = "manual" if key == "manual_jobs" else seed["source_id"]
            record = dict(seed, source_id=sid, pinned=True, status="saved" if sid == "manual" else "unknown")
            record["id"] = stable_job_id(sid, seed.get("requisition_id") or seed.get("id"), seed["url"])
            record.setdefault("category", (classify_job(seed["title"], "", seed.get("location", "")) or {}).get("category", "radio-localization"))
            record.update(first_seen=now, last_seen=now, last_checked=now, check_status="manual" if sid == "manual" else "partial")
            seed_job = _public_job(record, sources[sid])
            existing = next((j for j in jobs.values() if _same_job(j, seed_job)), None)
            if existing:
                existing["pinned"] = True
                existing["url"] = seed_job["url"]
            else:
                jobs[seed_job["id"]] = seed_job
    source_states = {s["id"]: dict(s) for s in previous.get("sources", []) if s.get("id") in sources}
    for result in results:
        source = sources.get(result.source_id)
        if not source or result.source_id == "manual":
            continue
        source_state = {"id": result.source_id, "name": source["name"], "status": result.status,
                        "checked_at": result.checked_at or now, "count": len(result.records)}
        if result.error:
            source_state["error"] = clean_text(result.error, 400)
        source_states[result.source_id] = source_state
        seen = set(result.seen_ids)
        returned = set()
        for raw in result.records:
            try:
                current = _public_job(raw, source)
            except (ValueError, TypeError, KeyError):
                source_state["status"] = "partial"
                source_state["error"] = "One or more job records failed safe parsing; availability not established"
                continue
            existing = next((j for j in jobs.values() if _same_job(j, current)), None)
            if existing:
                current["first_seen"] = existing.get("first_seen", now)
                current["pinned"] = existing.get("pinned", False) or current["pinned"]
                _carry_eligibility(current, existing)
                if existing.get("pinned"):
                    current["url"] = existing["url"]
                if existing["id"] != current["id"]:
                    del jobs[existing["id"]]
            else:
                current["first_seen"] = now
            current.update(last_seen=now, last_checked=now, last_verified=now, check_status="ok", missing_count=0)
            if current["id"] in result.attempted_ids:
                current["last_detail_attempt"] = now
            current.pop("last_check_error", None)
            jobs[current["id"]] = current
            returned.add(current["id"])
            seen.add(current["id"])
        if source.get("civilian_review_required"):
            # A successfully read, now-disallowed role cannot survive merely
            # because it appeared in the previous snapshot. An inventory/search
            # sighting without a successful role check is not such evidence.
            for rejected_id in set(result.rejected_ids) - returned:
                existing = jobs.get(rejected_id)
                if existing and existing["source_id"] == result.source_id:
                    del jobs[rejected_id]
        for job in list(jobs.values()):
            if job["source_id"] != result.source_id or job["id"] in returned:
                continue
            job["last_checked"] = now
            if job["id"] in result.attempted_ids:
                job["last_detail_attempt"] = now
            if job["id"] in result.closed_ids:
                job.update(status="closed", check_status="ok", last_verified=now)
                job.pop("last_check_error", None)
            elif result.status == "ok" and source_state["status"] == "ok" and result.complete and job["id"] not in seen:
                job["missing_count"] = job.get("missing_count", 0) + 1
                if job["missing_count"] >= 2:
                    job.update(status="closed", check_status="ok", last_verified=now)
                    job.pop("last_check_error", None)
                else:
                    job["check_status"] = "partial"
                    job["last_check_error"] = "Absent from one complete employer listing; closure not yet confirmed"
            else:
                if job["id"] in seen:
                    job["missing_count"] = 0
                    job["last_seen"] = now
                job["check_status"] = "error" if result.status == "error" else "partial"
                job["last_check_error"] = clean_text(result.error or "Not individually reconfirmed in this bounded check", 220)
    for job in jobs.values():
        if job["source_id"] == "manual":
            job.update(status="saved", check_status="manual", pinned=True)
            job["summary"] = "Manually saved reference. Availability is not checked automatically."
            job["match_reasons"] = ["User-saved reference; automatic verification is unavailable"]
            job["requirements"] = []
            job.pop("last_verified", None)
    # Reviews are explicit configuration, never inferred from an employer's
    # sponsorship history. Applying one does not change availability dates.
    for configured in config.get("eligibility_reviews", []):
        if job := jobs.get(configured["job_id"]):
            incoming = _normalize_eligibility_review(configured, strict=True)
            existing = job.get("eligibility_review")
            if not existing or (incoming["eligibility"].get("checked_at") or "") >= (existing["eligibility"].get("checked_at") or ""):
                _set_job_eligibility(job, job.get("eligibility_posting"), incoming)
    if config.get("manual_jobs"):
        source_states["manual"] = {"id": "manual", "name": "Manually saved references", "status": "manual",
                                   "checked_at": now, "count": sum(j["source_id"] == "manual" for j in jobs.values())}
    for sid, state in source_states.items():
        careers = _careers_url(sources[sid])
        if careers:
            state["careers_url"] = careers
        else:
            state.pop("careers_url", None)
    # Keep saved records, recent closed history, and the most relevant active records.
    cutoff = datetime.fromisoformat(now.replace("Z", "+00:00")) - timedelta(days=120)
    retained = []
    for job in jobs.values():
        if job["status"] == "closed" and not job["pinned"]:
            try:
                if datetime.fromisoformat(job.get("last_seen", now).replace("Z", "+00:00")) < cutoff:
                    continue
            except ValueError:
                pass
        retained.append(job)
    retained.sort(key=lambda j: (not j["pinned"], j["status"] == "closed", -j["score"], j["company"].casefold(), j["title"].casefold(), j["id"]))
    cap = min(max(int(config.get("max_retained_jobs", 300)), 20), 600)
    pinned_count = sum(j["pinned"] for j in retained)
    return {"schema_version": 1, "generated_at": now,
            "sources": [source_states[k] for k in sorted(source_states)], "jobs": retained[:max(cap, pinned_count)]}


def atomic_write_json(path, payload):
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".jobs-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "career/job-sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / "career/potential-jobs/jobs.json")
    parser.add_argument("--now", help="UTC ISO timestamp for deterministic runs/tests")
    parser.add_argument("--source", action="append", help="Check only this source; may be repeated")
    args = parser.parse_args(argv)
    try:
        config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
        now_dt = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        now = now_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {"jobs": [], "sources": []}
        if not isinstance(previous, dict) or not isinstance(previous.get("jobs", []), list):
            raise ValueError("Existing output is not a valid job collection; refusing to replace it")
        chosen = set(args.source or [])
        if chosen - {s["id"] for s in config["sources"]}:
            raise ValueError("Unknown --source id")
    except (ValueError, OSError, TypeError) as exc:
        print(f"Configuration/state error: {brief_error(exc)}", file=sys.stderr)
        return 2
    results = []
    budget = RunBudget(seconds=600, requests=240)
    for original in config["sources"]:
        if not original.get("enabled", True) or (chosen and original["id"] not in chosen):
            continue
        source = dict(original)
        source["_now"] = now
        source["_known_jobs"] = [j for j in previous.get("jobs", []) if j.get("source_id") == source["id"] and j.get("status") != "closed"]
        client = HttpClient(source["allowed_hosts"], source.get("max_requests", 30), budget=budget)
        result = fetch_source(source, client)
        result.checked_at = now
        results.append(result)
        print(f"{source['id']}: {result.status}; {len(result.records)} relevant postings; {result.requests} requests"
              + (f"; {result.error}" if result.error else ""), flush=True)
    if results and all(r.status == "error" for r in results):
        print("All selected employer sources failed; existing output left unchanged.", file=sys.stderr)
        return 1
    try:
        merged = merge_results(previous, results, config, now)
        atomic_write_json(args.output, merged)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"Output not replaced: {brief_error(exc)}", file=sys.stderr)
        return 2
    print(f"Saved {len(merged['jobs'])} jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

