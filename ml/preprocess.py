from __future__ import annotations

import re
import html
import pandas as pd
from bs4 import BeautifulSoup

# ----- CONFIG (match your SQL columns) -----
RAW_BODY_COL = "description"   # HTML in Dataverse
SUBJECT_COL = "subject"

# ===== Thread / Signature / Boilerplate Removal =====
BOILERPLATE_ANYWHERE_PATTERNS = [
    r"(?i)caution:\s*this email originated from outside of the organization\..*?(?=(please find|we are issuing|hi|good morning|hello|dear)\b)",
    r"(?i)do not click links or open attachments.*?(?=(please find|we are issuing|hi|good morning|hello|dear)\b)",
    r"(?i)external:\s*this email originated from outside.*?(?=(please find|we are issuing|hi|good morning|hello|dear)\b)",
]

SIGNATURE_ANYWHERE_PATTERNS = [
    r"(?i)\bwe value your business.*$",
]

THREAD_CUT_MARKERS = [
    r"(?im)^\s*-+\s*original message\s*-+\s*$",
    r"(?im)^\s*-+\s*forwarded message\s*-+\s*$",
    r"(?im)^\s*begin forwarded message:\s*$",
    r"(?im)^\s*forwarded message:\s*$",
    r"(?im)^\s*_{5,}\s*$",
    r"(?im)^\s*from:\s.+$",
    r"(?im)^\s*sent:\s.+$",
    r"(?im)^\s*to:\s.+$",
    r"(?im)^\s*subject:\s.+$",
    r"(?im)^\s*cc:\s.+$",
    r"(?im)^\s*bcc:\s.+$",
    r"(?im)^\s*on\s.+wrote:\s*$",
    r"(?im)^\s*>+",
]

SIGNATURE_CUT_MARKERS = [
    r"(?im)^\s*regards[,]?\s*$",
    r"(?im)^\s*kind regards[,]?\s*$",
    r"(?im)^\s*best[,]?\s*$",
    r"(?im)^\s*thanks[,]?\s*$",
    r"(?im)^\s*thank you[,]?\s*$",
    r"(?im)^\s*sincerely[,]?\s*$",
    r"(?im)^\s*respectfully[,]?\s*$",
    r"(?im)^\s*sent from my\s+.*$",
    r"(?im)^\s*customer care team.*$",
    r"(?im)^\s*commercial lines.*$",
    r"(?im)^\s*licensed advisor.*$",
    r"(?im)^\s*customer engagement.*$",
]

CAUTION_BANNER_PATTERNS = [
    r"(?is)\bCAUTION:\s*this email originated from outside of the organization\..*?safe\.\s*",
    r"(?is)\bEXTERNAL:\s*this email originated from outside.*?safe\.\s*",
    r"(?is)\bThis email originated from outside of the organization\..*?safe\.\s*",
]

BOILERPLATE_LINE_PATTERNS = [
    r"(?im)^\s*valued\s+.*customer.*$",
    r"(?im)^\s*brightway\s+insurance.*$",
    r"(?im)^\s*privacy\s+notice.*$",
    r"(?im)^\s*confidentiality\s+notice.*$",
    r"(?im)^\s*this\s+message.*confidential.*$",
    r"(?im)^\s*please\s+do\s+not\s+share.*$",
    r"(?im)^\s*all\s+rights\s+reserved.*$",
    r"(?im)^\s*unsubscribe.*$",
    r"(?im)^\s*caution:\s*this email originated from outside.*$",
    r"(?im)^\s*external:\s*this email originated from outside.*$",
    r"(?im)^\s*do not click links or open attachments.*$",
    r"(?im)^\s*thank you for being a part of the brightway.*$",
    r"(?im)^\s*we value your business.*$",
    r"(?im)^\s*thank you for choosing brightway.*$",
    r"(?im)^\s*brightway customer care team.*$",
    r"(?im)^\s*brightway\.com.*$",
]

ADDRESS_LINE_PATTERNS = [
    r"(?im)^\s*\d{2,5}\s+\w+.*(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane)\b.*$",
    r"(?im)^\s*(jacksonville|tampa|orlando|miami|atlanta|charlotte)\b.*$",
    r"(?im)^\s*\b(fl|ga|nc|sc|al|tn)\s+\d{5}(-\d{4})?\b.*$",
]

PHONE_RAW = r"(?:\+?1[\s\-\.]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}"
PHONE_TOKEN_OR_RAW = rf"(?:<PHONE>|{PHONE_RAW})"

CONTACT_US_PATTERNS = [
    rf"(?is)\byou may reach us at\s+{PHONE_TOKEN_OR_RAW}(?:\s*(?:or|/|,|and)\s*{PHONE_TOKEN_OR_RAW}){{0,2}}.*$",
    rf"(?is)\b(?:please\s+)?call\s+(?:us|me)\s+(?:at|on)\s+{PHONE_TOKEN_OR_RAW}(?:\s*(?:or|/|,|and)\s*{PHONE_TOKEN_OR_RAW}){{0,2}}.*$",
    rf"(?is)\byou can call\s+(?:us|me)\s+(?:at|on)\s+{PHONE_TOKEN_OR_RAW}(?:\s*(?:or|/|,|and)\s*{PHONE_TOKEN_OR_RAW}){{0,2}}.*$",
    rf"(?is)\bplease\s+contact\s+brightway\s+(?:insurance\s+)?(?:at|on)\s+{PHONE_TOKEN_OR_RAW}.*$",
    rf"(?is)\bcontact\s+brightway\s+(?:insurance\s+)?(?:at|on)\s+{PHONE_TOKEN_OR_RAW}.*$",
    rf"(?is)\bif you have any questions.*?\b(?:please\s+)?contact\s+brightway\s+(?:insurance\s+)?(?:at|on)\s+{PHONE_TOKEN_OR_RAW}.*$",
    rf"(?is)\bif you have any questions.*?\b(?:please\s+)?call\s+(?:us|me)\s+(?:at|on)\s+{PHONE_TOKEN_OR_RAW}.*$",
    r"(?is)\(m\s*-\s*f.*?(?:est|cst|pst|mst)\)\s*$",
    r"(?is)\b(m\s*-\s*f|mon(?:day)?\s*-\s*fri(?:day)?).*?\b(?:am|pm)\b.*?(?:est|cst|pst|mst)\b.*$",
]

SYSTEM_ARTIFACT_PATTERNS = [
    r"(?is)\bsentbyuser\s*:\s*[yn]\b.*$",
    r"(?is)\bsentbyuser\s*:\s*\w+\b.*$",
]

COVERAGE_DISCLAIMER_PATTERNS = [
    r"(?is)\binsurance coverage changes cannot be made\b.*$",
    r"(?is)\bcoverage changes cannot be made\b.*$",
    r"(?is)\bsuch changes must be made\b.*$",
    r"(?is)\bif you need any help with this, please reach out to us\b.*$",
]

GENERIC_HELP_FOOTER_PATTERNS = [
    r"(?is)\bif you have any questions[, ]+or require additional assistance\b.{0,250}$",
    r"(?is)\bif you have any questions\b.{0,200}$",
    r"(?is)\bplease let me know if you have any questions\b.{0,200}$",
]

EMAIL_FOOTER_PATTERNS = [
    r"(?is)\bresponses?\s+can\s+be\s+e-?mailed\s+to\s+<EMAIL>\s*[\.\)!]?\s*.*$",
    r"(?is)\bresponses?\s+may\s+be\s+e-?mailed\s+to\s+<EMAIL>\s*[\.\)!]?\s*.*$",
    r"(?is)\bplease\s+(email|e-?mail)\b.*?\b<EMAIL>\b\s*[\.\)!]?\s*.*$",
    r"(?is)\byou\s+may\s+(email|e-?mail)\b.*?\b<EMAIL>\b\s*[\.\)!]?\s*.*$",
]

FLUFF_FOOTER_PATTERNS = [
    r"(?is)\bwe appreciate both your time and attention to this matter\..*$",
    r"(?is)\bwe appreciate your time and attention\..*$",
]


def _first_match_line_index(lines: list[str], patterns: list[str]):
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.search(pat, line):
                return i
    return None


def cut_at_marker(text: str, patterns: list[str]) -> str:
    lines = text.splitlines()
    idx = _first_match_line_index(lines, patterns)
    if idx is None:
        return text.strip()
    return "\n".join(lines[:idx]).strip()


def remove_matching_lines(text: str, patterns: list[str]) -> str:
    out = []
    for line in text.splitlines():
        if any(re.search(pat, line) for pat in patterns):
            continue
        out.append(line)
    return "\n".join(out).strip()


def strip_anywhere_patterns(text: str, patterns: list[str]) -> str:
    out = text
    for pat in patterns:
        out = re.sub(pat, " ", out)
    return re.sub(r"\s+", " ", out).strip()


def clean_html_email(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    soup = BeautifulSoup(text, "lxml")

    cleaned = soup.get_text("\n", strip=True)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    cleaned = cut_at_marker(cleaned, THREAD_CUT_MARKERS)
    cleaned = remove_matching_lines(cleaned, BOILERPLATE_LINE_PATTERNS)
    cleaned = remove_matching_lines(cleaned, ADDRESS_LINE_PATTERNS)
    cleaned = cut_at_marker(cleaned, SIGNATURE_CUT_MARKERS)

    cleaned = re.sub(
        r"(?im)\b(regards|thanks|thank you|sincerely|best)[, ]+\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
        r"\1 <NAME>",
        cleaned
    )

    cleaned = strip_anywhere_patterns(cleaned, BOILERPLATE_ANYWHERE_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, SIGNATURE_ANYWHERE_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, CAUTION_BANNER_PATTERNS)

    # Tokenize PII-like items
    cleaned = re.sub(r"(?i)\bhttps?://\S+\b", " <URL> ", cleaned)
    cleaned = re.sub(r"(?i)\b[\w\.-]+@[\w\.-]+\.\w+\b", " <EMAIL> ", cleaned)
    cleaned = re.sub(r"(?i)\b(ext|x)\s*\d+\b", " <EXT> ", cleaned)
    cleaned = re.sub(r"\b(\+?1[\s\-\.]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b", " <PHONE> ", cleaned)
    cleaned = re.sub(r"\b\d{6,}\b", " <NUM> ", cleaned)

    cleaned = strip_anywhere_patterns(cleaned, EMAIL_FOOTER_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, FLUFF_FOOTER_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, CONTACT_US_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, GENERIC_HELP_FOOTER_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, COVERAGE_DISCLAIMER_PATTERNS)
    cleaned = strip_anywhere_patterns(cleaned, SYSTEM_ARTIFACT_PATTERNS)

    cleaned = re.sub(r"\b(?=[a-z0-9]{12,}\b)(?=.*[a-z])(?=.*\d)[a-z0-9]+\b", " <ID> ", cleaned, flags=re.I)
    cleaned = re.sub(r"(<ID>)[\.\,\:\;\)\]]+", r"\1", cleaned)

    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Safety fallback
    if len(cleaned) < 30:
        fallback = soup.get_text(" ", strip=True)
        fallback = re.sub(r"\s+", " ", fallback).strip()
        fallback = re.sub(r"(?i)\bhttps?://\S+\b", " <URL> ", fallback)
        fallback = re.sub(r"(?i)\b[\w\.-]+@[\w\.-]+\.\w+\b", " <EMAIL> ", fallback)
        fallback = re.sub(r"\b(\+?1[\s\-\.]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b", " <PHONE> ", fallback)
        fallback = re.sub(r"\b\d{6,}\b", " <NUM> ", fallback)
        fallback = re.sub(r"\s+", " ", fallback).strip()
        cleaned = fallback[:2000]

    return cleaned


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required inputs:
      - df['subject'] (optional but recommended)
      - df['description'] (raw HTML body)
    Output:
      - adds df['clean_text'], df['modeltext']
      - drops empty modeltext rows
    """
    df = df.copy()

    # Normalize headers to lowercase
    df.columns = [str(c).strip().lower() for c in df.columns]

    if RAW_BODY_COL not in df.columns:
        raise ValueError(f"Missing required column: {RAW_BODY_COL}")

    if SUBJECT_COL not in df.columns:
        df[SUBJECT_COL] = ""

    df["clean_text"] = df[RAW_BODY_COL].fillna("").astype(str).apply(clean_html_email)

    df["modeltext"] = (
        df[SUBJECT_COL].fillna("").astype(str).str.strip()
        + "  "
        + df["clean_text"].fillna("").astype(str).str.strip()
    ).str.slice(0, 2000)

    # Drop empty modeltext
    df["modeltext"] = df["modeltext"].fillna("").astype(str).str.strip()
    df = df[df["modeltext"].str.len() > 0].copy()

    return df
