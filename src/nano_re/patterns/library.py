"""Rule library for Polish business documents.

Rules are ordinary data, so a deployment adds its own document numbering scheme
by appending to a list rather than by editing extraction code.

Where a checksum exists it is enforced. An unvalidated ten digit run is a false
positive waiting to happen in a document full of amounts and dates.
"""

from __future__ import annotations

import re

from .rules import (
    PatternRule,
    alphanumeric_upper,
    digits_only,
    is_valid_iban,
    is_valid_nip,
    is_valid_pesel,
    is_valid_polish_account,
    is_valid_regon,
)

NIP = PatternRule(
    name="nip",
    entity_type="NIP",
    pattern=re.compile(
        r"(?:NIP|N\.I\.P\.)[:\s]*(?P<value>(?:PL)?\s?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}"
        r"|(?:PL)?\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{3}|(?:PL)?\s?\d{10})",
        re.IGNORECASE,
    ),
    validator=is_valid_nip,
    normaliser=digits_only,
    priority=30,
)

REGON = PatternRule(
    name="regon",
    entity_type="REGON",
    pattern=re.compile(
        r"REGON[:\s]*(?P<value>\d{9}(?:\d{5})?)",
        re.IGNORECASE,
    ),
    validator=is_valid_regon,
    normaliser=digits_only,
    priority=30,
)

KRS = PatternRule(
    name="krs",
    entity_type="KRS",
    pattern=re.compile(r"KRS[:\s]*(?P<value>\d{10})", re.IGNORECASE),
    normaliser=digits_only,
    priority=30,
)

PESEL = PatternRule(
    name="pesel",
    entity_type="PESEL",
    pattern=re.compile(r"PESEL[:\s]*(?P<value>\d{11})", re.IGNORECASE),
    validator=is_valid_pesel,
    normaliser=digits_only,
    priority=30,
)

IBAN = PatternRule(
    name="iban",
    entity_type="IBAN",
    pattern=re.compile(
        r"\b(?P<value>[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?)\b"
    ),
    validator=is_valid_iban,
    normaliser=alphanumeric_upper,
    priority=25,
)

BANK_ACCOUNT = PatternRule(
    name="bank_account",
    entity_type="IBAN",
    pattern=re.compile(
        r"\b(?P<value>\d{2}(?:[ -]?\d{4}){6})\b",
    ),
    validator=is_valid_polish_account,
    normaliser=digits_only,
    priority=20,
)

INVOICE_NUMBER = PatternRule(
    name="invoice_number",
    entity_type="INVOICE_NO",
    pattern=re.compile(
        r"(?:faktur[ay]?(?:\s+VAT)?|FV|F-ra|rachunek)\s*"
        r"(?:nr\.?|numer|no\.?)?[:\s]*"
        r"(?P<value>[A-Z0-9][A-Z0-9/_.\-]{2,30}\d)",
        re.IGNORECASE,
    ),
    priority=25,
)

DOCUMENT_NUMBER = PatternRule(
    name="document_number",
    entity_type="DOC_NO",
    pattern=re.compile(
        r"(?:umow[ay]|zam[oó]wieni[ae]|zlecen[ia]e?|protok[oó][łl])\s*"
        r"(?:nr\.?|numer|no\.?)[:\s]*(?P<value>[A-Z0-9][A-Z0-9/_.\-]{1,29}[A-Z0-9])",
        re.IGNORECASE,
    ),
    priority=20,
)

AMOUNT = PatternRule(
    name="amount",
    entity_type="AMOUNT",
    pattern=re.compile(
        r"\b(?P<value>\d{1,3}(?:[ . ]\d{3})*(?:,\d{2})?|\d+,\d{2})\s*"
        r"(?:z[łl]|PLN|EUR|USD|GBP)\b",
        re.IGNORECASE,
    ),
    priority=10,
)

DATE = PatternRule(
    name="date",
    entity_type="DATE",
    pattern=re.compile(
        r"\b(?P<value>\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4}|\d{4}-\d{2}-\d{2})\b"
    ),
    priority=15,
)

EMAIL = PatternRule(
    name="email",
    entity_type="EMAIL",
    pattern=re.compile(r"\b(?P<value>[\w.+-]+@[\w-]+\.[\w.-]+)\b"),
    priority=15,
)

PHONE = PatternRule(
    name="phone",
    entity_type="PHONE",
    pattern=re.compile(
        r"(?:tel\.?|telefon|kom\.?)[:\s]*(?P<value>\+?\d[\d\s\-()]{7,17}\d)",
        re.IGNORECASE,
    ),
    priority=15,
)

POLISH_BUSINESS_RULES: tuple[PatternRule, ...] = (
    NIP,
    REGON,
    KRS,
    PESEL,
    IBAN,
    BANK_ACCOUNT,
    INVOICE_NUMBER,
    DOCUMENT_NUMBER,
    AMOUNT,
    DATE,
    EMAIL,
    PHONE,
)
"""Default rule set for Polish invoices, contracts and related paperwork."""
