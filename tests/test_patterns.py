"""Tests for the deterministic identifier layer.

The checksums are the point of this layer: without them a run of ten digits in a
document full of amounts and dates is indistinguishable from a tax identifier.
Each test therefore pairs a genuinely valid value with one differing by a single
digit.
"""

from __future__ import annotations

from nano_re.patterns import (
    PatternExtractor,
    is_valid_iban,
    is_valid_nip,
    is_valid_pesel,
    is_valid_polish_account,
    is_valid_regon,
)

VALID_NIP = "5252248481"
VALID_REGON_9 = "012365169"
VALID_PESEL = "44051401359"
VALID_IBAN = "PL61109010140000071219812874"


def test_nip_accepts_valid_and_rejects_tampered() -> None:
    """A real NIP passes and a single altered digit does not."""
    assert is_valid_nip(VALID_NIP)
    assert not is_valid_nip("5252248482")


def test_nip_rejects_wrong_length() -> None:
    """A NIP must be exactly ten digits."""
    assert not is_valid_nip("525224848")
    assert not is_valid_nip("52522484811")
    assert not is_valid_nip("")


def test_regon_accepts_nine_digit_form() -> None:
    """The nine digit REGON checksum is enforced."""
    assert is_valid_regon(VALID_REGON_9)
    assert not is_valid_regon("012365164")


def test_regon_rejects_unsupported_length() -> None:
    """Only the nine and fourteen digit forms exist."""
    assert not is_valid_regon("12345")


def test_pesel_accepts_valid_and_rejects_tampered() -> None:
    """The PESEL checksum is enforced."""
    assert is_valid_pesel(VALID_PESEL)
    assert not is_valid_pesel("44051401358")


def test_iban_accepts_valid_and_rejects_tampered() -> None:
    """The IBAN modulo 97 check is enforced."""
    assert is_valid_iban(VALID_IBAN)
    assert not is_valid_iban("PL61109010140000071219812875")


def test_bare_polish_account_validates_as_iban_body() -> None:
    """A Polish account without its country prefix still validates."""
    assert is_valid_polish_account(VALID_IBAN[2:])
    assert not is_valid_polish_account("12345678901234567890123456")


def test_extractor_finds_identifiers_with_word_offsets() -> None:
    """Every rule type is found and aligned onto word indices."""
    text = (
        f"Faktura VAT nr FV/2024/03/0192 z dnia 14.03.2024. "
        f"NIP {VALID_NIP}, REGON {VALID_REGON_9}, KRS 0000123456. "
        f"IBAN {VALID_IBAN}, kwota 12 450,00 PLN, biuro@firma.pl."
    )
    extractor = PatternExtractor()
    matches = extractor.align(extractor.extract(text), text)
    found = {match.entity_type: match.value for match in matches}

    assert found["NIP"] == VALID_NIP
    assert found["REGON"] == VALID_REGON_9
    assert found["IBAN"] == VALID_IBAN
    assert found["INVOICE_NO"] == "FV/2024/03/0192"
    assert found["DATE"] == "14.03.2024"
    assert found["EMAIL"] == "biuro@firma.pl"
    assert all(match.word_start >= 0 and match.word_end > match.word_start
               for match in matches)


def test_extractor_rejects_fabricated_identifiers() -> None:
    """Identifiers failing their checksum are not reported at all."""
    text = "NIP 5252248482, REGON 012365164, IBAN PL61109010140000071219812875"
    extractor = PatternExtractor()
    assert extractor.extract(text) == []


def test_extractor_normalises_separators() -> None:
    """A NIP written with dashes is reported in canonical digit form."""
    extractor = PatternExtractor()
    matches = extractor.extract("NIP 525-224-84-81")
    assert [match.value for match in matches] == [VALID_NIP]


def test_matches_do_not_overlap() -> None:
    """Overlapping rule matches are arbitrated down to one."""
    text = f"IBAN {VALID_IBAN} oraz konto {VALID_IBAN[2:]}"
    extractor = PatternExtractor()
    matches = extractor.extract(text)
    for earlier, later in zip(matches, matches[1:]):
        assert earlier.char_end <= later.char_start
