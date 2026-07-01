"""Fixed invoice schema — the single source of truth.

Reused everywhere: synthetic data generation, dataset preparation, guided/
constrained decoding at serve time, and evaluation. Do not redefine these
fields elsewhere; import from here.

Pydantic v2.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ISO 8601 date format we normalize everything to.
DATE_FMT = "%Y-%m-%d"

# A few input date formats we tolerate and coerce to DATE_FMT. Extend as needed;
# keep generation/eval in sync by importing this list.
_ACCEPTED_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y/%m/%d",
)


def _coerce_date(value: Optional[str]) -> Optional[str]:
    """Return an ISO YYYY-MM-DD string, or raise if unparseable.

    None passes through (nullable dates). Already-ISO strings validate cheaply.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime(DATE_FMT)
    s = str(value).strip()
    if not s:
        return None
    for fmt in _ACCEPTED_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime(DATE_FMT)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value!r}")


def _coerce_number(value) -> Optional[float]:
    """Parse a numeric field that may arrive as a string with separators/symbols.

    Examples: '1,488.00' -> 1488.0, '$1.234,56' -> 1234.56, '' -> None.
    Handles both '1,234.56' (US) and '1.234,56' (EU) thousands/decimal styles.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # Strip currency symbols and spaces, keep digits, separators, sign.
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return None
    # Decide decimal separator by the last-seen separator.
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")
    if last_comma > last_dot:
        # EU style: comma is decimal, dot is thousands.
        s = s.replace(".", "").replace(",", ".")
    else:
        # US style: dot is decimal, comma is thousands.
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Unrecognized number format: {value!r}")


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _numbers(cls, v):
        return _coerce_number(v)


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: str = Field(description="ISO 8601 YYYY-MM-DD")
    due_date: Optional[str] = Field(default=None, description="ISO 8601 or null")
    vendor_name: str
    vendor_tax_id: Optional[str] = None
    bill_to: Optional[str] = None
    currency: str = Field(description="ISO 4217, e.g. USD, EUR")
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total: float
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def _dates(cls, v):
        return _coerce_date(v)

    @field_validator("subtotal", "tax_amount", "total", mode="before")
    @classmethod
    def _numbers(cls, v):
        return _coerce_number(v)

    @field_validator("currency")
    @classmethod
    def _currency(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError(f"currency must be a 3-letter ISO 4217 code, got {v!r}")
        return v


def json_schema() -> dict:
    """Return the JSON Schema for guided/constrained decoding at serve time."""
    return Invoice.model_json_schema()


if __name__ == "__main__":
    # Smoke test with messy-but-valid inputs.
    sample = Invoice(
        invoice_number="INV-2024-0007",
        invoice_date="March 3, 2024",
        due_date=None,
        vendor_name="Acme Corp",
        vendor_tax_id=None,
        bill_to="Globex LLC",
        currency="usd",
        subtotal="1,488.00",
        tax_amount="119.04",
        total="1,607.04",
        line_items=[
            {"description": "Widget", "quantity": "2", "unit_price": "744.00", "amount": "1,488.00"},
        ],
    )
    print(sample.model_dump_json(indent=2))
