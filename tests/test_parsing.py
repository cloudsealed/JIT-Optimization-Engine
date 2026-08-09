from datetime import date

import pytest

from cloudsealed_jit.parsing import ParseError, parse_billing_csv


AWS_CUR = """\
identity/LineItemId,lineItem/UsageStartDate,lineItem/ProductCode,product/ProductName,lineItem/UnblendedCost,lineItem/CurrencyCode
1,2026-01-01T00:00:00Z,AmazonEC2,Amazon Elastic Compute Cloud,10.00,USD
2,2026-01-01T00:00:00Z,AmazonRDS,Amazon Relational Database Service,5.00,USD
3,2026-01-02T00:00:00Z,AmazonEC2,Amazon Elastic Compute Cloud,12.00,USD
4,2026-01-03T00:00:00Z,AmazonEC2,Amazon Elastic Compute Cloud,11.00,USD
"""

GCP = """\
billing_account_id,service.description,usage_start_time,cost,currency
X,Compute Engine,2026-02-01 00:00:00 UTC,20.5,BRL
X,Cloud Storage,2026-02-01 00:00:00 UTC,3.5,BRL
X,Compute Engine,2026-02-02 00:00:00 UTC,21.0,BRL
X,Compute Engine,2026-02-04 00:00:00 UTC,22.0,BRL
"""

# FOCUS 1.0 — the vendor-neutral spec AWS/GCP/Azure/OCI export natively.
# BilledCost is the invoiced amount; ChargePeriodStart is the granular date.
FOCUS = """\
ChargePeriodStart,ChargePeriodEnd,ServiceName,ServiceCategory,BilledCost,EffectiveCost,BillingCurrency
2026-03-01T00:00:00Z,2026-03-02T00:00:00Z,Virtual Machines,Compute,40.00,38.00,USD
2026-03-01T00:00:00Z,2026-03-02T00:00:00Z,Object Storage,Storage,5.00,5.00,USD
2026-03-02T00:00:00Z,2026-03-03T00:00:00Z,Virtual Machines,Compute,42.00,40.00,USD
2026-03-03T00:00:00Z,2026-03-04T00:00:00Z,Virtual Machines,Compute,41.00,39.00,USD
"""


def test_aws_cur_aggregates_line_items_per_day():
    series = parse_billing_csv(AWS_CUR)

    assert series.days == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
    # Two line items on day one must sum, not count as two observations.
    assert series.costs[0] == pytest.approx(15.0)
    assert series.rows_parsed == 4
    assert series.rows_skipped == 0
    assert series.currency == "USD"


def test_service_attribution_is_aligned_with_days():
    series = parse_billing_csv(AWS_CUR)
    ec2 = series.by_service["Amazon Elastic Compute Cloud"]

    assert len(ec2) == len(series.days)
    assert ec2 == pytest.approx([10.0, 12.0, 11.0])


def test_gcp_export_fills_missing_days_with_zero():
    series = parse_billing_csv(GCP)

    # 2026-02-03 has no line items and must appear as a zero-spend day.
    assert series.days == [
        date(2026, 2, 1), date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)
    ]
    assert series.costs[2] == 0.0
    assert series.currency == "BRL"


def test_focus_export_uses_billed_cost_and_charge_period():
    series = parse_billing_csv(FOCUS)

    assert series.days == [date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)]
    # Day one: 40.00 (VM) + 5.00 (storage) of BilledCost, not EffectiveCost.
    assert series.costs[0] == pytest.approx(45.0)
    assert series.currency == "USD"


def test_focus_attributes_cost_by_service_name():
    series = parse_billing_csv(FOCUS)
    vms = series.by_service["Virtual Machines"]

    assert len(vms) == len(series.days)
    assert vms == pytest.approx([40.0, 42.0, 41.0])


def test_decimal_comma_is_parsed():
    csv_text = "date,cost\n2026-03-01,\"1.234,56\"\n2026-03-02,\"2.000,00\"\n"
    series = parse_billing_csv(csv_text)
    assert series.costs[0] == pytest.approx(1234.56)
    assert series.costs[1] == pytest.approx(2000.00)


def test_cost_center_column_is_not_mistaken_for_cost():
    csv_text = "date,costCenter,amount\n2026-03-01,ENG-42,10.0\n2026-03-02,ENG-42,12.0\n"
    series = parse_billing_csv(csv_text)
    assert series.costs == pytest.approx([10.0, 12.0])


def test_unparseable_rows_are_counted_not_dropped_silently():
    csv_text = "date,cost\n2026-03-01,10.0\nnot-a-date,oops\n2026-03-02,12.0\n"
    series = parse_billing_csv(csv_text)
    assert series.rows_parsed == 2
    assert series.rows_skipped == 1


def test_empty_export_raises():
    with pytest.raises(ParseError):
        parse_billing_csv("")


def test_missing_cost_column_raises():
    with pytest.raises(ParseError, match="date and a cost column"):
        parse_billing_csv("day,region\n2026-01-01,us-east-1\n")
