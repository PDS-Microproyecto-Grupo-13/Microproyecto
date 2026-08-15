"""Foorilla data ingestion package.

Scope: pull job/salary data from the Foorilla API, validate it, and write
it out as a dated CSV that's handed off to whoever owns data versioning.
That handoff, and everything downstream (EDA, feature engineering,
training), is out of scope for this module.
"""
