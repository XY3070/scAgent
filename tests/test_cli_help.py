#!/usr/bin/env python
"""Test that all CLI subcommands function with the --help flag."""

import subprocess
import pytest


SUBCOMMANDS = [
    "entrez",
    "sragent",
    "metadata",
    "tissue-ontology",
    "srx-info",
    "find-datasets",
]


def test_main_help():
    """Test that the main help command works."""
    result = subprocess.run(
        ["SRAgent", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "Subcommands" in result.stdout


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help(subcommand):
    """Test that each subcommand's help option works."""
    result = subprocess.run(
        ["SRAgent", subcommand, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    # Each subcommand should have its own usage section
    assert f"usage: SRAgent {subcommand}" in result.stdout
