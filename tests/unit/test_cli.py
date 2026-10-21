"""Unit tests for CANExplain-QI CLI."""

from click.testing import CliRunner

from canexplain.cli import main


def test_cli_version():
    """Test that CLI outputs correct version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_audit():
    """Test audit command."""
    runner = CliRunner()
    result = runner.invoke(main, ["audit", "--dataset", "can-train-and-test-v1.5"])
    assert result.exit_code == 0
    assert "Auditing dataset" in result.output


def test_cli_verify_reqs():
    """Test verify-reqs command verifying requirements lines 8-30."""
    runner = CliRunner()
    result = runner.invoke(main, ["verify-reqs"])
    assert result.exit_code == 0
    assert "ALL REQUIREMENTS" in result.output
    assert "PASS" in result.output
