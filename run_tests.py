#!/usr/bin/env python3
"""
Test runner script for Copenhagen Event Recommender.
Provides comprehensive test execution with coverage reporting.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime


def setup_test_environment():
    """Setup test environment and dependencies."""
    print("Setting up test environment...")

    # Add project paths to Python path
    project_root = Path(__file__).parent
    backend_path = project_root / "backend"
    ml_path = project_root / "ml"
    data_collection_path = project_root / "data-collection"

    paths_to_add = [str(backend_path), str(ml_path), str(data_collection_path)]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Set environment variables for testing
    os.environ["PYTHONPATH"] = ":".join(paths_to_add)
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "test_events.duckdb"
    os.environ["DISABLE_TORCH"] = "true"  # Disable torch for testing

    print(f"[OK] Test environment setup complete")
    print(f"   Python path includes: {len(paths_to_add)} project directories")


def run_pytest_command(args, test_path="tests", coverage=True, verbose=True):
    """Run pytest with specified arguments."""

    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=backend",
                "--cov=ml",
                "--cov=data-collection",
                "--cov-report=html:test_reports/coverage_html",
                "--cov-report=json:test_reports/coverage.json",
                "--cov-report=term-missing",
            ]
        )

    # Add test output formatting
    cmd.extend(
        [
            "--tb=short",
            "--color=yes",
            f"--html=test_reports/report.html",
            "--self-contained-html",
        ]
    )

    # Add test path
    cmd.append(test_path)

    # Add any additional pytest args
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=False)


def run_specific_tests(test_category):
    """Run tests for specific category."""

    test_files = {
        "api": "tests/test_api.py",
        "ml": "tests/test_ml_models.py",
        "data": "tests/test_data_collection.py",
        "database": "tests/test_database.py",
        "all": "tests",
    }

    if test_category not in test_files:
        print(f"[ERROR] Unknown test category: {test_category}")
        print(f"   Available categories: {', '.join(test_files.keys())}")
        return 1

    test_path = test_files[test_category]

    print(f"[TEST] Running {test_category} tests...")
    print(f"   Test path: {test_path}")

    return run_pytest_command([], test_path=test_path)


def run_quick_tests():
    """Run quick smoke tests."""
    print("[QUICK] Running quick smoke tests...")

    quick_args = [
        "-m",
        "not slow",  # Skip slow tests
        "--maxfail=5",  # Stop after 5 failures
        "-x",  # Stop on first failure
    ]

    return run_pytest_command(quick_args, coverage=False)


def run_full_test_suite():
    """Run complete test suite with full coverage."""
    print("[FULL] Running full test suite...")

    # Create test reports directory
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)

    full_args = [
        "--durations=10",  # Show 10 slowest tests
        "--strict-markers",  # Ensure all markers are defined
    ]

    result = run_pytest_command(full_args)

    if result.returncode == 0:
        print("[OK] All tests passed!")
        print(f"   Coverage report: test_reports/coverage_html/index.html")
        print(f"   Test report: test_reports/report.html")
    else:
        print("[ERROR] Some tests failed")
        print(f"   Check reports in: test_reports/")

    return result


def run_performance_tests():
    """Run performance and load tests."""
    print("[PERF] Running performance tests...")

    perf_args = ["-m", "performance", "--durations=0"]  # Show all test durations

    return run_pytest_command(perf_args, coverage=False)


def run_security_tests():
    """Run security-focused tests."""
    print("[SECURITY] Running security tests...")

    security_args = ["-m", "security", "-v"]

    return run_pytest_command(security_args, coverage=False)


def check_test_dependencies():
    """Check if all test dependencies are available."""
    print("[CHECK] Checking test dependencies...")

    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-html",
        "pytest-asyncio",
        "requests",
        "duckdb",
        "fastapi",
        "jwt",
        "geopy",
    ]
    
    # Only check torch if not disabled
    if not os.getenv("DISABLE_TORCH"):
        required_packages.append("torch")

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"[ERROR] Missing test dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\nInstall with: pip install {' '.join(missing_packages)}")
        return False

    print("[OK] All test dependencies available")
    return True


def generate_test_summary():
    """Generate test execution summary."""

    reports_dir = Path("test_reports")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_reports_generated": [],
        "coverage_available": False,
        "html_report_available": False,
    }

    # Check for coverage report
    coverage_json = reports_dir / "coverage.json"
    if coverage_json.exists():
        try:
            with open(coverage_json, "r") as f:
                coverage_data = json.load(f)
                summary["coverage_available"] = True
                summary["total_coverage"] = coverage_data.get("totals", {}).get(
                    "percent_covered", 0
                )
        except Exception as e:
            print(f"Warning: Could not parse coverage report: {e}")

    # Check for HTML report
    html_report = reports_dir / "report.html"
    if html_report.exists():
        summary["html_report_available"] = True

    # Save summary
    summary_path = reports_dir / "test_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SUMMARY] Test summary saved: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save test summary: {e}")

    return summary


def print_usage():
    """Print usage information."""
    print(
        """
Copenhagen Event Recommender - Test Runner

Usage: python run_tests.py [command] [options]

Commands:
  all            Run all tests with full coverage (default)
  quick          Run quick smoke tests
  api            Run API tests only
  ml             Run ML model tests only  
  data           Run data collection tests only
  database       Run database tests only
  performance    Run performance tests only
  security       Run security tests only
  deps           Check test dependencies

Options:
  --no-coverage  Skip coverage reporting
  --help, -h     Show this help message

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py quick              # Quick smoke test
  python run_tests.py api                # API tests only
  python run_tests.py ml --no-coverage   # ML tests without coverage
  python run_tests.py deps               # Check dependencies
"""
    )


def main():
    """Main test runner entry point."""

    parser = argparse.ArgumentParser(
        description="Copenhagen Event Recommender Test Runner"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "all",
            "quick",
            "api",
            "ml",
            "data",
            "database",
            "performance",
            "security",
            "deps",
        ],
        help="Test command to run",
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage reporting"
    )

    args = parser.parse_args()

    print("Copenhagen Event Recommender - Test Runner")
    print("=" * 50)

    # Setup test environment
    setup_test_environment()

    # Check dependencies first
    if not check_test_dependencies():
        return 1

    # Run specified command
    if args.command == "deps":
        return 0  # Dependencies already checked
    elif args.command == "quick":
        result = run_quick_tests()
    elif args.command == "performance":
        result = run_performance_tests()
    elif args.command == "security":
        result = run_security_tests()
    elif args.command in ["api", "ml", "data", "database"]:
        result = run_specific_tests(args.command)
    else:  # 'all' or default
        result = run_full_test_suite()

    # Generate summary
    if args.command == "all":
        summary = generate_test_summary()

        print("\n[SUMMARY] Test Execution Summary:")
        print(f"   Coverage available: {summary['coverage_available']}")
        if summary["coverage_available"]:
            print(f"   Total coverage: {summary.get('total_coverage', 0):.1f}%")
        print(f"   HTML report available: {summary['html_report_available']}")

    return result.returncode if result else 0


if __name__ == "__main__":
    sys.exit(main())
