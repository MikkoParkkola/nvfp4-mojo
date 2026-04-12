# Project test runner that executes all assertion-backed suites.

from tests.test_deltanet import run_suite as run_deltanet_suite
from tests.test_nvfp4 import run_suite as run_nvfp4_suite


fn main() raises:
    print("=" * 60)
    print("NVFP4 Mojo Test Runner")
    print("=" * 60)

    run_nvfp4_suite()
    print()
    run_deltanet_suite()

    print("\n" + "=" * 60)
    print("All test suites passed!")
    print("=" * 60)
