from __future__ import annotations

import unittest

from smoke import _can_run_runtime_smoke
from targets import Target


class RuntimeSmokeEligibilityTest(unittest.TestCase):
    def test_runs_only_when_host_image_and_binary_architectures_match(self) -> None:
        cases = [
            (Target.LINUX_ARM64, "aarch64", True),
            (Target.LINUX_ARM64, "arm64", True),
            (Target.LINUX_ARM64, "x86_64", False),
            (Target.LINUX_X64, "aarch64", False),
            (Target.LINUX_X64, "x86_64", False),
        ]

        for target, host_machine, expected in cases:
            with self.subTest(target=target, host_machine=host_machine):
                self.assertEqual(
                    _can_run_runtime_smoke(target, host_machine),
                    expected,
                )

    def test_android_never_runs_without_a_device(self) -> None:
        self.assertFalse(_can_run_runtime_smoke(Target.ANDROID_ARM64, "aarch64"))

    def test_unknown_host_architecture_skips_runtime_execution(self) -> None:
        self.assertFalse(_can_run_runtime_smoke(Target.LINUX_ARM64, "unknown"))


if __name__ == "__main__":
    unittest.main()
