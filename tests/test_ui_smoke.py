from __future__ import annotations

import unittest
from tkinter import TclError

from vrf_system.ui import FertilizerApp


class UISmokeTests(unittest.TestCase):
    def test_app_can_create_and_destroy(self) -> None:
        try:
            app = FertilizerApp(auto_load_models=False)
        except TclError as exc:
            self.skipTest(f"当前环境不可创建 tkinter 窗口：{exc}")
            return
        app.withdraw()
        app.update_idletasks()
        app.destroy()
