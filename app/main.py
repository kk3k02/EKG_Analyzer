from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def _build_light_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#F7F9FC"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#1F2933"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#EEF2F7"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#1F2933"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#1F2933"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1F2933"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#1565C0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#D6E9FF"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#102A43"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7B8794"))
    palette.setColor(QPalette.ColorRole.Light, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor("#E4E7EB"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("#C7D0D9"))
    palette.setColor(QPalette.ColorRole.Dark, QColor("#9AA5B1"))
    palette.setColor(QPalette.ColorRole.Shadow, QColor("#616E7C"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#7B8794"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#7B8794"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#7B8794"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#E4E7EB"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor("#7B8794"))
    return palette


def _apply_light_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(_build_light_palette())
    app.setStyleSheet(
        """
        QWidget {
            color: #1F2933;
            background-color: #F7F9FC;
        }
        QMenuBar, QMenu, QStatusBar, QToolTip {
            background-color: #FFFFFF;
            color: #1F2933;
        }
        QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
        QListWidget, QTableWidget, QTabWidget::pane, QScrollArea, QGroupBox {
            background-color: #FFFFFF;
        }
        QPushButton, QToolButton {
            background-color: #FFFFFF;
        }
        """
    )


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("EKG Viewer")
    _apply_light_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
