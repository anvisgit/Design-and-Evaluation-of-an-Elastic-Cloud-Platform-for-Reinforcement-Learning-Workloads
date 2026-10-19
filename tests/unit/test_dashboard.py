from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_demo_mode_renders_sample_data():
    app_path = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception
    assert any("Log in from the sidebar." in element.value for element in app.info)

    app.toggle[0].set_value(True).run()

    assert not app.exception
    assert any("CartPole demo study" in str(element.value) for element in app.dataframe)
    assert any("PPO baseline" in str(element.value) for element in app.dataframe)
