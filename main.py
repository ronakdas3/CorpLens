"""
CorpLens Main Entry Point
Runs the Streamlit Web Application.
"""

import sys
import subprocess


def main():
    print("Launching CorpLens Streamlit Application...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

