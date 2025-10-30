"""
Wrapper module to launch the Streamlit NBA dashboard.

This file acts as the entry point for Streamlit Cloud.  Many hosting
platforms (including streamlit.io) look specifically for
`streamlit_app.py` in the repository root.  By delegating to the
`app.main()` function we keep the core logic in `app.py` while
providing the expected entry module.
"""
from app import main


if __name__ == "__main__":
    main()