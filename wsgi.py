"""
wsgi.py — WSGI entrypoint untuk PythonAnywhere
================================================
Cara pakai:
1. Clone repo ke /home/<username>/chatbotweb
2. Bikin virtualenv, install requirements
3. Copy file ini ke /var/www/<username>_pythonanywhere_com_wsgi.py
   atau setting di Web tab → Code → WSGI configuration file
"""

import os
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
if PATH not in sys.path:
    sys.path.append(PATH)

# ── Environment variables (set via Web tab, bukan di sini) ──
# FIREBASE_CREDENTIALS — base64 JSON Firebase service account
# SECRET_KEY — Flask session key
# HF_HUB_DISABLE_SYMLINKS — disable symlinks warning

from app import app as application
