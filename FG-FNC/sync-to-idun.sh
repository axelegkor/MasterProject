#!/bin/bash
rsync -av --delete \
  --filter='protect venv/' \
  --filter='protect Makefile/' \
  --filter='protect LICENSE/' \
  --filter='protect logs/' \
  --exclude=venv \
  --exclude=__pycache__/ \
  --exclude="*.pyc" \
  --exclude=".git" \
  --exclude="Makefile" \
  --exclude="venv/" \
  --exclude="sync-to-idun.sh" \
  ~/dev/MasterProject/FG-FNC/ \
  axelle@idun-login2.hpc.ntnu.no:/cluster/home/axelle/FG-FNC
