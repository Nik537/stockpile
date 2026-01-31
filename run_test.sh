#!/bin/bash
cd /Users/niknoavak/Desktop/YT/stockpile
rm -f src/broll_processor.log /tmp/stockpile_run.log
/Users/niknoavak/Desktop/YT/stockpile/.venv/bin/python /Users/niknoavak/Desktop/YT/stockpile/run_with_preferences.py > /tmp/stockpile_run.log 2>&1
