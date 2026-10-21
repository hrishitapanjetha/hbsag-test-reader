# HBsAg Rapid Test Reader (Streamlit app)

Upload photographs of HBsAg rapid tests and get a report. For each photograph the app finds every device, reads the result (positive, negative or invalid), reads the barcode, handwritten ID and printed test name, flags anything a person should check, and produces a Word report, a PDF report, a CSV table and a ZIP of annotated photos.

## Run

    pip install -r requirements.txt
    ./run_app.sh

The app opens at http://localhost:8501. The first run downloads the handwriting reader (TrOCR) and loads the text reader, so it is slower.

## Files

- app.py: the Streamlit interface (always reads the barcode, handwritten ID and test type)
- pipeline.py: detection, result logic and reading (the same logic as the evaluated notebook)
- report.py: Word and PDF report builders
- hbsag_model_v2.pt: updated detection model
- samples/: four example photographs for a quick test

## Notes

- On macOS the barcode scanner needs the zbar library (`brew install zbar`). The app finds it in /opt/homebrew/lib automatically.
- A photograph is marked "needs review" when a device has no control line, two lines of the same type, devices disagree, the barcode is unreadable or does not match the handwriting, or the photograph is too dark, bright, small or blurred.
- Handwritten ID and test-name reading are weaker than result reading, so they are shown as secondary checks.
- The report is an aid for laboratory staff and is not a clinical validation.
