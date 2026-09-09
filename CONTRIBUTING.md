# Contributing to ASL Vision

Thank you for your interest in contributing to **ASL Vision**!

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/HarshitaSmriti/asl-vision.git
   cd asl-vision
   ```
3. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Development Guidelines

- **Architecture Integrity**: The trained 95-class `ASLTransformer` model checkpoint and 696-dimensional velocity feature pipeline are fixed. Do not alter checkpoint weights or model input dimensions without matching preprocessing.
- **Testing**: Run `python backend/test_pipeline.py` and `python backend/test_e2e.py` before submitting a Pull Request.
- **Frontend Code Quality**: Ensure `npm run build` succeeds without warnings.

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/amazing-feature`).
2. Commit your changes (`git commit -m 'feat: add amazing feature'`).
3. Push to the branch (`git push origin feature/amazing-feature`).
4. Open a Pull Request on GitHub.
