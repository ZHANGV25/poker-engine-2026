# PokerBotEngine2025

## How to run the engine

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source .venv/bin/activate
     ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

```bash
pytest --cov=gym_env --cov-report=term-missing --cov-report=html --cov-branch
```
