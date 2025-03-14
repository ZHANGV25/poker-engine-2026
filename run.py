import logging
import multiprocessing

from agents.test_agents import AllInAgent
from match import run_api_match
from submission.player import PlayerAgent


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    process0 = multiprocessing.Process(target=AllInAgent.run, args=(False, 8000))
    process1 = multiprocessing.Process(target=PlayerAgent.run, args=(True, 8001))

    process0.start()
    process1.start()

    logger.info("Starting API-based match")
    # When running run.py by itself, just write match.csv locally:
    result = run_api_match(
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        logger,
        csv_path="./match.csv",
    )
    logger.info(f"Match result: {result}")

    # Clean up processes
    process0.terminate()
    process1.terminate()
    process0.join()
    process1.join()


if __name__ == "__main__":
    main()
