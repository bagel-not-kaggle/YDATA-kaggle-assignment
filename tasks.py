from invoke import task
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@task
def pipeline(c):
    """
    Runs the preprocessing and training scripts sequentially.
    """
    logger.info("Running preprocessing...")
    c.run("python preprocess.py", pty=True)
    logger.info("Preprocessing completed.")

    logger.info("Running training...")
    c.run("python train.py", pty=True)
    logger.info("Training completed.")

    logger.info("Pipeline finished successfully!")

@task
def serve(c):
    """
    Starts the FastAPI server
    """
    logger.info("Starting API server...")
    c.run("uvicorn app.main:app --reload", pty=True)