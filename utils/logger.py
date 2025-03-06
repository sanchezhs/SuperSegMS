from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO
from logging import Logger
import sys

def get_logger(name: str, level: int = INFO) -> Logger:
    logger = getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to parent loggers
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    handler = StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    logger = get_logger("test_logger", level=DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")