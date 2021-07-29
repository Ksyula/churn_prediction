import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.dataprocessing import Dataprocesser
from src.data.utils import request_raw_data_by_url
from config import Config


def main(config: dict):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready for model training (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # Request raw data
    request_raw_data_by_url(config["input_url"], config["raw_data_path"])
    # Process raw data
    dp = Dataprocesser(config["raw_data_path"], config["data_processed_path"])
    dp.process_data()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print("project_dir: ", project_dir)

    config = Config()
    main(config.input_dict)
