import os

import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import numpy as np
import joblib
import logging

logging_str = "[%(asctime)s:%(levelname)s:%(module)s]%(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "app_logs.log"),level=logging.INFO, format=logging_str,
                    filemode="a")


def main(data, eta, epochs, modelfilename, plotfilename):
    df = pd.DataFrame(data)
    logging.info("This is actual dataframe{df}")
    X, y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, modelfilename)
    save_plot(df, plotfilename, model)


if __name__ == '__main__':
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1],
    }
    ETA = 0.3  # 0 and 1
    EPOCHS = 10
    try:
        logging.info("<<<< started training <<<<<")
        main(data=AND, eta=ETA, epochs=EPOCHS, modelfilename="and.model", plotfilename="and.png")
        logging.info(">>>> training completed >>>>>")
    except Exception as e:
        logging.exception(e)
