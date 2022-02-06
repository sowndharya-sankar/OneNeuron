import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import numpy as np
import joblib


def main(data, eta, epochs, modelfilename, plotfilename):
    df = pd.DataFrame(data)

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
    main(data=AND, eta=ETA, epochs=EPOCHS, modelfilename="and.model", plotfilename="and.png")
