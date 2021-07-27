from Data import Data
from Layer import MovieRecommendation
from Model import Model
from config import Config

if __name__ == "__main__":
    # import numpy as np
    # import base64
    # a = np.array([1, 2, 3], dtype=np.float32)
    # np.savez('t', a=a, b=a)
    u_path = r"./data/movielens/ml-100k/u.user"
    item_path = r"./data/movielens/ml-100k/u.item"
    config = Config()

    train_rating_path = r"./data/movielens/ml-100k/u1.base"
    test_rating_path = r"./data/movielens/ml-100k/u1.test"
    train_data = Data(u_path=u_path, item_path=item_path, rating_path=train_rating_path, config=config)
    test_data = Data(u_path=u_path, item_path=item_path, rating_path=test_rating_path, config=config)

    network = MovieRecommendation(config=config)

    model = Model(net=network)

    model.compile(early_stop=50, device="cuda:0")

    model.fit(epochs=5000, dl_train=train_data, dl_val=test_data, batch_sie=256, check_path="./check_path")
