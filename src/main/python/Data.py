import numpy as np
from typing import List, Tuple, Dict, Union
from config import Config
import random


class Data(object):

    def __init__(self, u_path: str, item_path: str, rating_path: str, config: Config):
        """

        :param u_path: 用户特征数据
        :param item_path: 物品特征数据
        :param rating_path: 评分特征数据
        """

        self.config = config
        self.user_info_dict = self._read_user_info(u_path=u_path)
        self.item_dict = self._read_item_dict(item_path=item_path)
        self.rating_list = self._read_rating_dict(rating_path=rating_path)

        self.instance_num = len(self.rating_list)
        self.index_list = list(range(self.instance_num))
        self.point = 0

    def shuffle(self):
        self.point = 0
        random.shuffle(self.index_list)

    def get_user_info(self) -> Dict[str, np.ndarray]:
        """
        返回所有用户特征
        :return:
        """
        length = len(self.user_info_dict)
        batch_user_id = np.zeros((length,), dtype=np.int64)
        batch_age = np.zeros((length, 1), dtype=np.float32)
        batch_occupation = np.zeros((length, self.config.occupation_num), dtype=np.float32)
        index = 0
        for user_id, (age, sex, occupation) in self.user_info_dict.items():
            age = min([1, age / 100])
            occupation_one_hot = self._occupation_one_hot(occupation=occupation)
            batch_user_id[index] = user_id
            batch_age[index][0] = age
            batch_occupation[index] = np.array(occupation_one_hot)
            index += 1
        return {"user_id": batch_user_id,
                "age": batch_age,
                "occupation": batch_occupation}

    def get_item_info(self) -> Dict[str, np.ndarray]:
        """
        返回所有物品特征
        :return:
        """
        length = len(self.item_dict)
        batch_item_id = np.zeros((length,), dtype=np.int64)
        batch_detail = np.zeros((length, self.config.style_num), dtype=np.float32)
        index = 0
        for item_id, detail in self.item_dict.items():
            batch_item_id[index] = item_id
            batch_detail[index] = np.array(detail)
            index += 1
        return {"item_id": batch_item_id,
                "detail": batch_detail}

    def next(self, batch_size: int) -> Union[Dict[str, np.ndarray], None]:

        if self.point + batch_size > self.instance_num:
            return None

        batch_index_list = self.index_list[self.point: self.point + batch_size]
        length = len(batch_index_list)

        batch_user_id = np.zeros((length,), dtype=np.int64)
        batch_item_id = np.zeros((length,), dtype=np.int64)
        batch_age = np.zeros((length, 1), dtype=np.float32)
        batch_occupation = np.zeros((length, self.config.occupation_num), dtype=np.float32)
        batch_detail = np.zeros((length, self.config.style_num), dtype=np.float32)
        batch_rating = np.zeros((length,), dtype=np.float32)
        for i in range(len(batch_index_list)):
            user_id, item_id, rating = self.rating_list[batch_index_list[i]]
            age, sex, occupation = self.user_info_dict[user_id]
            age = min([1, age / 100])
            occupation_one_hot = self._occupation_one_hot(occupation=occupation)
            detail = self.item_dict[item_id]
            batch_user_id[i] = user_id
            batch_item_id[i] = item_id
            batch_age[i][0] = age
            batch_occupation[i] = np.array(occupation_one_hot)
            batch_detail[i] = np.array(detail)
            batch_rating[i] = rating / 5

        self.point = self.point + batch_size

        return {"user_id": batch_user_id,
                "item_id": batch_item_id,
                "age": batch_age,
                "occupation": batch_occupation,
                "detail": batch_detail,
                "rating": batch_rating}

    def _occupation_one_hot(self, occupation: int) -> List[int]:
        occupation_one_hot = [0] * self.config.occupation_num
        occupation_one_hot[occupation] = 1
        return occupation_one_hot

    def _read_user_info(self, u_path: str) -> Dict[int, Tuple[int, int, int]]:
        """

        :param u_path:
        :return:
        """
        user_info_dict: Dict[int, Tuple[int, int, int]] = {}
        for line in open(u_path, "r", encoding="utf-8"):
            info = line.strip().split("|")
            user_id = int(info[0])
            age = int(info[1])
            sex = self.config.gender_dict[info[2]]
            occupation = self.config.occupation_dict[info[3]]
            user_info_dict[user_id] = (age, sex, occupation)
        return user_info_dict

    @staticmethod
    def _read_item_dict(item_path: str) -> Dict[int, List[int]]:
        item_dict: Dict[int, List[int]] = {}
        for line in open(item_path, "r", encoding="ISO-8859-1"):
            info = line.strip().split("|")
            item_id = int(info[0])
            detail = [int(i) for i in info[5:]]
            item_dict[item_id] = detail
        return item_dict

    @staticmethod
    def _read_rating_dict(rating_path: str) -> List[Tuple[int, int, int]]:
        rating_list = []
        for line in open(rating_path, "r", encoding="utf-8"):
            info = line.strip().split("\t")
            user_id = int(info[0])
            item_id = int(info[1])
            rating = int(info[2])
            rating_list.append((user_id, item_id, rating))
        return rating_list

# if __name__ == "__main__":
#     u_path = r"./data/movielens/ml-100k/u.user"
#     item_path = r"./data/movielens/ml-100k/u.item"
#     rating_path = r"./data/movielens/ml-100k/u.data"
#     config = Config()
#     data = Data(u_path=u_path, item_path=item_path, rating_path=rating_path, config=config)
#
#     while True:
#         batch_data = data.next(batch_size=512)
#         if batch_data is None:
#             break
#         for key, v in batch_data.items():
#             print("key:{0}, shape:{1}".format(key, np.shape(v)))
#         print("####################")
