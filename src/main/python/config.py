class Config(object):
    def __init__(self):
        # 职业
        self.occupation_dict = {"administrator": 0,
                                "artist": 1,
                                "doctor": 2,
                                "educator": 3,
                                "engineer": 4,
                                "entertainment": 5,
                                "executive": 6,
                                "healthcare": 7,
                                "homemaker": 8,
                                "lawyer": 9,
                                "librarian": 10,
                                "marketing": 11,
                                "none": 12,
                                "other": 13,
                                "programmer": 14,
                                "retired": 15,
                                "salesman": 16,
                                "scientist": 17,
                                "student": 18,
                                "technician": 19,
                                "writer": 20}
        self.occupation_num = len(self.occupation_dict)
        # 风格
        self.item_style = {"unknown": 0,
                           "Action": 1,
                           "Adventure": 2,
                           "Animation": 3,
                           "Children's": 4,
                           "Comedy": 5,
                           "Crime": 6,
                           "Documentary": 7,
                           "Drama": 8,
                           "Fantasy": 9,
                           "Film-Noir": 10,
                           "Horror": 11,
                           "Musical": 12,
                           "Mystery": 13,
                           "Romance": 14,
                           "Sci-Fi": 15,
                           "Thriller": 16,
                           "War": 17,
                           "Western": 18}
        self.style_num = len(self.item_style)

        self.gender_dict = {"M": 1, "F": 0}

        self.class_num = 1

        self.user_num = 1000
        self.item_num = 2000

        self.user_id_embedding_size = 16
        self.item_id_embedding_size = 16

        self.user_feature_dim = 32
        self.item_feature_dim = 32
