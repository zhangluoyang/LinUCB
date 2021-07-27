package src.simulation;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import src.arm.BandArmLearning;
import src.arm.LinUCBLearning;
import src.feature.impl.ItemMlFeature;
import src.feature.impl.UserMlFeature;
import src.instance.Item;
import src.instance.User;
import src.utils.U;

import java.io.*;
import java.util.*;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;


public class MovieSimulation {

    /**
     * 应用于模拟离线计算逻辑 离线仿真 多臂老虎机进行选择计算
     */

    private BandArmLearning armLearning;

    /**
     * 每次返回 topK 个数据进行反馈
     */
    private final int topK;

    /**
     * 物品
     */
    private Map<Long, Item> itemMap;

    /**
     * 物品
     */
    private Map<Long, ItemMlFeature> itemMlMap;

    /**
     * 用户
     */
    private Map<Long, User> userMap;

    /**
     * 用户
     */
    private Map<Long, UserMlFeature> userMlMap;

    /**
     * 用户 物品
     */
    private Map<Long, List<Long>> userItemMap;

    /**
     * 用户 - 物品 评分
     */
    private Map<Long, Map<Long, Float>> ratingMap;

    /**
     * 下标 - 物品 映射
     */
    Map<Integer, ItemMlFeature> itemSliceMlMap;

    /**
     * 下标 - 用户 映射
     */
    Map<Integer, UserMlFeature> userSliceMlMap;


    private MovieSimulation(BandArmLearning armLearning,
                            Map<Integer, ItemMlFeature> itemSliceMlMap,
                            Map<Integer, UserMlFeature> userSliceMlMap,
                            Map<Long, Item> itemMap,
                            Map<Long, User> userMap,
                            Map<Long, List<Long>> userItemMap,
                            Map<Long, Map<Long, Float>> ratingMap,
                            Map<Long, UserMlFeature> userMlMap,
                            Map<Long, ItemMlFeature> itemMlMap,
                            int topK) {
        this.armLearning = armLearning;
        this.itemSliceMlMap = itemSliceMlMap;
        this.userSliceMlMap = userSliceMlMap;
        this.itemMap = itemMap;
        this.userMap = userMap;
        this.userItemMap = userItemMap;
        this.ratingMap = ratingMap;
        this.userMlMap = userMlMap;
        this.itemMlMap = itemMlMap;
        this.topK = topK;
    }

    /**
     * 返回 p向量最大值的下标
     */
    private Integer calArgMax(INDArray p) {
        float maxValue = Nd4j.max(p).getFloat(0);
        int[] binary = p.eq(maxValue).toIntVector();

        List<Integer> indexList = new ArrayList<>();
        for (int i = 0; i < binary.length; i++) {
            if (binary[i] == 1) {
                indexList.add(i);
            }
        }
        if (indexList.size() > 0) {
            Collections.shuffle(indexList);
            return indexList.get(0);
        }
        return Nd4j.argMax(p).getInt(0);
    }

    /**
     * 具体的多臂老虎机学习过程 序列化最终的结果
     * 这里每一个epoch过程 每一个用户都会进行一次 sample 更新 reward
     */
    public void run(int epoch, float alpha) {
        System.out.println("start training....");
        System.out.println("userNum: " + userMap.size());
        System.out.println("itemNum: " + itemMap.size());
        boolean debug = true;
        long cUserId = 18L;
        for (int e = 0; e < epoch; e++) {
            float positiveReward = 0;
            int count = 0;
            // 遍历每一个用户
            long startTime = System.currentTimeMillis();
            List<Long> userIds = new ArrayList<>(this.userMap.keySet());
            Collections.shuffle(userIds);
            for (long userId : userIds) {
                UserMlFeature userMl = this.userMlMap.get(userId);
                // 计算该用户选择每一个arm的概率
                INDArray p = this.armLearning.calUcbScore(userMl.getSliceIndex(), alpha);
                // 选择最大值
                int[] indexList = U.argTopK(p, 128);
                for (int i = 0; i < indexList.length; i++) {
                    int argmaxIndex = indexList[i];
                    // 选择的商品
                    ItemMlFeature itemMl = this.itemSliceMlMap.get(argmaxIndex);
                    // 计算实际评分值
                    assert itemMl != null;
                    float reward = this.calReward(userId, itemMl.getItemId());
                    // 更新A b
                    this.armLearning.update(userMl.getSliceIndex(), itemMl.getSliceIndex(), reward);
                    if (i == 0) {
                        // 仅仅计算具有正反馈的奖赏值
                        positiveReward += reward;
                        count++;
                    }
                }
            }
            long endTime = System.currentTimeMillis();
            System.out.println("epoch: " + e + " positiveReward:"
                    + positiveReward / count +
                    " time: " + (int) (endTime - startTime) / 1000);
        }

    }

    /**
     * 推荐物品的reward
     *
     * @param userId
     * @param itemId
     */
    public float calReward(long userId, long itemId) {
        Map<Float, Float> scoreRewardMap = new HashMap<>();
        scoreRewardMap.put(1f, -1f);
        scoreRewardMap.put(2f, -0.5f);
        scoreRewardMap.put(2.5f, 0.0f);
        scoreRewardMap.put(3f, 0.5f);
        scoreRewardMap.put(3.5f, 1f);
        scoreRewardMap.put(4f, 5f);
        scoreRewardMap.put(5f, 10f);
        // 如果是用户已经购买过的物品 根据评分值
        if (this.ratingMap.containsKey(userId) &&
                this.ratingMap.get(userId).containsKey(itemId)) {
            float rating = this.ratingMap.get(userId).get(itemId);
            return scoreRewardMap.get(rating);
        }

        if (!this.userItemMap.containsKey(userId)) {
            this.userItemMap.put(userId, new ArrayList<>());
            this.ratingMap.put(userId, new HashMap<>());
        }
        List<Long> userBuyAllItems = this.userItemMap.get(userId);

        // 属性相同的次数
        float styleLikeNums = 0;
        // 总次数
        float styleCounts = 0.001f;

        Item item = this.itemMap.get(itemId);

        for (long buyItemId : userBuyAllItems) {
            float rating = this.ratingMap.get(userId).get(buyItemId);
            if (rating >= 4) {
                for (String style : item.getStyles()) {
                    // 正反馈电影里面风格相同
                    if (this.itemMap.get(buyItemId).getStyles().contains(style)) {
                        styleLikeNums += 1;
                        styleCounts += 1;
                    }
                }
            } else {
                for (String style : item.getStyles()) {
                    // 负反馈电影里面风格相同
                    if (this.itemMap.get(buyItemId).getStyles().contains(style)) {
                        styleLikeNums -= 1;
                        styleCounts += 1;
                    }
                }
            }
        }

        float styleLikeRatio = styleLikeNums / styleCounts;
        styleLikeRatio = Math.max(0, styleLikeRatio);
        // 贝努利采样 决定奖赏值
        BinomialDistribution distribution = new BinomialDistribution(1, styleLikeRatio);
        int sampleRating = distribution.sample();
        this.userItemMap.get(userId).add(item.getItemId());
        if (sampleRating == 1) {
            this.ratingMap.get(userId).put(item.getItemId(), 3f);
            return scoreRewardMap.get(3.5f);
        } else {
            this.ratingMap.get(userId).put(item.getItemId(), 2.5f);
            return scoreRewardMap.get(2.5f);
        }
    }

    /**
     * 没见过物品的 特征值模型计算 物以类聚
     * */

    /**
     * 没见过用户的 特征值模型计算 人以群分
     */
    public static String basePath = "C:/Users/zhangluoyang/Desktop/LinUCB/src/main/resources";

    public static void readGenreMap(Map<Integer, String> genreMap) {
        String path = basePath + "/u.genre";
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(new File(path)));
            String line = null;
            while ((line = reader.readLine()) != null) {
                String[] lineSplits = line.split("\\|");
                genreMap.put(Integer.parseInt(lineSplits[1]), lineSplits[0]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void readRatingMap(Map<Long, Map<Long, Float>> ratingMap, Map<Long, List<Long>> userItemMap) {
        String path = basePath + "/u.data";
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(new File(path)));
            String line = null;
            while ((line = reader.readLine()) != null) {
                String[] lineSplits = line.split("\t");
                long userId = Long.parseLong(lineSplits[0]);
                long itemId = Long.parseLong(lineSplits[1]);
                if (maxUserId != null && userId > maxUserId) {
                    continue;
                }
                if (maxItemId != null && itemId > maxItemId) {
                    continue;
                }
                float rating = Float.parseFloat(lineSplits[2]);

                if (!ratingMap.containsKey(userId)) {
                    ratingMap.put(userId, new HashMap<>());
                    userItemMap.put(userId, new ArrayList<>());
                }
                userItemMap.get(userId).add(itemId);
                ratingMap.get(userId).put(itemId, rating);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void readUserMap(Map<Long, User> userMap) {
        String path = basePath + "/u.user";
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(new File(path)));
            String line = null;
            while ((line = reader.readLine()) != null) {
                String[] lineSplits = line.split("\\|");
                long userId = Long.parseLong(lineSplits[0]);
                if (maxUserId != null && userId > maxUserId) {
                    continue;
                }
                int age = Integer.parseInt(lineSplits[1]);
                String gender = lineSplits[2];
                String occupation = lineSplits[3];
                User user = User.create(userId, age, gender, occupation);
                userMap.put(userId, user);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void readItemMap(Map<Long, Item> itemMap, Map<Integer, String> genreMap) {
        String path = basePath + "/u.item";
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(new File(path)));
            String line = null;
            while ((line = reader.readLine()) != null) {
                String[] lineSplits = line.split("\\|");
                long itemId = Long.parseLong(lineSplits[0]);
                if (maxItemId != null && itemId > maxItemId) {
                    continue;
                }
                List<String> styles = new ArrayList<>();
                for (int i = 5; i < lineSplits.length; i++) {
                    int index = Integer.parseInt(lineSplits[i]);
                    if (index == 1) {
                        styles.add(genreMap.get(i - 5));
                    }
                }
                Item item = Item.create(itemId, styles);
                itemMap.put(itemId, item);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static INDArray readItemMlMap(Map<Long, ItemMlFeature> itemMlMap,
                                         Map<Integer, ItemMlFeature> itemSliceMlMap,
                                         int dim) throws Exception {
        String path = basePath + "/item.npz";
        Map<String, INDArray> map = Nd4j.createFromNpzFile(new File(path));
        long[] itemIds = map.get("itemId").toLongVector();
        float[][] itemFeature = map.get("itemFeature").toFloatMatrix();
        INDArray itemFeatures = Nd4j.zeros(DataType.FLOAT, itemIds.length, dim);
        if (maxItemId != null) {
            itemFeatures = Nd4j.zeros(DataType.FLOAT, maxItemId, dim);
        }
        for (int i = 0; i < itemIds.length; i++) {
            long itemId = itemIds[i];
            if (maxItemId != null && itemId > maxItemId) {
                continue;
            }
            float[] feature = itemFeature[i];
            INDArray ndFeature = Nd4j.createFromArray(feature);
            itemFeatures.putSlice(i, ndFeature);
            ItemMlFeature itemMlFeature = ItemMlFeature.create(itemId, feature.length, i);
            itemMlMap.put(itemId, itemMlFeature);
            itemSliceMlMap.put(i, itemMlFeature);
        }
        return itemFeatures;
    }

    public static INDArray readUserMlMap(Map<Long, UserMlFeature> userMlMap,
                                         Map<Integer, UserMlFeature> userSliceMlMap,
                                         int dim) throws Exception {
        String path = basePath + "/user.npz";
        Map<String, INDArray> map = Nd4j.createFromNpzFile(new File(path));
        long[] userIds = map.get("userId").toLongVector();
        float[][] userFeature = map.get("userFeature").toFloatMatrix();
        INDArray userFeatures = Nd4j.zeros(DataType.FLOAT, userIds.length, dim);
        if (maxUserId != null) {
            userFeatures = Nd4j.zeros(DataType.FLOAT, maxUserId, dim);
        }
        for (int i = 0; i < userIds.length; i++) {
            long userId = userIds[i];
            if (maxUserId != null && userId > maxUserId) {
                continue;
            }
            float[] feature = userFeature[i];
            INDArray ndFeature = Nd4j.createFromArray(feature);
            userFeatures.putSlice(i, ndFeature);
            UserMlFeature userMlFeature = UserMlFeature.create(userId, feature.length, i);
            userMlMap.put(userId, userMlFeature);
            userSliceMlMap.put(i, userMlFeature);
        }
        return userFeatures;
    }

    /**
     * 最多考虑的物品与用户数目
     */
    public static Integer maxUserId = 100;

    public static Integer maxItemId = 2000;

    public static MovieSimulation create(int userDim, int itemDim) throws Exception {
        Map<Integer, String> genreMap = new HashMap<>();
        Map<Long, Map<Long, Float>> ratingMap = new HashMap<>();
        Map<Long, List<Long>> userItemMap = new HashMap<>();
        Map<Long, User> userMap = new HashMap<>();
        Map<Long, Item> itemMap = new HashMap<>();
        Map<Long, ItemMlFeature> itemMlMap = new HashMap<>();
        Map<Long, UserMlFeature> userMlMap = new HashMap<>();
        Map<Integer, ItemMlFeature> itemSliceMlMap = new HashMap<>();
        Map<Integer, UserMlFeature> userSliceMlMap = new HashMap<>();
        readGenreMap(genreMap);
        readRatingMap(ratingMap, userItemMap);
        readUserMap(userMap);
        readItemMap(itemMap, genreMap);
        INDArray itemFeatures = readItemMlMap(itemMlMap, itemSliceMlMap, itemDim);
        INDArray userFeatures = readUserMlMap(userMlMap, userSliceMlMap, userDim);
        LinUCBLearning armLearning = LinUCBLearning.create(userFeatures, itemFeatures);
        // 每一次参考 64个 topK 结果
        int topK = 64;
        return new MovieSimulation(armLearning, itemSliceMlMap, userSliceMlMap, itemMap, userMap,
                userItemMap, ratingMap, userMlMap, itemMlMap, topK);
    }

    public static void main(String[] args) throws Exception {
        MovieSimulation movieSimulation = create(32, 32);
        movieSimulation.run(100000, 200f);
    }

}
