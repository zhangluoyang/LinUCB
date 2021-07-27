package src.arm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import src.utils.U;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;

public class LinUCBLearning implements BandArmLearning {

    /**
     * 用户特征矩阵
     */
    private final INDArray userFeatures;

    /**
     * 物品特征矩阵
     */
    private final INDArray itemFeatures;

    /**
     * A 矩阵 (item_num, d, d)
     */
    private final INDArray As;

    /**
     * InvA 矩阵 (item_num, d, d)
     */
    private final INDArray InvAs;

    /**
     * b 矩阵 (item, d, 1)
     */
    private final INDArray bs;

    /**
     * theta = Inv(A) * b
     * p = theta_T * x + alpha * sqrt(x_t * Inv(A) * x)
     * 批量计算所有Ucb分值
     *
     * @param userIndex 用户特征下标
     * @param alpha     控制探索的参数
     */
    public INDArray calUcbScore(int userIndex, float alpha) {
        // shape = (1, dim)
        INDArray userFeature = this.userFeatures.getRows(userIndex);
        int itemNum = (int) this.itemFeatures.shape()[0];
        // shape = (itemNum, dim)
        userFeature = Nd4j.tile(userFeature, itemNum, 1);
        // context 上下文特征 shape = (itemNum, d)
        INDArray xs = userFeature.mul(itemFeatures);
        // (itemNum, d, 1)
        xs = Nd4j.expandDims(xs, 2);
        // InvA * b (item_num, d, d) * (item_num, d, 1) = (item_num, d, 1)
        INDArray theta = U.dot(this.InvAs, this.bs);
        // theta_T * xs  (item_num, 1, d) * (item_num, d, 1) = (item_num, 1, 1)
        INDArray exploitation = U.dot(theta.permute(0, 2, 1), xs);
        // xs_t * InvA_t * xs (itemNum, 1, d) * (item_num, d, d) * (itemNum, d, 1) = (item_num, 1, 1)
        INDArray exploration = Nd4j.math.sqrt(U.dot(U.dot(xs.permute(0, 2, 1), this.InvAs), xs)).mul(alpha);
        return Nd4j.toFlattened(exploitation.add(exploration));
    }

    /**
     * A = A + x_t * x
     * b = b + r * x
     *
     * @param userIndex 用户下标
     * @param itemIndex 物品下标
     * @param reward    具体的奖赏值
     */
    public void update(int userIndex, int itemIndex, float reward) {
        INDArray userFeature = this.userFeatures.getRows(userIndex);
        INDArray itemFeature = this.itemFeatures.getRows(itemIndex);
        // (1, d)
        INDArray x = userFeature.mul(itemFeature);
        // 更新A矩阵 shape = (d, d)
        INDArray A = this.As.slice(itemIndex, 0);
        A = A.add(x.transpose().mmul(x));
        this.As.putSlice(itemIndex, A);
        INDArray InvA = InvertMatrix.invert(A, false);
        this.InvAs.putSlice(itemIndex, InvA);
        // 更新b向量
        INDArray b = this.bs.slice(itemIndex, 0);
        b = b.add(x.transpose().mul(reward));
        this.bs.putSlice(itemIndex, b);
    }

    private LinUCBLearning(INDArray userFeatures,
                           INDArray itemFeatures,
                           INDArray As,
                           INDArray invAs,
                           INDArray bs) {
        this.userFeatures = userFeatures;
        this.itemFeatures = itemFeatures;
        this.As = As;
        InvAs = invAs;
        this.bs = bs;
    }

    /**
     * @param userFeatures 用户特征
     * @param itemFeatures 物品特征
     */
    public static LinUCBLearning create(INDArray userFeatures,
                                        INDArray itemFeatures) {
        long[] userShape = userFeatures.shape();
        long[] itemShape = itemFeatures.shape();
        INDArray eye = Nd4j.eye(userShape[1]);
        eye = Nd4j.expandDims(eye, 0);

        INDArray As = Nd4j.tile(eye, (int) itemShape[0], 1, 1);
        INDArray invAs = Nd4j.tile(eye, (int) itemShape[0], 1, 1);
        INDArray bs = Nd4j.zeros(itemShape[0], userShape[1], 1);

        return new LinUCBLearning(userFeatures, itemFeatures, As, invAs, bs);
    }

}
