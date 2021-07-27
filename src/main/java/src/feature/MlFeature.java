package src.feature;

import java.io.Serializable;

public abstract class MlFeature implements Serializable {

    /**
     * 特征对应的下标
     */
    final int sliceIndex;
    /**
     * 特征维度
     */
    protected final int dim;

    protected MlFeature(int sliceIndex, int dim) {
        this.sliceIndex = sliceIndex;
        this.dim = dim;
    }

    public int getDim() {
        return dim;
    }

    public int getSliceIndex() {
        return sliceIndex;
    }

    @Override
    public String toString() {
        return "MlFeature{" +
                "sliceIndex=" + sliceIndex +
                ", dim=" + dim +
                '}';
    }
}
