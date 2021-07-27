package src.arm;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface BandArmLearning extends Serializable {

    public INDArray calUcbScore(int userIndex, float alpha);

    public void update(int userIndex, int itemIndex, float reward);

}
