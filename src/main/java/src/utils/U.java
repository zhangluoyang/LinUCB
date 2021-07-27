package src.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.net.Inet4Address;
import java.util.*;
import java.util.stream.Collectors;

public class U {

    /**
     * 矩阵常用的操作
     */
    public static int[] argSort(INDArray indArray) {
        INDArray array = Nd4j.zerosLike(indArray);
        Nd4j.copy(indArray, array);
        INDArray[] indArrays = Nd4j.sortWithIndices(array, 0, true);
        return indArrays[0].toIntVector();
    }

    /**
     * 返回 排序后的index
     *
     * @param indArray 矩阵
     * @param topK
     */
    public static int[] argTopK(INDArray indArray, int topK) {
        // sortWithIndices 这个函数会对数据 进行重新排序
        INDArray array = Nd4j.zerosLike(indArray);
        Nd4j.copy(indArray, array);
        INDArray[] indArrays = Nd4j.sortWithIndices(array, 0, false);
        INDArray index = indArrays[0];
        int num = (int) index.shape()[0];
        int rNum = Math.min(num, topK);
        int[] Index = new int[rNum];
        for (int i = 0; i < rNum; i++) {
            Index[i] = index.getInt(i, 0);
        }
        return Index;
    }

    /**
     * 三维矩阵的 dot 操作 用空间换取时间的操作 暂时还没有找到更加合适的方法
     *
     * @param indArrayA
     * @param indArrayB
     */
    public static INDArray dot(INDArray indArrayA, INDArray indArrayB) {
        assert indArrayA.shape().length == indArrayA.shape().length;
        assert indArrayA.shape().length == 3;
        long[] shapeA = indArrayA.shape();
        long[] shapeB = indArrayB.shape();
        assert shapeA[2] == shapeB[1];
        assert shapeA[1] != shapeB[2];
        INDArray B = null;
        if (shapeA[1] > shapeB[2]) {
            B = Nd4j.tile(indArrayB.permute(0, 2, 1), 1, (int) shapeA[1], 1);
            return indArrayA.mul(B).sum(true, -1);
        } else {
            B = Nd4j.tile(indArrayA.permute(0, 2, 1), 1, 1, (int) shapeB[2]);
            return B.mul(indArrayB).sum(true, 1);
        }
    }


    public static boolean isEmpty(INDArray array) {
        if (array == null || array.isEmpty()) {
            return true;
        }
        return false;
    }

//    public static void main(String[] args) {
//        INDArray a = Nd4j.createFromArray(new float[][][]{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
//        INDArray b = Nd4j.createFromArray(new float[][][]{{{1, 2}}, {{0.5f, 1}}});
//        INDArray b = Nd4j.createFromArray(new float[][][]{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
//        INDArray a = Nd4j.createFromArray(new float[][][]{{{1}, {2}}, {{0.5f}, {1}}});
//        System.out.println(Arrays.toString(a.shape()));
//        System.out.println(Arrays.toString(b.shape()));
//        System.out.println(dot(b, a));
//    }
}
