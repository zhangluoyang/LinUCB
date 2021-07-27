package src.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.jackson.shaded.NDArrayDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArraySerializer;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.bytedeco.openblas.global.openblas_nolapack;

import java.io.File;
import java.util.Arrays;
import java.util.Map;

public class S {

    /**
     * 序列化一些工具
     */
    private static ObjectMapper objectMapper;

    static {
        objectMapper = new ObjectMapper();
        SimpleModule nd4j = new SimpleModule("nd4j");
        nd4j.addDeserializer(INDArray.class, new NDArrayDeSerializer());
        nd4j.addSerializer(INDArray.class, new NDArraySerializer());
        objectMapper.registerModule(nd4j);
    }

    /**
     * 矩阵序列化
     */
    public static String serialize(INDArray indArray) throws JsonProcessingException {
        return objectMapper.writeValueAsString(indArray);
    }

    /**
     * 矩阵反序列化
     */
    public static INDArray deSerialize(String string) throws JsonProcessingException {
        return objectMapper.readValue(string, INDArray.class);
    }
}
