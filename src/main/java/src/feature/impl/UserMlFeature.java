package src.feature.impl;

import src.feature.MlFeature;

public class UserMlFeature extends MlFeature {

    private final long userId;

    private UserMlFeature(long userId, int dim, int sliceIndex) {
        super(sliceIndex, dim);
        this.userId = userId;
    }

    public long getUserId() {
        return userId;
    }

    @Override
    public String toString() {
        return "UserMlFeature{" +
                "dim=" + dim +
                ", userId=" + userId +
                '}';
    }

    public static UserMlFeature create(long userId, int dim, int sliceIndex) {
        return new UserMlFeature(userId, dim, sliceIndex);
    }
}
