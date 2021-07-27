package src.feature.impl;

import src.feature.MlFeature;

public class ItemMlFeature extends MlFeature {

    private final long itemId;

    private ItemMlFeature(long itemId, int dim, int sliceIndex) {
        super(sliceIndex, dim);
        this.itemId = itemId;
    }

    public long getItemId() {
        return itemId;
    }

    @Override
    public String toString() {
        return "ItemMlFeature{" +
                "dim=" + dim +
                ", itemId=" + itemId +
                '}';
    }

    public static ItemMlFeature create(long itemId, int dim, int sliceIndex) {
        return new ItemMlFeature(itemId, dim, sliceIndex);
    }

}
