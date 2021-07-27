package src.instance;

import java.util.List;

public class Item implements Instance {

    /**
     * 物品Id
     */
    private final long itemId;

    /**
     * 物品风格
     */
    private final List<String> styles;


    private Item(long itemId, List<String> styles) {
        this.itemId = itemId;
        this.styles = styles;
    }

    public long getItemId() {
        return itemId;
    }

    public List<String> getStyles() {
        return styles;
    }

    public static Item create(long itemId, List<String> styles) {
        return new Item(itemId, styles);
    }
}
