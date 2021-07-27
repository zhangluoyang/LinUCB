package src.instance;

public class User implements Instance {
    /**
     * 用户特征
     * */

    /**
     * 用户Id
     */
    private final long userId;

    /**
     * 用户年龄
     */
    private final int age;

    /**
     * 性别
     */
    private final String gender;

    /**
     * 职业
     */
    private final String occupation;


    private User(long userId, int age, String gender, String occupation) {
        this.userId = userId;
        this.age = age;
        this.gender = gender;
        this.occupation = occupation;
    }

    public long getUserId() {
        return userId;
    }

    public int getAge() {
        return age;
    }

    public String getGender() {
        return gender;
    }

    public String getOccupation() {
        return occupation;
    }

    @Override
    public String toString() {
        return "User{" +
                "userId=" + userId +
                ", age=" + age +
                ", gender='" + gender + '\'' +
                ", occupation='" + occupation + '\'' +
                '}';
    }

    public static User create(long userId, int age, String gender, String occupation) {
        return new User(userId, age, gender, occupation);
    }

}
