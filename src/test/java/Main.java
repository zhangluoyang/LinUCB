import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

import java.util.concurrent.TimeUnit;

/**
 * Created by Administrator on 2017/9/26.
 */
class FactorialCalculator implements Callable<Integer> {

    private Integer number;

    public FactorialCalculator(Integer number) {
        this.number = number;
    }

    public Integer call() throws Exception {
        int result = 1;

        if (number == 0 || number == 1) {
            result = 1;
        } else {
            for (int i = 2; i < number; i++) {
                result *= i;
                TimeUnit.MICROSECONDS.sleep(200);
                if (i == 5) {
                    throw new IllegalArgumentException("excepion happend");//计算5以上的阶乘都会抛出异常. 根据需要注释该if语句
                }
            }
        }
        System.out.printf("%s: %d\n", Thread.currentThread().getName(), result);
        return result;
    }
}

public class Main {

    public static void main(String[] args) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(0, Integer.MAX_VALUE,
                                                            60L, TimeUnit.SECONDS,
                                                            new SynchronousQueue<Runnable>());
        List<Future<Integer>> resultList = new ArrayList<Future<Integer>>();
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
            int rand = random.nextInt(10);

            FactorialCalculator factorialCalculator = new FactorialCalculator(rand);
            Future<Integer> res = executor.submit(factorialCalculator);//异步提交, non blocking.
            resultList.add(res);
        }

        // in loop check out the result is finished
        do {
//            System.out.printf("number of completed tasks: %d\n", executor.getCompletedTaskCount());
            for (int i = 0; i < resultList.size(); i++) {
                Future<Integer> result = resultList.get(i);
                System.out.printf("Task %d : %s \n", i, result.isDone());
            }
            try {
                TimeUnit.MILLISECONDS.sleep(50);

            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } while (executor.getCompletedTaskCount() < resultList.size());


        System.out.println("Results as folloers:");
        for (int i = 0; i < resultList.size(); i++) {
            Future<Integer> result = resultList.get(i);
            Integer number = null;

            try {
                number = result.get();// blocking method
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
            System.out.printf("task: %d, result %d:\n", i, number);
        }
        executor.shutdown();
    }
}
