package experiments;

import algorithm.*;
import indoor_entitity.Point;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Experiment_Pop {
    static int safe_duration = 90;
    static int range = 1500;
    static int sampleInterval = 10;
//    static int populationThreshold = 2;
    static double probabilityThreshold = 0.5;
    public static String result_pre_population_ge = System.getProperty("user.dir") + "/result/result_pre_population_ge.txt";
    public static String result_pre_population_le = System.getProperty("user.dir") + "/result/result_pre_population_le.txt";

    public static void run(String outFileTime, String outFileMemory, String outFileNum) throws IOException {
        String resultTime = "population" + "\t" + "ge" + "\t" + "le" + "\n";
        String resultMemo = "population" + "\t" + "ge" + "\t" + "le" + "\n";
        String resultNum = "population" + "\t" + "le" + "\n";

        ArrayList<Integer> populations = new ArrayList<>(Arrays.asList(2, 4, 6, 8, 10));

        String s1 = "";
        String s2 = "";
        for (int i = 0; i < populations.size(); i++) {
            int populationThreshold = populations.get(i);
            System.out.println("population: " + populationThreshold + "-------------------------");

            resultTime += populationThreshold + "\t";
            resultMemo += populationThreshold + "\t";


            ArrayList<Long> arrTime1 = new ArrayList<>();
            ArrayList<Long> arrTime2 = new ArrayList<>();


            ArrayList<Long> arrMem1 = new ArrayList<>();
            ArrayList<Long> arrMem2 = new ArrayList<>();

            ArrayList<Integer> arrNum2 = new ArrayList<>();

            s1 += populationThreshold;
            s2 += populationThreshold;





            for (int j = 0; j < 10; j++) {
                s1 += "\t" + j;
                s2 += "\t" + j;
                System.out.println("traId: " + j + "---------------------------------------------------");
                HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);
                PTQ_Range_le.pros.clear();
                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s1 += ";" + t;
                    System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                    // ge
                    Runtime runtime1 = Runtime.getRuntime();
                    runtime1.gc();
                    ArrayList<Integer> result1 = new ArrayList<>();
                    long startMem1 = runtime1.totalMemory() - runtime1.freeMemory();
                    long startTime1 = System.currentTimeMillis();
                    result1 = PTQ_Range_ge.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration);
                    System.out.println("ge result: " + result1);
                    long endTime1 = System.currentTimeMillis();
                    long endMem1 = runtime1.totalMemory() - runtime1.freeMemory();
                    long mem1 = (endMem1 - startMem1) / 1024; // kb
                    long time1 = endTime1 - startTime1;
                    arrTime1.add(time1);
                    arrMem1.add(mem1);


                    for (int parId: result1) {
                        s1 += "," + parId;
                    }

//                    System.out.println("exact memory " + mem1);
                }

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s2 += ";" + t;
                    // le
                    Runtime runtime2 = Runtime.getRuntime();
                    runtime2.gc();
                    ArrayList<Integer> result2 = new ArrayList<>();
                    long startMem2 = runtime2.totalMemory() - runtime2.freeMemory();
                    long startTime2 = System.currentTimeMillis();
                    result2 = PTQ_Range_le.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration);
                    System.out.println("le result: " + result2);
                    long endTime2 = System.currentTimeMillis();
                    long endMem2 = runtime2.totalMemory() - runtime2.freeMemory();
                    long mem2 = (endMem2 - startMem2) / 1024; // kb
                    long time2 = endTime2 - startTime2;
                    arrTime2.add(time2);
                    arrMem2.add(mem2);
                    arrNum2.add(PTQ_Range_le.number);

                    for (int parId: result2) {
                        s2 += "," + parId;
                    }

//                    System.out.println("inexactS memory " + mem2);
                }
            }

            s1 += "\n";
            s2 += "\n";

            ArrayList<ArrayList<Long>> arrTimeAll = new ArrayList<>();
            arrTimeAll.add(arrTime1);
            arrTimeAll.add(arrTime2);

            for (int j = 0; j < arrTimeAll.size(); j++) {
                long sum = 0;
                long ave = 0;
                for (int h = 0; h < arrTimeAll.get(j).size(); h++) {
                    sum += arrTimeAll.get(j).get(h);
                }
                ave = sum / arrTimeAll.get(j).size();
                resultTime += ave + "\t";
            }
            resultTime += "\n";

            ArrayList<ArrayList<Long>> arrMemAll = new ArrayList<>();
            arrMemAll.add(arrMem1);
            arrMemAll.add(arrMem2);

            for (int j = 0; j < arrMemAll.size(); j++) {
                long sum = 0;
                long ave = 0;
                for (int h = 0; h < arrMemAll.get(j).size(); h++) {
                    sum += arrMemAll.get(j).get(h);
                }
                ave = sum / arrMemAll.get(j).size();
                resultMemo += ave + "\t";
            }
            resultMemo += "\n";

            double sum = 0;
            double ave = 0;
            for (int h = 0; h < arrNum2.size(); h++) {
                sum += arrNum2.get(h);
            }
            ave = sum / arrNum2.size();
            resultNum += ave + "\n";


        }

        FileOutputStream outputTime = new FileOutputStream(outFileTime);
        outputTime.write(resultTime.getBytes());
        outputTime.flush();
        outputTime.close();

        FileOutputStream outputMem = new FileOutputStream(outFileMemory);
        outputMem.write(resultMemo.getBytes());
        outputMem.flush();
        outputMem.close();

        FileOutputStream outputNum = new FileOutputStream(outFileNum);
        outputNum.write(resultNum.getBytes());
        outputNum.flush();
        outputNum.close();

        try {
            FileWriter output = new FileWriter(result_pre_population_ge);
            output.write(s1);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_population_le);
            output.write(s2);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }


    public static void main(String arg[]) throws IOException {
        PrintOut printOut = new PrintOut();
        Init.init("pre", 10);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "newHSM", sampleInterval);



        // population
        String outFileTime = System.getProperty("user.dir") + "/result/" + "newHSM_population_time.csv";
        String outFileMemory = System.getProperty("user.dir") + "/result/" + "newHSM_population_memory.csv";
        String outFileNum = System.getProperty("user.dir") + "/result/" + "newHSM_population_num.csv";

        Experiment_Pop.run(outFileTime, outFileMemory, outFileNum);

    }

}
