package experiments;

import algorithm.Init;
import algorithm.PTQ_Range;
import indoor_entitity.Point;
import utilities.DataGenConstant;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

public class AccuracyTest_Range {

    static int sampleInterval = 10;
    static int populationThreshold = 2;
    static double probabilityThreshold = 0.5;
    public static String result_true_range = System.getProperty("user.dir") + "/result/result_true_range.txt";
    public static String result_pre_range_ge = System.getProperty("user.dir") + "/result/result_pre_range_ge.txt";
    public static String result_pre_range_le = System.getProperty("user.dir") + "/result/result_pre_range_le.txt";

    public static String outFileAcc = System.getProperty("user.dir") + "/result/" + "TData_range_precision.csv";
    public static String outFileRecall = System.getProperty("user.dir") + "/result/" + "TData_range_recall.csv";

    public static void getGroundTruth() throws IOException {
        Init.init("true", 10);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "TData", sampleInterval);
        System.out.println("start querying...");
        ArrayList<Integer> ranges = new ArrayList<>(Arrays.asList(1000, 1500, 2000, 2500, 3000));
        String s = "";
        for (int i = 0; i < ranges.size(); i++) {
            int range = ranges.get(i);
            s += range;
            for (int j = 0; j < 10; j++) {
                s += "\t" + j;
                System.out.println("traId: " + j);
                HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s += ";" + t;
                    ArrayList<Integer> result = PTQ_Range.range(p, range, t, populationThreshold, probabilityThreshold, false);
                    for (int parId : result) {
                        s += "," + parId;
                    }
                }

            }
            s += "\n";
        }

        try {
            FileWriter output = new FileWriter(result_true_range);
            output.write(s);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public static void calAccuracy() throws IOException {
        String resultAcc = "range" + "\t" + "ge" + "\t" + "le" + "\n";
        String resultRecall = "range" + "\t" + "ge" + "\t" + "le" + "\n";

        Path path = Paths.get(result_true_range);
        Scanner scanner = new Scanner(path);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> range_true_map = new HashMap<>();
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr1 = line.split("\t");
            int range = Integer.parseInt(tempArr1[0]);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_map = new HashMap<>();
            for (int m = 1; m < tempArr1.length; m++) {
                String [] tempArr2 = tempArr1[m].split(";");
                int traId = Integer.parseInt(tempArr2[0]);
                HashMap<Integer, ArrayList<Integer>> time_map = new HashMap<>();
                for (int n = 1; n < tempArr2.length; n++) {
                    String [] tempArr3 = tempArr2[n].split(",");
                    int t = Integer.parseInt(tempArr3[0]);
                    ArrayList<Integer> true_list = new ArrayList<>();
                    for (int i = 1; i < tempArr3.length - 1; i++) {
                        int parId = Integer.parseInt(tempArr3[i]);
                        true_list.add(parId);
                    }
                    time_map.put(t, true_list);
                }
                tra_map.put(traId, time_map);
            }

            range_true_map.put(range, tra_map);
        }

        Path path1 = Paths.get(result_pre_range_ge);
        Scanner scanner1 = new Scanner(path1);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> range_ge_map = new HashMap<>();
        while (scanner1.hasNextLine()) {
            String line = scanner1.nextLine();
            String [] tempArr1 = line.split("\t");
            int range = Integer.parseInt(tempArr1[0]);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_map = new HashMap<>();
            for (int m = 1; m < tempArr1.length; m++) {
                String [] tempArr2 = tempArr1[m].split(";");
                int traId = Integer.parseInt(tempArr2[0]);
                HashMap<Integer, ArrayList<Integer>> time_map = new HashMap<>();
                for (int n = 1; n < tempArr2.length; n++) {
                    String [] tempArr3 = tempArr2[n].split(",");
                    int t = Integer.parseInt(tempArr3[0]);
                    ArrayList<Integer> pre_list = new ArrayList<>();
                    for (int i = 1; i < tempArr3.length - 1; i++) {
                        int parId = Integer.parseInt(tempArr3[i]);
                        pre_list.add(parId);
                    }
                    time_map.put(t, pre_list);
                }
                tra_map.put(traId, time_map);
            }

            range_ge_map.put(range, tra_map);
        }

        Path path2 = Paths.get(result_pre_range_le);
        Scanner scanner2 = new Scanner(path2);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> range_le_map = new HashMap<>();
        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String [] tempArr1 = line.split("\t");
            int range = Integer.parseInt(tempArr1[0]);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_map = new HashMap<>();
            for (int m = 1; m < tempArr1.length; m++) {
                String [] tempArr2 = tempArr1[m].split(";");
                int traId = Integer.parseInt(tempArr2[0]);
                HashMap<Integer, ArrayList<Integer>> time_map = new HashMap<>();
                for (int n = 1; n < tempArr2.length; n++) {
                    String [] tempArr3 = tempArr2[n].split(",");
                    int t = Integer.parseInt(tempArr3[0]);
                    ArrayList<Integer> pre_list = new ArrayList<>();
                    for (int i = 1; i < tempArr3.length - 1; i++) {
                        int parId = Integer.parseInt(tempArr3[i]);
                        pre_list.add(parId);
                    }
                    time_map.put(t, pre_list);
                }
                tra_map.put(traId, time_map);
            }

            range_le_map.put(range, tra_map);
        }

        ArrayList<Integer> ranges = new ArrayList<>(Arrays.asList(1000, 1500, 2000, 2500, 3000));

        for (int range: ranges) {
            resultAcc += range;
            resultRecall += range;

            System.out.println("range: " + range);
            ArrayList<Double> accList1 = new ArrayList<>();
            ArrayList<Double> recallList1 = new ArrayList<>();

            ArrayList<Double> accList2 = new ArrayList<>();
            ArrayList<Double> recallList2 = new ArrayList<>();

            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_ge_map = range_ge_map.get(range);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_true_map = range_true_map.get(range);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_le_map = range_le_map.get(range);
            System.out.println("tra_true_map size: " + tra_true_map.size());
            System.out.println("tra_ge_map size: " + tra_ge_map.size());
            System.out.println("tra_le_map size: " + tra_le_map.size());

            for (int i = 0; i < 10; i++) {
                System.out.println("traId: " + i);
                HashMap<Integer, ArrayList<Integer>> time_ge_map = tra_ge_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_true_map = tra_true_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_le_map = tra_le_map.get(i);
                System.out.println("time_true_map size: " + time_true_map.size());
                System.out.println("time_ge_map size: " + time_ge_map.size());
                System.out.println("time_le_map size: " + time_le_map.size());

                for (int t = 0; t < DataGenConstant.endTime; t += sampleInterval) {
                    System.out.println("t: " + t);
                    ArrayList<Integer> ge_list = time_ge_map.get(t);
                    ArrayList<Integer> true_list = time_true_map.get(t);
                    ArrayList<Integer> le_list = time_le_map.get(t);
                    if (ge_list == null || true_list == null || le_list == null) continue;
                    System.out.println("true_list size: " + true_list.size());
                    System.out.println("ge_list size: " + ge_list.size());
                    System.out.println("le_list size: " + le_list.size());
                    int num1 = 0;
                    int num2 = 0;
                    for (int true_parId: true_list) {
                        for (int ge_parId: ge_list) {
                            if (ge_parId == true_parId) {
                                num1++;
                            }
                        }
                        for (int le_parId: le_list) {
                            if (le_parId == true_parId) {
                                num2++;
                            }
                        }
                    }

                    double acc1;
                    double recall1;

                    double acc2;
                    double recall2;

                    if (ge_list.size() == 0) {
                        acc1 = 1;
                    }
                    else {
                        acc1 = (double)num1 / (double)ge_list.size();
                    }

                    if (true_list.size() == 0) {
                        recall1 = 1;
                        recall2 = 1;
                    }
                    else {
                        recall1 = (double)num1 / (double)true_list.size();
                        recall2 = (double)num2 / (double)true_list.size();
                    }

                    if (le_list.size() == 0) {
                        acc2 = 1;
                    }
                    else {
                        acc2 = (double)num2 / (double)le_list.size();
                    }

                    System.out.println("acc1: " + acc1 + " recall1: " + recall1);
                    System.out.println("acc2: " + acc2 + " recall2: " + recall2);
                    accList1.add(acc1);
                    recallList1.add(recall1);
                    accList2.add(acc2);
                    recallList2.add(recall2);
                }

            }

            System.out.println("accList1: " + accList1);
            System.out.println("accList2: " + accList2);
            System.out.println("recallList1: " + recallList1);
            System.out.println("recallList2: " + recallList2);

            ArrayList<ArrayList<Double>> accLists = new ArrayList<>();
            accLists.add(accList1);
            accLists.add(accList2);

            for (int i = 0; i < accLists.size(); i++) {
                double sum = 0;
                double ave = 0;
                ArrayList<Double> accList = accLists.get(i);
                for (double acc: accList) {
                    sum += acc;
                    System.out.println("sum + acc: " + sum);
                }
                ave = sum / accList.size();
                resultAcc += "\t" + ave;
            }
            resultAcc += "\n";

            ArrayList<ArrayList<Double>> recallLists = new ArrayList<>();
            recallLists.add(recallList1);
            recallLists.add(recallList2);
            for (int i = 0; i < recallLists.size(); i++) {
                double sum = 0;
                double ave = 0;
                ArrayList<Double> recallList = recallLists.get(i);
                for (double recall: recallList) {
                    sum += recall;
                }
                ave = sum / recallList.size();
                resultRecall += "\t" + ave;
            }
            resultRecall += "\n";

        }

        FileOutputStream outputTime = new FileOutputStream(outFileAcc);
        outputTime.write(resultAcc.getBytes());
        outputTime.flush();
        outputTime.close();

        FileOutputStream outputMem = new FileOutputStream(outFileRecall);
        outputMem.write(resultRecall.getBytes());
        outputMem.flush();
        outputMem.close();


    }

    public static void main(String[] arg) throws IOException{
        getGroundTruth();
        calAccuracy();
    }
}
