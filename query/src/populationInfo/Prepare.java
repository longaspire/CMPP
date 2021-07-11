package populationInfo;

import algorithm.PrintOut;
import datagenerate.HSMDataGenRead;
import iDModel.GenTopology;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import utilities.DataGenConstant;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Prepare {
    public static String inputParIdInfo = System.getProperty("user.dir") + "/map_to_txt_data/parIDInfo.txt";
    public static String populationInfo_le = System.getProperty("user.dir") + "/population/le_query_";
    public static String populationInfo_ge = System.getProperty("user.dir") + "/population/ge_query_";
//    public static String populationInfo_true = System.getProperty("user.dir") + "/population/query_true";
    public static String outputPopulation_le = System.getProperty("user.dir") + "/population/le_population_final_";
    public static String outputPopulation_ge = System.getProperty("user.dir") + "/population/ge_population_final_";
//    public static String outputPopulation_true = System.getProperty("user.dir") + "/population/population_true_final";
    public static String outputTimestamp = System.getProperty("user.dir") + "/population/timestamp.txt";

    public static HashMap<Integer, Integer> parIds = new HashMap<>(); // key: new Id; value: original Id;
    public static HashMap<Integer, ArrayList<ArrayList<Double>>> population = new HashMap<>(); // key: original parId; value: <t1, a1, u1>, <t2, a2, u2>, ...
    public static ArrayList<Double> timestamps = new ArrayList<>();


    public static void findParId() throws IOException {
        Path path = Paths.get(inputParIdInfo);
        Scanner scanner = new Scanner(path);

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr = line.split("\t");
            int originalId = Integer.parseInt(tempArr[1]);
            int newId = Integer.parseInt((tempArr[0]));
            parIds.put(newId, originalId);
        }
    }

    public static void readOriginalPopulation(String s, int sampleInterval, String modelType) throws IOException {
        String populationInfo = "";
        if (modelType.equals("le")) {
            populationInfo = populationInfo_le;
        }
        else {
            populationInfo = populationInfo_ge;
        }
        Path path = Paths.get(populationInfo + s + "_" + sampleInterval + ".csv");
        Scanner scanner = new Scanner(path);

        Boolean recordTimestamp = false;
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr = line.split("\t");
            int originalId = Integer.parseInt(tempArr[0]);
            ArrayList<ArrayList<Double>> tempLists = new ArrayList<>();
            for (int i = 1; i < tempArr.length; i++) {
                String[] info = tempArr[i].split(",");
                double t = (Double.parseDouble(info[0]) - 10770) * sampleInterval;
                double a = Double.parseDouble(info[1]);
                double u = Double.parseDouble(info[2]);
                if (t > DataGenConstant.endTime) break;
                if (!recordTimestamp) {
                    timestamps.add(t);
                }

                tempLists.add(new ArrayList<Double>(Arrays.asList(t, a, u)));
            }
            population.put(originalId, tempLists);
            recordTimestamp = true;
        }
    }

    public static void setPopulation(String modelType) {
        for (int i = 0; i < IndoorSpace.iPartitions.size(); i++) {
            Partition par = IndoorSpace.iPartitions.get(i);
            int parOriginalId = parIds.get(i);
            if (population.get(parOriginalId) != null) {
                ArrayList<ArrayList<Double>> lists = population.get(parOriginalId);
                for (int j = 0; j < lists.size(); j++) {
                    ArrayList<Double> info = lists.get(j);
                    int t = (int)(double)info.get(0);
                    double a = info.get(1);
                    double u = info.get(2);

                    if (modelType.equals("le")) {
                        par.addPopulation_le(t, a, u);
                    }
                    else {
                        par.addPopulation_ge(t, a, u);
                    }
                }
            }
            else {
                for (int j = 0; j < timestamps.size(); j++) {
                    int t = (int)(double)timestamps.get(j);
                    if (modelType.equals("le")) {
                        par.addPopulation_le(t, 0, 0);
                    }
                    else {
                        par.addPopulation_ge(t, 0, 0);
                    }
                }
            }
        }
    }

    public static void savePopulation(String s, int sampleInterval, String modelType) throws IOException{
        String outputPopulation = "";
        if (modelType.equals("le")) {
            outputPopulation = outputPopulation_le;
        }
        else {
            outputPopulation = outputPopulation_ge;
        }
        String results = "";
        for (int i = 0; i < IndoorSpace.iPartitions.size(); i++) {
            System.out.println("parId " + i);
            Partition par = IndoorSpace.iPartitions.get(i);
            results += par.getmID();
            for (int j = 0; j < timestamps.size(); j++) {
                int t = (int)(double)timestamps.get(j);
                ArrayList<Double> info = new ArrayList<>();
                if (modelType.equals("le")) {
                    info = par.getPopulation_le(t);
                }
                else{
                    info = par.getPopulation_ge(t);
                }
                double a = info.get(0);
                double u = info.get(1);
                results += "\t" + t + "," + a + "," + u;
            }
            results += "\n";
        }
        System.out.println("writing...");
        try {
            FileWriter output = new FileWriter(outputPopulation + s + "_" + sampleInterval + ".txt");
            output.write(results);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public static void saveTimestamp() throws IOException {
        String results = "";
        for (int i = 0; i < timestamps.size(); i++) {
            double t = timestamps.get(i);
            results += t + "\n";
        }
        try {
            FileWriter output = new FileWriter(outputTimestamp);
            output.write(results);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public static void main(String arg[]) throws IOException{
        PrintOut printOut = new PrintOut();

        HSMDataGenRead dateGenReadMen = new HSMDataGenRead();
        dateGenReadMen.dataGen("TData");

        GenTopology genTopology = new GenTopology();
        genTopology.genTopology();

        Prepare.findParId();
        Prepare.readOriginalPopulation("true", 10, "ge");
//        Prepare.saveTimestamp(10);

        System.out.println("set population...");
        Prepare.setPopulation("ge");

        System.out.println("save population...");
        Prepare.savePopulation("true", 10, "ge");



        System.out.println(timestamps);
        System.out.println(timestamps.size());
        System.out.println(population.size());
        System.out.println(population.get(160));
        System.out.println(population.get(160).size());
    }

}
