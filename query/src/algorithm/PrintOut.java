package algorithm;

import java.io.FileNotFoundException;
import java.io.PrintStream;

public class PrintOut {
    public static void main(String arg[]) {
        PrintOut p = new PrintOut();
        System.out.print("Reallly?");
        System.out.println("Yes");
        System.out.println("So easy");



    }
    public PrintOut(){
        try {

            PrintStream print=new PrintStream(System.getProperty("user.dir") + "/printInfo/printInfo.txt");
            System.setOut(print);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
