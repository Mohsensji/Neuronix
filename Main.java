import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
      

        ArrayList<Double> x1 = new ArrayList<>(Arrays.asList(2.0, -1.0, 0.0));
        ArrayList<Double> x2 = new ArrayList<>(Arrays.asList(3.0, -1.0, 0.5));
        ArrayList<Double> x3 = new ArrayList<>(Arrays.asList(0.5, 1.0, 1.0));
        ArrayList<Double> x4 = new ArrayList<>(Arrays.asList(1.0, 1.0, -1.0));
        ArrayList<ArrayList<Double>> xs = new ArrayList<>();
        xs.add(x1);xs.add(x2);xs.add(x3);xs.add(x4);
        ArrayList<ArrayList<Value>> ypreds = new ArrayList<>();
        NN.MLP MLP = new NN.MLP(3,new ArrayList<>(Arrays.asList(4,4,1))    );
        
        for(ArrayList<Double> x: xs){
            ypreds.add(MLP.run(x));
        }
        System.out.println(ypreds);
        
    }
}



