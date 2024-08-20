import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        // Value a = new Value("a",1.0,0.0,"" );
        // Value b  = new Value("b",2.0,0.0,"");
        // Value c = a.add(b);
        // Value d  = new Value("b",4.0,0.0,"");
        // Value e = c.mul(d);
        // Value f = new Value("b",7.0,0.0,"");
        // Value g = e.add(f);
        // Value o =  g.relu();
        // o.backward();


        // NN.Neuron alpha = new NN.Neuron(3);
        // alpha.run(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0)));

        ArrayList<Double> x1 = new ArrayList<>(Arrays.asList(2.0, 3.0, -1.0, 0.0));
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
        // System.out.println("a_grad: " + a.grad +" b_grad" + b.grad + " c_grad" + c.grad+" d_grad" + d.grad+" e_grad" + e.grad+" f_grad" + f.grad+" g_grad" + g.grad+" o_grad" + o.grad);
        // System.out.println("a_grad: " + a.grad +" b_grad" + b.grad + " c_grad" + c.grad);

        
    }
}



