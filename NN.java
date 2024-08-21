
import java.util.ArrayList;

public class NN {

    static public class Neuron {

        ArrayList<Value> w = new ArrayList<>();
        Value b = new Value("bias", Neuronix.randn(-1, 1), 0, "");
        Value out;

        public Neuron(int nin) {
            for (int i = 0; i < nin; i++) {
                this.w.add(new Value("", Neuronix.randn(-1, 1), 0, ""));
            }
        }

        public Value run(ArrayList<Double> x) {
            Value activation = new Value("", 0, 0, "");
            for (int i = 0; i < w.size(); i++) {
                if (i == 0 ){
                    System.out.println("wsize" + w.size());
                    System.out.println("xsize" + x.size());
                    activation= this.w.get(i).double_mul(x.get(i));
                }
                else{
                    activation = activation.add(this.w.get(i).double_mul(x.get(i)));
                }
            }
            activation.add(this.b);
            this.out = activation.relu();
            return out;
        }

        public ArrayList<Value> parameters() {
            ArrayList<Value> parameters = w;
            parameters.add(this.b);
            return parameters;
        }

        @Override
        public String toString() {
            return this.out + "";

        }
    }

    static public class Layer {

        ArrayList<NN.Neuron> neurons = new ArrayList<>();

        public Layer(int nin, int nout) {
            for (int i = 0; i < nout; i++) {
                this.neurons.add(new NN.Neuron(nin));
            }
        }

        public ArrayList<Value> run(ArrayList<Double> x) {
            ArrayList<Value> out = new ArrayList<>();
            for (NN.Neuron n : this.neurons) {
                out.add(n.run(x));
            }
            return out;
        }

        public ArrayList<Value> parameters() {
            ArrayList<Value> parameters = new ArrayList<>();
            for (NN.Neuron n : this.neurons) {
                for (Value param : n.parameters()) {
                    parameters.add(param);
                }
            }
            return parameters;
        }

        @Override
        public String toString() {
            return this.neurons + "";

        }
    }

    static public class MLP {

        ArrayList<NN.Layer> layers = new ArrayList<>();

        public MLP(Integer nin, ArrayList<Integer> nouts) {
            ArrayList<Integer> sz = new ArrayList<>(nouts);
            sz.add(0, nin);
            for (int i = 0; i < nouts.size(); i++) {
                this.layers.add(new Layer(sz.get(i), sz.get(i + 1)));
            }
        }

        public ArrayList<Value> run(ArrayList<Double> x) {
            ArrayList<Value> out = new ArrayList<>();
            for (NN.Layer l : this.layers) {
                out = l.run(x);
            }
            return out;
        }

        public ArrayList<Value> parameters() {
            ArrayList<Value> parameters = new ArrayList<>();
            for (NN.Layer l : this.layers) {
                for (Value param : l.parameters()) {
                    parameters.add(param);
                }
            }
            return parameters;
        }

        @Override
        public String toString() {
            return this.layers + "";

        }

    }
}
