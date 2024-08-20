
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

public class Value {

    // Initialization
    public double data = 0.0;
    public double grad = 0.0;
    public String op = "";
    public String label = "";
    public List<Value> prev = new ArrayList<>();
    public Runnable _backward = () -> {
    };

    // Constructor Method
    public Value(String label, double data, double grad, String operation) {
        this.data = data;
        this.grad = grad;
        this.op = operation;

        this.label = label;
    }

    // Setter Methods
    void set_prev(Value prev) {
        this.prev.add(prev);
    }

    void set_backward(Runnable backward_function) {
        this._backward = backward_function;
    }

    // Non Linearity Functions (Activation Functions)
    public Value tanh() {
        double tanh_out = (Math.exp(this.data * 2.0) - 1.0) / (Math.exp(this.data * 2.0) + 1.0);
        Value result = new Value("", tanh_out, 0.0, "tanh");
        result.set_prev(this);
        result.set_backward(() -> {
            this.grad += (1 - Math.pow(tanh_out, 2)) * result.grad;
        });
        return result;
    }

    public Value relu() {
        double relu_out = Math.max(0.0, this.data);
        Value result = new Value("", relu_out, 0.0, "relu");
        result.set_prev(this);
        result.set_backward(() -> {
            this.grad += (relu_out == 0.0) ? 0.0 : 1.0;
        });
        return result;
    }

    // Arithematic Operations
    public Value add(Value other) {
        Value result = new Value("", (this.data + other.data), 0.0, "+");

        result.set_backward(() -> {
            this.grad += 1 * result.grad;
            other.grad += 1 * result.grad;
        });
        result.set_prev(this);
        result.set_prev(other);
        return result;
    }

    public Value double_add(double other) {
        Value res = new Value("", other, 0.0, "");
        return this.add(res);
    }

    public Value sub(Value other) {
        Value result = new Value("", (this.data - other.data), 0.0, "-");
        result.set_prev(this);
        result.set_prev(other);
        return result;
    }

    public Value double_sub(double other) {
        Value res = new Value("", other, 0.0, "");
        return this.sub(res);
    }

    public Value mul(Value other) {
        Value result = new Value("", (this.data * other.data), 0.0, "*");
        result.set_backward(() -> {
            this.grad += other.data * result.grad;
            other.grad = this.data * result.grad;
        });
        result.set_prev(this);
        result.set_prev(other);
        return result;
    }

    public Value double_mul(double other) {
        Value res = new Value("", other, 0.0, "");
        return this.mul(res);
    }

    public Value div(Value other) {
        Value result = new Value("", (this.data / other.data), 0.0, "/");
        result.set_prev(this);
        result.set_prev(other);
        return result;
    }

    public Value double_div(double other) {
        Value res = new Value("", other, 0.0, "");
        return this.div(res);
    }

    // Topological Order for Computational Graph Parameters
    private void build_topo(Value G, HashSet<Value> visited, List<Value> topo) {

        if (!visited.contains(G)) {
            visited.add(G);
            for (Value elem : G.prev) {
                build_topo(elem, visited, topo);
            }
            topo.add(G);
        }
    }

    // Backward Pass function
    public void backward() {
        List<Value> topo = new ArrayList<>();
        HashSet<Value> visited = new HashSet<>();
        this.grad = 1;
        Value G = this;
        build_topo(G, visited, topo);
        Collections.reverse(topo);
        for (Value param : topo) {
            param._backward.run();
        }
        // System.err.println(topo);
    }

    @Override
    public String toString() {
        return "Data: " + data + ", Grad: " + grad + ", op: " + op + ", prevs: " + prev;
    }
}
