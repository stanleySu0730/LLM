public class GELU {
    public static Matrix forward(Matrix x) {
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        return x.multiply(0.5).multiply(
                Matrix.ones(x.getRows(), x.getCols())
                        .add(x.multiply(sqrt2OverPi)
                                .multiply(x.pow(3).multiply(0.044715))
                                .tanh())
        );
    }
}
