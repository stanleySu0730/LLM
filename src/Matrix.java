public class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = data;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double[][] getData() {
        return data;
    }

    public static Matrix random(int rows, int cols, double mean, double std) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = mean + std * new java.util.Random().nextGaussian();
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for subtraction.");
        }
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    public Matrix subtract(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - scalar;
            }
        }
        return result;
    }

    // Add a scalar or a matrix
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition.");
        }
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    public Matrix add(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + scalar;
            }
        }
        return result;
    }

    // Element-wise square root
    public static Matrix sqrt(Matrix m) {
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[i][j] = Math.sqrt(m.data[i][j]);
            }
        }
        return result;
    }

    // Mean along a dimension (-1 for row-wise mean)
    public Matrix mean(int axis) {
        if (axis == -1) { // Row-wise mean
            Matrix result = new Matrix(this.rows, 1);
            for (int i = 0; i < rows; i++) {
                double sum = 0.0;
                for (int j = 0; j < cols; j++) {
                    sum += this.data[i][j];
                }
                result.data[i][0] = sum / cols;
            }
            return result;
        }
        throw new UnsupportedOperationException("Mean only supports axis=-1 (row-wise) for now.");
    }

    // Variance along a dimension (-1 for row-wise variance)
    public Matrix variance(int axis, boolean unbiased) {
        if (axis == -1) { // Row-wise variance
            Matrix mean = this.mean(-1);
            Matrix result = new Matrix(this.rows, 1);
            for (int i = 0; i < rows; i++) {
                double sum = 0.0;
                for (int j = 0; j < cols; j++) {
                    double diff = this.data[i][j] - mean.data[i][0];
                    sum += diff * diff;
                }
                result.data[i][0] = sum / (unbiased ? (cols - 1) : cols);
            }
            return result;
        }
        throw new UnsupportedOperationException("Variance only supports axis=-1 (row-wise) for now.");
    }

    // Divide by scalar or element-wise by a matrix
    public Matrix divide(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for division.");
        }
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] / other.data[i][j];
            }
        }
        return result;
    }

    public Matrix divide(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] / scalar;
            }
        }
        return result;
    }

    // Element-wise multiplication
    public Matrix multiply(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for element-wise multiplication.");
        }
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }
}

    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[j][i] = m.data[i][j];
            }
        }
        return result;
    }

    public static Matrix applyMask(Matrix m, Matrix mask) {
        if (m.rows != mask.rows || m.cols != mask.cols) {
            throw new IllegalArgumentException("Mask dimensions must match matrix dimensions.");
        }
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[i][j] = mask.data[i][j] == 0 ? -1e9 : m.data[i][j];
            }
        }
        return result;
    }
//softmax
    public static Matrix softmax(Matrix m) {
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            double rowMax = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < m.cols; j++) {
                rowMax = Math.max(rowMax, m.data[i][j]);
            }

            double sum = 0.0;
            for (int j = 0; j < m.cols; j++) {
                sum += Math.exp(m.data[i][j] - rowMax);
            }

            for (int j = 0; j < m.cols; j++) {
                result.data[i][j] = Math.exp(m.data[i][j] - rowMax) / sum;
            }
        }
        return result;
    }
// applies random dropout of rate rate
    public static Matrix dropout(Matrix m, double rate) {
        Matrix result = new Matrix(m.rows, m.cols);
        java.util.Random random = new java.util.Random();
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[i][j] = random.nextDouble() > rate ? m.data[i][j] : 0.0;
            }
        }
        return result;
    }
}
