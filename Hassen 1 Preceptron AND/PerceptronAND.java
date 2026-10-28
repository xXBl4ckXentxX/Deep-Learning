
/**
 * HANDS-ON 1: Perceptrón de Rosenblatt (sin velocidad de aprendizaje) para la compuerta AND.
 * Archivo único: contiene la clase Perceptron y el programa principal con main().
 *
 * Basado en los artículos de referencia:
 *  - Akshay L. Chandra, "Perceptron Learning Algorithm: A Graphical Explanation Of Why It Works"
 *      * el bias se trata como un peso w0 con entrada fija x0 = 1
 *      * la decisión es el producto punto w·x (frontera de decisión: w·x = 0)
 *      * si x es positivo (P) y w·x < 0  -> se SUMA x a w
 *      * si x es negativo (N) y w·x >= 0 -> se RESTA x de w

 *
 * Reglas académicas cumplidas:
 *  - POO: clase Perceptron instanciable, con atributos y métodos.
 *  - SIN learning rate en la regla de actualización.
 *  - Solo compuerta AND.
 *  - Datos (patrones, pesos y bias) definidos en el código fuente; sin Scanner.
 *  - Cero librerías externas: todo se calcula con arreglos y variables primitivas.
 *  - Imprime pesos óptimos, bias óptimo y el cálculo paso a paso de cada patrón.
 */

// ========================================================================
// CLASE PRINCIPAL (PÚBLICA): debe llamarse igual que el archivo -> PerceptronAND.
// Va PRIMERO en el archivo para que "java PerceptronAND.java" (ejecución directa
// de un solo archivo sin compilar antes) encuentre el main() correctamente.
// ========================================================================
public class PerceptronAND {

    public static void main(String[] args) {

        // 1. Patrones de entrenamiento: tabla de verdad AND
        //    Clase positiva P: salida 1  |  Clase negativa N: salida 0
        double[][] entradas = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] salidasEsperadas = {0, 0, 0, 1};

        // 2. Parámetros iniciales (pequeños, definidos en el código)
        double[] pesosIniciales = {0.0,0.0}; // w1, w2
        double biasInicial = 0.0;            // b = w0

        // 3. Instanciar el perceptrón
        Perceptron perceptron = new Perceptron(pesosIniciales, biasInicial);

        System.out.println("=== PERCEPTRÓN DE ROSENBLATT - COMPUERTA AND ===");
        System.out.println("Parámetros iniciales: w1 = " + pesosIniciales[0]
                + ", w2 = " + pesosIniciales[1] + ", b = " + biasInicial);
        System.out.println();

        // 4. Entrenamiento (regla: w_i = w_i + (y_true - y_pred) * x_i)
        System.out.println("--- CORRECCIONES DURANTE EL ENTRENAMIENTO ---");
        int epocas = perceptron.entrenar(entradas, salidasEsperadas, 100, true);
        System.out.println();
        System.out.println("Entrenamiento finalizado en " + epocas + " época(s).");
        System.out.println();

        // 5. Valores óptimos
        double w1 = perceptron.getPeso(1);
        double w2 = perceptron.getPeso(2);
        double b = perceptron.getBias();

        System.out.println("--- VALORES ÓPTIMOS ---");
        System.out.println("w_1 = " + w1);
        System.out.println("w_2 = " + w2);
        System.out.println("b   = " + b);
        System.out.println();

        // 6. Verificación paso a paso por patrón
        System.out.println("--- VERIFICACIÓN PASO A PASO ---");
        boolean todoCorrecto = true;

        for (int p = 0; p < entradas.length; p++) {
            double x1 = entradas[p][0];
            double x2 = entradas[p][1];

            double z = perceptron.sumaPonderada(entradas[p]); // z = (w1*x1 + w2*x2) + b
            int yPred = perceptron.activacion(z);             // z >= 0 -> 1, z < 0 -> 0
            int yTrue = salidasEsperadas[p];

            System.out.println("Patrón " + (p + 1) + ": (x1 = " + (int) x1 + ", x2 = " + (int) x2 + ")");
            System.out.println("  Suma ponderada: z = (" + w1 + " * " + x1 + " + " + w2 + " * " + x2
                    + ") + (" + b + ") = " + z);
            System.out.println("  Activación: z " + (z >= 0 ? ">= 0" : "< 0") + "  ->  y_pred = " + yPred);
            System.out.println("  Esperado (y_true) = " + yTrue
                    + "  ->  " + (yPred == yTrue ? "CORRECTO" : "INCORRECTO"));
            System.out.println();

            if (yPred != yTrue) {
                todoCorrecto = false;
            }
        }

        System.out.println(todoCorrecto
                ? "El perceptrón convergió: clasifica correctamente los 4 patrones de la compuerta AND."
                : "El perceptrón NO convergió dentro del máximo de épocas.");
    }
}

// ========================================================================
// CLASE Perceptron: abstracción de la neurona artificial
// (no lleva "public" porque en un archivo solo puede haber una clase public,
//  y esa debe ser la que coincide con el nombre del archivo: PerceptronAND)
// ========================================================================
class Perceptron {

    // --------------------------------------------------------------
    // ATRIBUTOS
    // --------------------------------------------------------------

    /**
     * Vector de pesos AUMENTADO: [w0, w1, w2]
     *   pesos[0] = w0 = b (el bias, con entrada fija x0 = 1)
     *   pesos[1] = w1
     *   pesos[2] = w2
     */
    private double[] pesos;

    // --------------------------------------------------------------
    // CONSTRUCTOR
    // --------------------------------------------------------------

    /**
     * @param pesosIniciales valores iniciales de [w1, w2] (definidos en el código fuente)
     * @param biasInicial    valor inicial del bias b = w0
     */
    public Perceptron(double[] pesosIniciales, double biasInicial) {
        pesos = new double[pesosIniciales.length + 1];
        pesos[0] = biasInicial;                       // w0 = b
        for (int i = 0; i < pesosIniciales.length; i++) {
            pesos[i + 1] = pesosIniciales[i];         // w1, w2
        }
    }

    // --------------------------------------------------------------
    // VECTOR AUMENTADO: agrega x0 = 1 al inicio de la entrada
    // --------------------------------------------------------------

    /** Convierte [x1, x2] en [1, x1, x2] para que el bias entre en el producto punto. */
    private double[] aumentar(double[] x) {
        double[] xa = new double[x.length + 1];
        xa[0] = 1.0;                                  // x0 = 1
        for (int i = 0; i < x.length; i++) {
            xa[i + 1] = x[i];
        }
        return xa;
    }

    // --------------------------------------------------------------
    // SUMA PONDERADA: z = w·x = w0*1 + w1*x1 + w2*x2 = (w1*x1 + w2*x2) + b
    // --------------------------------------------------------------

    public double sumaPonderada(double[] x) {
        double[] xa = aumentar(x);
        double z = 0.0;
        for (int i = 0; i < pesos.length; i++) {
            z += pesos[i] * xa[i];                    // producto punto manual
        }
        return z;
    }

    // --------------------------------------------------------------
    // FUNCIÓN DE ACTIVACIÓN (Binary Step / Heaviside)
    //   z >= 0 -> 1 (clase positiva P)      z < 0 -> 0 (clase negativa N)
    // --------------------------------------------------------------

    public int activacion(double z) {
        return (z >=0) ? 1 : 0;
    }

    public int predecir(double[] x) {
        return activacion(sumaPonderada(x));
    }

    // --------------------------------------------------------------
    // ACTUALIZACIÓN DE PESOS (sin learning rate)
    //   error = y_true - y_pred
    //   w_i   = w_i + error * x_i      (incluye w0 = b con x0 = 1)
    //
    // Equivale a los dos casos del artículo de Chandra:
    //   Caso 1: x en P y w·x < 0   -> error = +1 -> w = w + x
    //   Caso 2: x en N y w·x >= 0  -> error = -1 -> w = w - x
    // --------------------------------------------------------------

    /** @return el error (y_true - y_pred) de este patrón; 0 si no hubo que corregir */
    public int entrenarPatron(double[] x, int yTrue) {
        int yPred = predecir(x);
        int error = yTrue - yPred;

        if (error != 0) {
            double[] xa = aumentar(x);
            for (int i = 0; i < pesos.length; i++) {
                pesos[i] = pesos[i] + error * xa[i];
            }
        }
        return error;
    }

    /**
     * Repite sobre todos los patrones hasta que una época completa no tenga errores
     * (convergencia) o se llegue a maxEpocas. Si mostrarPasos es true, imprime cada corrección.
     *
     * @return número de épocas ejecutadas
     */
    public int entrenar(double[][] entradas, int[] salidas, int maxEpocas, boolean mostrarPasos) {
        int epoca = 0;
        boolean huboErrores = true;

        while (huboErrores && epoca < maxEpocas) {
            huboErrores = false;
            epoca++;

            for (int p = 0; p < entradas.length; p++) {
                double zAntes = sumaPonderada(entradas[p]);
                int error = entrenarPatron(entradas[p], salidas[p]);

                if (error != 0) {
                    huboErrores = true;
                    if (mostrarPasos) {
                        String caso = (error > 0)
                                ? "Caso 1 (x en P y w·x < 0): w = w + x"
                                : "Caso 2 (x en N y w·x >= 0): w = w - x";
                        System.out.println("  Época " + epoca + ", patrón " + (p + 1)
                                + " (" + (int) entradas[p][0] + "," + (int) entradas[p][1] + ")"
                                + ": z = " + zAntes + " -> " + caso);
                        System.out.println("     nuevos pesos: w1 = " + getPeso(1)
                                + ", w2 = " + getPeso(2) + ", b = " + getBias());
                    }
                }
            }
        }
        return epoca;
    }

    // --------------------------------------------------------------
    // GETTERS
    // --------------------------------------------------------------

    /** @param indice 1 para w1, 2 para w2 */
    public double getPeso(int indice) {
        return pesos[indice];
    }

    /** El bias es w0 (pesos[0]). */
    public double getBias() {
        return pesos[0];
    }
}

// ========================================================================
// CLASE PRINCIPAL (PÚBLICA): debe llamarse igual que el archivo -> PerceptronAND
// ========================================================================
