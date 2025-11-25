// ============================================
// ARCHIVO 1: neuralNetwork.js (Backend/Lógica)
// ============================================

// Dataset de entrenamiento
const trainingData = [
    [7, 8, 95, 4.5, 4.7, 1],
    [4, 6, 70, 3.0, 3.2, 1],
    [9, 7, 98, 4.8, 4.9, 1],
    [3, 5, 60, 2.5, 2.7, 0],
    [8, 8, 92, 4.3, 4.5, 1],
    [5, 7, 75, 3.5, 3.6, 1],
    [10, 8, 100, 5.0, 5.0, 1],
    [2, 4, 50, 2.0, 2.2, 0],
    [6, 7, 85, 4.0, 4.1, 1],
    [8, 6, 88, 4.2, 4.3, 1],
    [4, 5, 65, 2.8, 3.0, 1],
    [9, 9, 96, 4.6, 4.8, 1],
    [7, 7, 90, 4.2, 4.4, 1],
    [3, 6, 55, 2.3, 2.5, 0],
    [10, 7, 98, 4.7, 4.9, 1],
    [5, 8, 80, 3.7, 3.9, 1],
    [6, 6, 82, 3.9, 4.0, 1],
    [8, 9, 94, 4.5, 4.6, 1],
    [4, 4, 68, 3.1, 3.3, 1],
    [7, 8, 87, 4.1, 4.2, 1],
    [2, 5, 55, 2.4, 2.6, 0],
    [3, 4, 45, 1.8, 2.0, 0],
    [5, 6, 72, 3.2, 3.4, 1],
    [6, 8, 88, 4.0, 4.2, 1],
    [4, 7, 78, 3.4, 3.6, 1]
];

/**
 * Red Neuronal Multi-Tarea
 * Arquitectura: 4 entradas → N ocultas → 2 salidas (Regresión + Clasificación)
 */
class MultiTaskNeuralNetwork {
    constructor(inputSize, hiddenSize, regressionOutput, classificationOutput) {
        // Pesos para capa oculta
        this.weightsIH = this.randomMatrix(inputSize, hiddenSize);
        this.biasH = this.randomMatrix(1, hiddenSize);
        
        // Pesos para regresión (predecir nota)
        this.weightsHR = this.randomMatrix(hiddenSize, regressionOutput);
        this.biasR = this.randomMatrix(1, regressionOutput);
        
        // Pesos para clasificación (aprobado/reprobado)
        this.weightsHC = this.randomMatrix(hiddenSize, classificationOutput);
        this.biasC = this.randomMatrix(1, classificationOutput);
    }

    /**
     * Genera matriz con valores aleatorios entre -1 y 1
     */
    randomMatrix(rows, cols) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = Math.random() * 2 - 1;
            }
        }
        return matrix;
    }

    /**
     * Función de activación Sigmoid
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivada de Sigmoid para backpropagation
     */
    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    /**
     * Multiplicación de matrices
     */
    matrixMultiply(a, b) {
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < b.length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    /**
     * Propagación hacia adelante (Forward Pass)
     */
    forward(input) {
        // Capa oculta (compartida entre regresión y clasificación)
        const hiddenInput = this.matrixMultiply([input], this.weightsIH);
        const hidden = hiddenInput[0].map((val, i) => 
            this.sigmoid(val + this.biasH[0][i])
        );

        // Salida de regresión (predice la nota)
        const regressionInput = this.matrixMultiply([hidden], this.weightsHR);
        const regressionOutput = regressionInput[0].map((val, i) => 
            this.sigmoid(val + this.biasR[0][i])
        );

        // Salida de clasificación (predice aprobado/reprobado)
        const classificationInput = this.matrixMultiply([hidden], this.weightsHC);
        const classificationOutput = classificationInput[0].map((val, i) => 
            this.sigmoid(val + this.biasC[0][i])
        );

        return { hidden, regressionOutput, classificationOutput };
    }

    /**
     * Entrena la red neuronal usando backpropagation
     * @param {Array} data - Dataset de entrenamiento
     * @param {number} epochs - Número de iteraciones
     * @param {number} learningRate - Tasa de aprendizaje
     */
    train(data, epochs = 8000, learningRate = 0.15) {
        console.log(`🔄 Entrenando red neuronal con ${epochs} épocas...`);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            // Mostrar progreso cada 1000 épocas
            if (epoch % 1000 === 0) {
                console.log(`Época ${epoch}/${epochs}`);
            }

            for (let i = 0; i < data.length; i++) {
                const input = data[i].slice(0, 4);
                const targetRegression = [(data[i][4] - 1) / 4]; // Normalizar nota (1-5 → 0-1)
                const targetClassification = [data[i][5]]; // Aprobado (1) o Reprobado (0)

                // Normalizar entradas
                const normalizedInput = [
                    input[0] / 12,          // Horas de estudio
                    input[1] / 9,           // Horas de sueño
                    input[2] / 100,         // Asistencia
                    (input[3] - 1) / 4      // Nota anterior
                ];

                // Forward pass
                const { hidden, regressionOutput, classificationOutput } = this.forward(normalizedInput);

                // ===== BACKPROPAGATION PARA REGRESIÓN =====
                const regressionError = targetRegression.map((t, idx) => t - regressionOutput[idx]);
                const regressionDelta = regressionError.map((e, idx) => 
                    e * this.sigmoidDerivative(regressionOutput[idx])
                );

                // ===== BACKPROPAGATION PARA CLASIFICACIÓN =====
                const classificationError = targetClassification.map((t, idx) => t - classificationOutput[idx]);
                const classificationDelta = classificationError.map((e, idx) => 
                    e * this.sigmoidDerivative(classificationOutput[idx])
                );

                // Error en capa oculta (combinado de ambas tareas)
                const hiddenErrorR = this.weightsHR.map(weights => 
                    weights.reduce((sum, w, idx) => sum + w * regressionDelta[idx], 0)
                );
                const hiddenErrorC = this.weightsHC.map(weights => 
                    weights.reduce((sum, w, idx) => sum + w * classificationDelta[idx], 0)
                );
                const hiddenError = hiddenErrorR.map((e, idx) => e + hiddenErrorC[idx]);
                const hiddenDelta = hiddenError.map((e, idx) => 
                    e * this.sigmoidDerivative(hidden[idx])
                );

                // Actualizar pesos de regresión
                for (let j = 0; j < this.weightsHR.length; j++) {
                    for (let k = 0; k < this.weightsHR[0].length; k++) {
                        this.weightsHR[j][k] += learningRate * regressionDelta[k] * hidden[j];
                    }
                }
                for (let j = 0; j < this.biasR[0].length; j++) {
                    this.biasR[0][j] += learningRate * regressionDelta[j];
                }

                // Actualizar pesos de clasificación
                for (let j = 0; j < this.weightsHC.length; j++) {
                    for (let k = 0; k < this.weightsHC[0].length; k++) {
                        this.weightsHC[j][k] += learningRate * classificationDelta[k] * hidden[j];
                    }
                }
                for (let j = 0; j < this.biasC[0].length; j++) {
                    this.biasC[0][j] += learningRate * classificationDelta[j];
                }

                // Actualizar pesos de capa oculta
                for (let j = 0; j < this.weightsIH.length; j++) {
                    for (let k = 0; k < this.weightsIH[0].length; k++) {
                        this.weightsIH[j][k] += learningRate * hiddenDelta[k] * normalizedInput[j];
                    }
                }
                for (let j = 0; j < this.biasH[0].length; j++) {
                    this.biasH[0][j] += learningRate * hiddenDelta[j];
                }
            }
        }
        
        console.log('✅ Entrenamiento completado!');
    }

    /**
     * Realiza una predicción con los datos de entrada
     * @param {Array} input - [horas_estudio, horas_sueño, asistencia, nota_anterior]
     * @returns {Object} - {score: nota predicha, probability: probabilidad de aprobar}
     */
    predict(input) {
        // Normalizar entradas
        const normalizedInput = [
            input[0] / 12,
            input[1] / 9,
            input[2] / 100,
            (input[3] - 1) / 4
        ];
        
        const { regressionOutput, classificationOutput } = this.forward(normalizedInput);
        
        // Desnormalizar la nota (0-1 → 1-5)
        const score = regressionOutput[0] * 4 + 1;
        
        // Probabilidad de aprobar (0-1)
        const probability = classificationOutput[0];
        
        return { score, probability };
    }
}

// Exportar para uso en el frontend
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MultiTaskNeuralNetwork, trainingData };
}