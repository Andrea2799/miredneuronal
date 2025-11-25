// ============================================
// ARCHIVO 3: app.js (Controlador del Frontend)
// ============================================

// Variable global para el modelo
let model = null;

/**
 * Inicializa y entrena el modelo cuando la página carga
 */
window.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Inicializando Red Neuronal Multi-Tarea...');
    console.log('📊 Arquitectura: 4 entradas → 10 ocultas → 2 salidas');
    console.log('   - Salida 1: Regresión (Nota 1.0-5.0)');
    console.log('   - Salida 2: Clasificación (Aprobado/Reprobado)');
    
    // Crear instancia de la red neuronal
    model = new MultiTaskNeuralNetwork(4, 10, 1, 1);
    
    // Entrenar el modelo con el dataset
    model.train(trainingData);
    
    console.log('✅ Modelo listo para hacer predicciones!');
    
    // Configurar validaciones en los inputs
    setupValidations();
});

/**
 * Configura las validaciones de los campos de entrada
 */
function setupValidations() {
    // Validación para horas de estudio
    document.getElementById('studyHours').addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        if (e.target.value !== '' && (val < 1 || val > 12)) {
            e.target.value = Math.max(1, Math.min(12, val));
        }
    });

    // Validación para horas de sueño
    document.getElementById('sleepHours').addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        if (e.target.value !== '' && (val < 4 || val > 9)) {
            e.target.value = Math.max(4, Math.min(9, val));
        }
    });

    // Validación para asistencia
    document.getElementById('attendance').addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        if (e.target.value !== '' && (val < 0 || val > 100)) {
            e.target.value = Math.max(0, Math.min(100, val));
        }
    });

    // Validación para nota anterior
    document.getElementById('previousScore').addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        if (e.target.value !== '' && (val < 1 || val > 5)) {
            e.target.value = Math.max(1, Math.min(5, val));
        }
    });
}

/**
 * Maneja el evento de predicción cuando el usuario presiona el botón
 */
function handlePredict() {
    // Obtener valores de los inputs
    const studyHours = parseFloat(document.getElementById('studyHours').value);
    const sleepHours = parseFloat(document.getElementById('sleepHours').value);
    const attendance = parseFloat(document.getElementById('attendance').value);
    const previousScore = parseFloat(document.getElementById('previousScore').value);

    // Validar que todos los campos estén completos
    if (!studyHours || !sleepHours || !attendance || !previousScore) {
        alert('Por favor completa todos los campos');
        return;
    }

    // Validar rangos
    if (studyHours < 1 || studyHours > 12) {
        alert('Las horas de estudio deben estar entre 1 y 12');
        return;
    }
    if (sleepHours < 4 || sleepHours > 9) {
        alert('Las horas de sueño deben estar entre 4 y 9');
        return;
    }
    if (attendance < 0 || attendance > 100) {
        alert('La asistencia debe estar entre 0 y 100');
        return;
    }
    if (previousScore < 1 || previousScore > 5) {
        alert('La nota anterior debe estar entre 1.0 y 5.0');
        return;
    }

    console.log('📥 Datos de entrada:', {
        studyHours,
        sleepHours,
        attendance,
        previousScore
    });

    // Preparar datos de entrada
    const input = [studyHours, sleepHours, attendance, previousScore];
    
    // Realizar predicción usando la red neuronal
    const result = model.predict(input);
    
    console.log('📤 Predicción raw:', result);
    
    // Procesar resultados
    const score = Math.min(5.0, Math.max(1.0, result.score)).toFixed(1);
    const probability = (result.probability * 100).toFixed(1);
    const passed = result.probability >= 0.5;

    console.log('✨ Resultados finales:', {
        score,
        probability: probability + '%',
        passed
    });

    // Mostrar resultados en la interfaz
    displayResults(score, probability, passed);
}

/**
 * Muestra los resultados en la interfaz
 * @param {string} score - Nota predicha
 * @param {string} probability - Probabilidad de aprobar
 * @param {boolean} passed - Si aprobó o no
 */
function displayResults(score, probability, passed) {
    // Actualizar nota estimada (Regresión)
    document.getElementById('estimatedScore').textContent = score;
    
    // Actualizar probabilidad (Clasificación)
    document.getElementById('probability').textContent = probability + '%';
    
    // Actualizar badge de estado
    const statusBadge = document.getElementById('statusBadge');
    if (passed) {
        statusBadge.innerHTML = `
            <div class="bg-green-100 text-green-800 py-3 px-4 rounded-lg font-bold text-lg text-center">
                ✓ Aprobado
            </div>
        `;
    } else {
        statusBadge.innerHTML = `
            <div class="bg-red-100 text-red-800 py-3 px-4 rounded-lg font-bold text-lg text-center">
                ✗ Reprobado
            </div>
        `;
    }

    // Mostrar la sección de resultados
    document.getElementById('results').style.display = 'block';
    
    // Hacer scroll suave hacia los resultados
    document.getElementById('results').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}