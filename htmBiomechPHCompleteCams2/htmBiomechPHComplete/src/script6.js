// Función para verificar si un landmark es válido
function isValidLandmark(landmark) {
  return landmark && landmark.visibility > 0.5; // Ajusta el umbral de visibilidad según sea necesario
}

// Función para calcular el promedio ponderado entre dos landmarks
function weightedAverageLandmarks(landmark1, landmark2) {
  const weight1 = landmark1.visibility || 0;
  const weight2 = landmark2.visibility || 0;
  const totalWeight = weight1 + weight2;

  if (totalWeight === 0) return null; // Ambos landmarks son inválidos

  return {
    x: (landmark1.x * weight1 + landmark2.x * weight2) / totalWeight,
    y: (landmark1.y * weight1 + landmark2.y * weight2) / totalWeight,
    z: (landmark1.z * weight1 + landmark2.z * weight2) / totalWeight,
    visibility: totalWeight / 2 // Promedio de visibilidad
  };
}

// Función para combinar un conjunto de landmarks
function combineLandmarks(landmarks1, landmarks2) {
  const length = Math.max(landmarks1?.length || 0, landmarks2?.length || 0);
  const combined = [];

  for (let i = 0; i < length; i++) {
    const l1 = landmarks1?.[i];
    const l2 = landmarks2?.[i];

    if (isValidLandmark(l1) && isValidLandmark(l2)) {
      combined.push(weightedAverageLandmarks(l1, l2));
    } else if (isValidLandmark(l1)) {
      combined.push(l1);
    } else if (isValidLandmark(l2)) {
      combined.push(l2);
    } else {
      combined.push(null); // Ningún landmark válido
    }
  }

  return combined;
}

// Función principal para combinar resultados de dos cámaras
function combineResults(results1, results2) {
  return {
    poseLandmarks: combineLandmarks(results1.poseLandmarks, results2.poseLandmarks),
    faceLandmarks: combineLandmarks(results1.faceLandmarks, results2.faceLandmarks),
    leftHandLandmarks: combineLandmarks(results1.leftHandLandmarks, results2.leftHandLandmarks),
    rightHandLandmarks: combineLandmarks(results1.rightHandLandmarks, results2.rightHandLandmarks)
  };
}

// Función para procesar resultados y dibujar combinados
function processResults() {
  if (results1 && results2) {
    const combinedResults = combineResults(results1, results2);
    drawResults(combinedResults, canvasCtx, canvasElement);
  }
}

// Ejemplo de cómo inicializar los resultados de las cámaras
let results1 = null; // Resultados de la primera cámara
let results2 = null; // Resultados de la segunda cámara

// Contexto y elemento del lienzo
const canvasElement = document.getElementById("outputCanvas");
const canvasCtx = canvasElement.getContext("2d");

// Función para simular el dibujo de resultados en el lienzo
function drawResults(results, context, canvas) {
  if (!results) return;

  context.clearRect(0, 0, canvas.width, canvas.height);

  // Dibuja landmarks de pose
  results.poseLandmarks?.forEach(landmark => {
    if (landmark) {
      context.beginPath();
      context.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
      context.fillStyle = "rgba(0, 255, 0, 0.7)";
      context.fill();
    }
  });

  // Dibuja landmarks adicionales (cara, manos) según sea necesario...
}

// Simula la recepción de datos de las cámaras
function updateCameraResults(cam1Data, cam2Data) {
  results1 = cam1Data;
  results2 = cam2Data;
  processResults();
}

