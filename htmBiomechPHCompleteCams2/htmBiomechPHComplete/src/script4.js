
  //video1-Cellphone
  const videoElement1 = document.getElementsByClassName('input_video')[0];
  const canvasElement = document.getElementsByClassName('output_canvas')[0];
  const canvasCtx = canvasElement.getContext('2d');
  //video2-WebCam
  const videoElement2 = document.getElementsByClassName('input_video2')[0];
  const canvasElement2 = document.getElementsByClassName('output_canvas2')[0];
  const canvasCtx2 = canvasElement2.getContext('2d');



// Función para listar los dispositivos de video disponibles
async function listVideoDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(device => device.kind === 'videoinput');
  
  console.log("Dispositivos de video disponibles:");
  videoDevices.forEach((device, index) => {
    console.log(`${index}: ${device.label || `Dispositivo ${index + 1}`} (ID: ${device.deviceId})`);
  });
  
  return videoDevices;
}
  


async function setupCamera() {
  const videoDevices = await listVideoDevices();
  
  // Buscar específicamente el dispositivo iVCam
  const iVCamDevice = videoDevices.find(device => device.label.includes('e2eSoft iVCam'));

  try {
    const constraints = {
      video: {
        deviceId: {exact: iVCamDevice.deviceId},
        width: {ideal: 1280},
        height: {ideal: 720}
      }
    };

   
    const stream = await navigator.mediaDevices.getUserMedia(constraints);

    videoElement1.srcObject = stream;
    videoElement1.onloadedmetadata = () => {
      videoElement1.play();

      // Inicializar Holistic después de que la cámara esté lista
      initHolistic();

    };
  




    // Dibujar el video en el canvas
    videoElement1.addEventListener('play', function() {
      function step() {
        canvasCtx.drawImage(videoElement1, 0, 0, canvasElement.width, canvasElement.height);
        requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });

  } catch (error) {}
     
 // Inicializar la segunda cámara (webcam)
 const holistic2 = new mpHolistic.Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
}});

/*

 */

holistic2.onResults(onResults2);

const camera2 = new Camera(videoElement2, {
  onFrame: async () => {
    await holistic2.send({image: videoElement2});
  },
  width: 1280,
  height: 720
});
camera2.start();

  }
  
  // Llama a esta función cuando tu página se cargue
  setupCamera();

// Función para inicializar Holistic
function initHolistic() {
  const holistic = new mpHolistic.Holistic({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
  }});

  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: true,
    smoothSegmentation: true,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  holistic.onResults(onResults);

  // Utilizar requestAnimationFrame para enviar el video al modelo Holistic
  async function sendFrameToHolistic() {
    await holistic.send({image: videoElement1});
    requestAnimationFrame(sendFrameToHolistic);
  }

  sendFrameToHolistic();
}



 

  const mpHolistic = window;
  const {drawConnectors, drawLandmarks} = window;
  const {POSE_CONNECTIONS, HAND_CONNECTIONS, FACEMESH_TESSELATION} = window;
  
  let results1 = null;
  let results2 = null;



  function removeElements(landmarks, elements) {
    for (const element of elements) {
      delete landmarks[element];
    }
  }
  
  function removeLandmarks(results) {
    if (results.poseLandmarks) {
      removeElements(results.poseLandmarks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
    }
  }
  
  function connect(ctx, connectors) {
    const canvas = ctx.canvas;
    for (const connector of connectors) {
      const from = connector[0];
      const to = connector[1];
      if (from && to) {
        if (from.visibility && to.visibility && (from.visibility < 0.1 || to.visibility < 0.1)) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
        ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
        ctx.stroke();
      }
    }
  }

  function onResults(results) {
    console.log("Procesando imagen:", results.image);
    results1 = results;
  // Usa videoElement1 directamente en lugar de results.image
  drawResults({...results, image: videoElement1}, canvasCtx, canvasElement);
  processResults();
  }



  function onResults2(results) {
  results2 = results;
  drawResults(results, canvasCtx2, canvasElement2); 
  processResults();
  }






  function drawResults(results, ctx, canvas) {
    removeLandmarks(results);
  
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Asegúrate de que results.image sea videoElement1
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 5;
    if (results.poseLandmarks) {
      if (results.rightHandLandmarks) {
        ctx.strokeStyle = 'white';
        connect(ctx, [[
          results.poseLandmarks[mpHolistic.POSE_LANDMARKS.RIGHT_ELBOW],
          results.rightHandLandmarks[0]
        ]]);
      }
      if (results.leftHandLandmarks) {
        ctx.strokeStyle = 'white';
        connect(ctx, [[
          results.poseLandmarks[mpHolistic.POSE_LANDMARKS.LEFT_ELBOW],
          results.leftHandLandmarks[0]
        ]]);
      }
    }
  
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS,
                   {color: '#00FF00', lineWidth: 4});
    drawLandmarks(ctx, results.poseLandmarks,
                  {color: '#FF0000', lineWidth: 2});
    drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION,
                   {color: '#C0C0C070', lineWidth: 1});
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS,
                   {color: '#CC0000', lineWidth: 5});
    drawLandmarks(ctx, results.leftHandLandmarks,
                  {color: '#00FF00', lineWidth: 2});
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS,
                   {color: '#00CC00', lineWidth: 5});
    drawLandmarks(ctx, results.rightHandLandmarks,
                  {color: '#FF0000', lineWidth: 2});
  
    ctx.restore();
  }



  function combineResults(results1, results2) {
    // Crear un nuevo objeto para almacenar los resultados combinados
    const combinedResults = {
      poseLandmarks: [],
      faceLandmarks: [],
      leftHandLandmarks: [],
      rightHandLandmarks: []
    };
  
    // Función para promediar dos landmarks
    function averageLandmarks(landmark1, landmark2) {
      return {
        x: (landmark1.x + landmark2.x) / 2,
        y: (landmark1.y + landmark2.y) / 2,
        z: (landmark1.z + landmark2.z) / 2,
        visibility: (landmark1.visibility + landmark2.visibility) / 2
      };
    }
  
    // Combinar poseLandmarks
    if (results1.poseLandmarks || results2.poseLandmarks) {
      const length = Math.max(
        results1.poseLandmarks?.length || 0,
        results2.poseLandmarks?.length || 0
      );
  
      for (let i = 0; i < length; i++) {
        if (results1.poseLandmarks && results2.poseLandmarks) {
          combinedResults.poseLandmarks.push(averageLandmarks(results1.poseLandmarks[i], results2.poseLandmarks[i]));
        } else if (results1.poseLandmarks) {
          combinedResults.poseLandmarks.push(results1.poseLandmarks[i]);
        } else if (results2.poseLandmarks) {
          combinedResults.poseLandmarks.push(results2.poseLandmarks[i]);
        }
      }
    }
  
    // Combinar faceLandmarks
    if (results1.faceLandmarks || results2.faceLandmarks) {
      const length = Math.max(
        results1.faceLandmarks?.length || 0,
        results2.faceLandmarks?.length || 0
      );
  
      for (let i = 0; i < length; i++) {
        if (results1.faceLandmarks && results2.faceLandmarks) {
          combinedResults.faceLandmarks.push(averageLandmarks(results1.faceLandmarks[i], results2.faceLandmarks[i]));
        } else if (results1.faceLandmarks) {
          combinedResults.faceLandmarks.push(results1.faceLandmarks[i]);
        } else if (results2.faceLandmarks) {
          combinedResults.faceLandmarks.push(results2.faceLandmarks[i]);
        }
      }
    }
  
    // Combinar leftHandLandmarks
    if (results1.leftHandLandmarks || results2.leftHandLandmarks) {
      const length = Math.max(
        results1.leftHandLandmarks?.length || 0,
        results2.leftHandLandmarks?.length || 0
      );
  
      for (let i = 0; i < length; i++) {
        if (results1.leftHandLandmarks && results2.leftHandLandmarks) {
          combinedResults.leftHandLandmarks.push(averageLandmarks(results1.leftHandLandmarks[i], results2.leftHandLandmarks[i]));
        } else if (results1.leftHandLandmarks) {
          combinedResults.leftHandLandmarks.push(results1.leftHandLandmarks[i]);
        } else if (results2.leftHandLandmarks) {
          combinedResults.leftHandLandmarks.push(results2.leftHandLandmarks[i]);
        }
      }
    }
  
    // Combinar rightHandLandmarks
    if (results1.rightHandLandmarks || results2.rightHandLandmarks) {
      const length = Math.max(
        results1.rightHandLandmarks?.length || 0,
        results2.rightHandLandmarks?.length || 0
      );
  
      for (let i = 0; i < length; i++) {
        if (results1.rightHandLandmarks && results2.rightHandLandmarks) {
          combinedResults.rightHandLandmarks.push(averageLandmarks(results1.rightHandLandmarks[i], results2.rightHandLandmarks[i]));
        } else if (results1.rightHandLandmarks) {
          combinedResults.rightHandLandmarks.push(results1.rightHandLandmarks[i]);
        } else if (results2.rightHandLandmarks) {
          combinedResults.rightHandLandmarks.push(results2.rightHandLandmarks[i]);
        }
      }
    }
  
    return combinedResults;
  }
  
  

  function processResults() {

    if (results1) {

  
      // Puedes usar combinedResults para dibujar en un canvas o para otra lógica
      drawResults(results1, canvasCtx, canvasElement);
      
      // Limpiar los resultados para el próximo frame
      results1 = null;
      results2 = null;
    }

    if (results2) {
    
      // Puedes usar combinedResults para dibujar en un canvas o para otra lógica
      drawResults(results2, canvasCtx2, canvasElement2);
      
      // Limpiar los resultados para el próximo frame
      results1 = null;
      results2 = null;
    }

    /*
    if (results1 && results2) {
      const combinedResults = combineResults(results1, results2);
  
      // Puedes usar combinedResults para dibujar en un canvas o para otra lógica
      drawResults(combinedResults, canvasCtx, canvasElement);
      
      // Limpiar los resultados para el próximo frame
      results1 = null;
      results2 = null;
    }
      */

  }

