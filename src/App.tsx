import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';

const YOLO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

interface Detection {
  class: string;
  classNumber: number;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

function App() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastProcessedTimeRef = useRef(0);

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedSession = await ort.InferenceSession.create('./yolov10n_web.onnx', { executionProviders: ['wasm'] });
        setSession(loadedSession);
        setIsLoading(false);
        console.log('YOLO model loaded successfully');
      } catch (error) {
        console.error('Failed to load YOLO model:', error);
        setIsLoading(false);
      }
    }
    loadModel();

    // Set up webcam
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play().catch(e => console.log("Play error:", e));
            };
          }
        })
        .catch(error => console.error("Error accessing webcam:", error));
    }
  }, []);

  useEffect(() => {
    if (!session) return;

    const detectFrame = async (time: number) => {
      if (videoRef.current && canvasRef.current && session &&
        videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {

        // Limit to ~9 FPS
        if (time - lastProcessedTimeRef.current > 110) {
          try {
            const tensor = imageDataToTensor(videoRef.current);
            const results = await session.run({ images: tensor });
            const detectedObjects = await processResults(results);
            setDetections(detectedObjects);
            lastProcessedTimeRef.current = time;
          } catch (error) {
            console.error('Error during inference:', error);
            setDetections([]);
          }
        }
      }
      requestAnimationFrame(detectFrame);
    };

    requestAnimationFrame(detectFrame);
  }, [session]);

  if (isLoading) {
    return <div>Loading YOLO model...</div>;
  }

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline muted style={{ display: 'block', width: '640px', height: '480px' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} width={640} height={640} />
      <h2>Detections:</h2>
      <ul>
        {detections.map((item, index) => (
          <li key={index}>
            {`Class ${item.classNumber} (${item.class}): ${item.confidence.toFixed(4)} - 
            Position: (${item.x.toFixed(2)}, ${item.y.toFixed(2)}) - 
            Size: ${item.width.toFixed(2)}x${item.height.toFixed(2)}`}
          </li>
        ))}
      </ul>
      <h2>Debug Info:</h2>
      <pre>{JSON.stringify(detections, null, 2)}</pre>
    </div>
  );

  function imageDataToTensor(video: HTMLVideoElement): ort.Tensor {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Could not get 2D context');
    }
  
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 640, 640);
    
    const imageData = ctx.getImageData(0, 0, 640, 640);
    const input = new Float32Array(1 * 3 * 640 * 640);
  
    for (let i = 0; i < imageData.data.length; i += 4) {
      input[i / 4] = imageData.data[i] / 255.0;
      input[i / 4 + 640 * 640] = imageData.data[i + 1] / 255.0;
      input[i / 4 + 2 * 640 * 640] = imageData.data[i + 2] / 255.0;
    }
  
    return new ort.Tensor('float32', input, [1, 3, 640, 640]);
  }

  async function processResults(results: ort.InferenceSession.OnnxValueMapType): Promise<Detection[]> {
    const output = results.output0.data as Float32Array;
    const [batch, boxes, data] = results.output0.dims;

    const detections: Detection[] = [];
    const imgWidth = 640;
    const imgHeight = 640;

    for (let i = 0; i < boxes; i++) {
      const slicedOutput = output.slice(i * data, (i + 1) * data);
      const x = slicedOutput[0];
      const y = slicedOutput[1];
      const w = slicedOutput[2];
      const h = slicedOutput[3];
      const confidence = slicedOutput[4];
      const classScores = Array.from(slicedOutput.slice(5));

      const maxScore = Math.max(...classScores);
      const maxScoreIndex = classScores.indexOf(maxScore);

      if (confidence > 0.5) { // Confidence threshold
        detections.push({
          class: YOLO_CLASSES[maxScoreIndex] || 'unknown',
          classNumber: maxScoreIndex,
          confidence: confidence,
          x: (x - w / 2) * imgWidth,
          y: (y - h / 2) * imgHeight,
          width: w * imgWidth,
          height: h * imgHeight
        });
      }
    }

    // Apply Non-Maximum Suppression
    return nonMaxSuppression(detections, 0.5); // IOU threshold
  }

  function nonMaxSuppression(boxes: Detection[], iouThreshold: number): Detection[] {
    return boxes.sort((a, b) => b.confidence - a.confidence)
      .filter((box, index, array) =>
        index === array.findIndex((otherBox) =>
          calculateIOU(box, otherBox) > iouThreshold
        )
      );
  }

  function calculateIOU(box1: Detection, box2: Detection): number {
    const intersectionX = Math.max(box1.x, box2.x);
    const intersectionY = Math.max(box1.y, box2.y);
    const intersectionW = Math.min(box1.x + box1.width, box2.x + box2.width) - intersectionX;
    const intersectionH = Math.min(box1.y + box1.height, box2.y + box2.height) - intersectionY;

    if (intersectionW <= 0 || intersectionH <= 0) return 0;

    const intersectionArea = intersectionW * intersectionH;
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;

    return intersectionArea / (box1Area + box2Area - intersectionArea);
  }
}


// Export the App component as the default export
export default App;