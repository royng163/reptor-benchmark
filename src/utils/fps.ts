export const calculateFPS = (frameCount: number, elapsedTime: number): number => {
  if (elapsedTime <= 0) return 0;
  return (frameCount / elapsedTime) * 1000; // Convert to FPS
};

export const resetFPSCounter = () => {
  return {
    frameCount: 0,
    startTime: performance.now(),
  };
};

export const updateFPSCounter = (counter: { frameCount: number; startTime: number }) => {
  counter.frameCount += 1;
  const currentTime = performance.now();
  const elapsedTime = currentTime - counter.startTime;

  return {
    fps: calculateFPS(counter.frameCount, elapsedTime),
    counter,
  };
};
