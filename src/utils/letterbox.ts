export interface LetterboxParams {
  inputSize: number; // model square input (e.g., 256/384/640)
  scale: number; // scale applied to src -> resized
  dx: number; // left padding in pixels in model input space
  dy: number; // top padding in pixels in model input space
  resized: { width: number; height: number };
}

/**
 * Compute centered letterbox to fit src into a square inputSize.
 * Returns scale and padding for accurate coordinate unprojection.
 */
export function computeLetterbox(srcW: number, srcH: number, inputSize: number): LetterboxParams {
  const s = Math.min(inputSize / srcW, inputSize / srcH);
  const newW = Math.round(srcW * s);
  const newH = Math.round(srcH * s);
  const dx = Math.floor((inputSize - newW) / 2);
  const dy = Math.floor((inputSize - newH) / 2);
  return { inputSize, scale: s, dx, dy, resized: { width: newW, height: newH } };
}

/**
 * Map point from model input space back to original source pixel space.
 * If model outputs normalized coords [0..1], set normalized=true.
 */
export function mapFromLetterbox(
  x: number,
  y: number,
  srcW: number,
  srcH: number,
  p: LetterboxParams,
  normalized = false
): { x: number; y: number } {
  const px = normalized ? x * p.inputSize : x;
  const py = normalized ? y * p.inputSize : y;
  const ox = Math.max(0, Math.min(srcW, (px - p.dx) / p.scale));
  const oy = Math.max(0, Math.min(srcH, (py - p.dy) / p.scale));
  return { x: ox, y: oy };
}
