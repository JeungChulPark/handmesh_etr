import CoreVideo
import CoreImage
import Accelerate
import simd

/// Builds the dual-stream model input from an ARKit RGB frame + LiDAR depth,
/// porting the Python pipeline (datasets/depth_synth.normalize_depth + the
/// 256 square-crop in infer_zed_dualstream.preprocess).
///
/// NOTE (scaffold): the Python `augmentation("test")` crop is a square bbox around
/// the hand, resized to 256, no flip/rotation. We reproduce that with a square
/// pixel bbox + bilinear(RGB)/nearest(depth) resample. Verify on the Mac.
enum Preprocess {

    static let size = 256
    static let normScale: Float = 0.1   // metres mapped to 1.0 (== norm_scale)

    /// A square pixel bbox around the hand, clamped to the image.
    struct BBox { var x: Int; var y: Int; var side: Int }

    /// Build the square crop bbox from Vision hand landmark pixels (margin frac
    /// ~0.5 matches infer_*.py --margin default).
    static func squareBBox(landmarksPx: [SIMD2<Float>], imgW: Int, imgH: Int,
                           marginFrac: Float = 0.5) -> BBox? {
        guard !landmarksPx.isEmpty else { return nil }
        var minX = Float.greatestFiniteMagnitude, minY = Float.greatestFiniteMagnitude
        var maxX: Float = 0, maxY: Float = 0
        for p in landmarksPx {
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }
        let w = maxX - minX, h = maxY - minY
        let side = max(w, h) * (1 + marginFrac)
        let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
        var x0 = Int(cx - side / 2), y0 = Int(cy - side / 2)
        var s = Int(side)
        x0 = max(0, x0); y0 = max(0, y0)
        s = max(8, min(s, min(imgW - x0, imgH - y0)))
        return BBox(x: x0, y: y0, side: s)
    }

    /// Crop a BGRA/ RGB CVPixelBuffer to the bbox and resample to 256x256 RGB,
    /// returning planar float [3*256*256] in 0..1 (channel-major, R,G,B).
    static func rgbCrop(_ pixelBuffer: CVPixelBuffer, bbox: BBox) -> [Float] {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let crop = ci.cropped(to: CGRect(x: bbox.x, y: bbox.y, width: bbox.side, height: bbox.side))
        let scale = CGFloat(size) / CGFloat(bbox.side)
        let scaled = crop.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let ctx = CIContext(options: nil)
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        ctx.render(scaled, toBitmap: &buf, rowBytes: size * 4,
                   bounds: CGRect(x: scaled.extent.minX, y: scaled.extent.minY,
                                  width: CGFloat(size), height: CGFloat(size)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        var planar = [Float](repeating: 0, count: 3 * size * size)
        let plane = size * size
        for i in 0..<plane {
            planar[0 * plane + i] = Float(buf[i * 4 + 0]) / 255.0  // R
            planar[1 * plane + i] = Float(buf[i * 4 + 1]) / 255.0  // G
            planar[2 * plane + i] = Float(buf[i * 4 + 2]) / 255.0  // B
        }
        return planar
    }

    /// Sample LiDAR depth (Float32 metres CVPixelBuffer, 0/NaN = invalid) over the
    /// SAME hand bbox (bbox is in RGB pixels; depthMap is lower-res, so we map by
    /// normalized coords). Returns (depthChannel[256*256] in [-1,1], depthMed metres).
    static func depthCropAndMedian(_ depthMap: CVPixelBuffer, bboxNorm: CGRect) -> ([Float], Float) {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        let dw = CVPixelBufferGetWidth(depthMap), dh = CVPixelBufferGetHeight(depthMap)
        let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        let base = CVPixelBufferGetBaseAddress(depthMap)!
        func depthAt(_ u: Int, _ v: Int) -> Float {
            let row = base.advanced(by: v * rowBytes).assumingMemoryBound(to: Float.self)
            return row[u]
        }
        // map the normalized bbox into depth-pixel space
        let dx0 = Int(bboxNorm.minX * CGFloat(dw)), dy0 = Int(bboxNorm.minY * CGFloat(dh))
        let dside = max(1, Int(bboxNorm.width * CGFloat(dw)))
        var raw = [Float](repeating: 0, count: size * size)
        var valid: [Float] = []
        valid.reserveCapacity(size * size)
        for j in 0..<size {
            for i in 0..<size {
                let su = dx0 + dside * i / size
                let sv = dy0 + dside * j / size
                var d: Float = 0
                if su >= 0 && su < dw && sv >= 0 && sv < dh {
                    d = depthAt(su, sv)
                    if !d.isFinite || d <= 0 { d = 0 }
                }
                raw[j * size + i] = d
                if d > 0 { valid.append(d) }
            }
        }
        let med = valid.isEmpty ? 0 : valid.sorted()[valid.count / 2]
        // normalize_depth: clip((d - med)/scale, -1, 1); background (0) stays 0
        var ch = [Float](repeating: 0, count: size * size)
        if med > 0 {
            for k in 0..<raw.count where raw[k] > 0 {
                ch[k] = max(-1, min(1, (raw[k] - med) / normScale))
            }
        }
        return (ch, med)
    }
}
