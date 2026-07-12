import Foundation
import Vision
import CoreVideo

/// On-device hand-landmark detection via Apple Vision (VNDetectHumanHandPoseRequest),
/// exposed to Unity as C functions. Unity submits a BGRA camera frame (sensor orientation);
/// Vision runs asynchronously on a background queue; Unity polls the latest 21 landmarks.
/// Normalized output, origin TOP-left (Vision's bottom-left y is flipped). iOS 14+.
final class HandPoseVision {
    static let shared = HandPoseVision()

    private let lock = NSLock()
    private var xy = [Float](repeating: -1, count: 42)  // 21 * (x,y); -1 = joint missing
    private var count: Int32 = 0
    private var busy = false
    private let queue = DispatchQueue(label: "com.handmesh.handpose.vision")

    private let request: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 1
        return r
    }()

    // Model joint order (0..20): wrist, thumb(CMC..tip), index, middle, ring, little.
    private static let jointOrder: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip
    ]

    func submit(_ data: Data, _ w: Int, _ h: Int, _ bpr: Int) {
        lock.lock()
        if busy { lock.unlock(); return }   // drop this frame if still processing the last
        busy = true
        lock.unlock()
        queue.async {
            self.process(data, w, h, bpr)
            self.lock.lock(); self.busy = false; self.lock.unlock()
        }
    }

    private func process(_ data: Data, _ w: Int, _ h: Int, _ bpr: Int) {
        var pb: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs, &pb)
        guard let buf = pb else { setCount(0); return }

        CVPixelBufferLockBaseAddress(buf, [])
        if let dst = CVPixelBufferGetBaseAddress(buf) {
            let dstBpr = CVPixelBufferGetBytesPerRow(buf)
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                if let src = raw.baseAddress {
                    let n = min(bpr, dstBpr)
                    for row in 0..<h {
                        memcpy(dst.advanced(by: row * dstBpr), src.advanced(by: row * bpr), n)
                    }
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buf, [])

        let handler = VNImageRequestHandler(cvPixelBuffer: buf, orientation: .up, options: [:])
        do {
            try handler.perform([request])
            guard let obs = request.results?.first else { setCount(0); return }
            var tmp = [Float](repeating: -1, count: 42)
            var found = 0
            for (i, name) in HandPoseVision.jointOrder.enumerated() {
                if let p = try? obs.recognizedPoint(name), p.confidence > 0.3 {
                    tmp[i * 2]     = Float(p.location.x)         // normalized, left->right
                    tmp[i * 2 + 1] = Float(1.0 - p.location.y)   // Vision y is bottom-up -> top-left
                    found += 1
                }
            }
            lock.lock(); xy = tmp; count = found > 0 ? 21 : 0; lock.unlock()
        } catch {
            setCount(0)
        }
    }

    private func setCount(_ c: Int32) { lock.lock(); count = c; lock.unlock() }

    func get(_ out: UnsafeMutablePointer<Float>, _ maxPts: Int32) -> Int32 {
        lock.lock(); defer { lock.unlock() }
        let n = min(Int(maxPts), 21) * 2
        for i in 0..<n { out[i] = xy[i] }
        return count
    }
}

// ---- C entry points for Unity [DllImport("__Internal")] ----

@_cdecl("HandVision_Submit")
public func HandVision_Submit(_ ptr: UnsafeRawPointer, _ w: Int32, _ h: Int32, _ bpr: Int32) {
    let len = Int(bpr) * Int(h)
    let data = Data(bytes: ptr, count: len)   // copy so the async task owns the pixels
    HandPoseVision.shared.submit(data, Int(w), Int(h), Int(bpr))
}

@_cdecl("HandVision_Get")
public func HandVision_Get(_ out: UnsafeMutablePointer<Float>, _ maxPts: Int32) -> Int32 {
    return HandPoseVision.shared.get(out, maxPts)
}
