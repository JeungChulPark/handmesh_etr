import UIKit
import ARKit
import Vision
import simd

/// Drives the camera: ARKit sceneDepth (LiDAR) + RGB, Vision hand bbox, dual-stream
/// Core ML inference, and a skeleton overlay. Scaffold — wire into an iOS app target.
final class ARViewController: UIViewController, ARSessionDelegate {

    private let session = ARSession()
    private let overlay = SkeletonOverlayView()
    private var model: HandPoseModel!
    private let handRequest = VNDetectHumanHandPoseRequest()
    private var busy = false

    // MediaPipe/our 21-joint connections (wrist=0, then 5 fingers x4).
    static let bones: [(Int, Int)] = [
        (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    override func viewDidLoad() {
        super.viewDidLoad()
        model = try? HandPoseModel()
        overlay.frame = view.bounds
        overlay.backgroundColor = .clear
        view.addSubview(overlay)
        handRequest.maximumHandCount = 1
        session.delegate = self
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        let cfg = ARWorldTrackingConfiguration()
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            present(alert("This device has no LiDAR / sceneDepth."), animated: true); return
        }
        cfg.frameSemantics = .sceneDepth
        session.run(cfg)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard !busy, let model = model, let depth = frame.sceneDepth?.depthMap else { return }
        busy = true
        let rgb = frame.capturedImage
        let imgW = CVPixelBufferGetWidth(rgb), imgH = CVPixelBufferGetHeight(rgb)
        let intrinsics = frame.camera.intrinsics
        let viewSize = view.bounds.size

        DispatchQueue.global(qos: .userInitiated).async {
            defer { DispatchQueue.main.async { self.busy = false } }

            // --- Vision hand landmarks -> pixel bbox ---
            let handler = VNImageRequestHandler(cvPixelBuffer: rgb, orientation: .up)
            try? handler.perform([self.handRequest])
            guard let obs = self.handRequest.results?.first,
                  let pts = try? obs.recognizedPoints(.all) else {
                DispatchQueue.main.async { self.overlay.joints = nil; self.overlay.setNeedsDisplay() }
                return
            }
            // Vision points are normalized, origin bottom-left -> flip y to pixels.
            var lmPx: [SIMD2<Float>] = []
            for (_, p) in pts where p.confidence > 0.3 {
                lmPx.append(SIMD2<Float>(Float(p.location.x) * Float(imgW),
                                         (1 - Float(p.location.y)) * Float(imgH)))
            }
            guard let bbox = Preprocess.squareBBox(landmarksPx: lmPx, imgW: imgW, imgH: imgH) else { return }
            let bboxNorm = CGRect(x: CGFloat(bbox.x) / CGFloat(imgW),
                                  y: CGFloat(bbox.y) / CGFloat(imgH),
                                  width: CGFloat(bbox.side) / CGFloat(imgW),
                                  height: CGFloat(bbox.side) / CGFloat(imgH))

            // --- preprocess + infer ---
            let rgbPlanar = Preprocess.rgbCrop(rgb, bbox: bbox)
            let (depthCh, med) = Preprocess.depthCropAndMedian(depth, bboxNorm: bboxNorm)
            guard let joints = try? model.infer(rgbPlanar: rgbPlanar, depthChannel: depthCh, depthMed: med) else { return }

            // --- project camera-frame metres -> screen pts ---
            let fx = intrinsics[0,0], fy = intrinsics[1,1]
            let cx = intrinsics[2,0], cy = intrinsics[2,1]
            let sx = Float(viewSize.width) / Float(imgW), sy = Float(viewSize.height) / Float(imgH)
            var screen: [CGPoint] = []
            for p in joints {
                let z = max(p.z, 1e-3)
                let u = (p.x / z * fx + cx) * sx
                let v = (p.y / z * fy + cy) * sy
                screen.append(CGPoint(x: CGFloat(u), y: CGFloat(v)))
            }
            DispatchQueue.main.async {
                self.overlay.joints = screen
                self.overlay.setNeedsDisplay()
            }
        }
    }

    private func alert(_ msg: String) -> UIAlertController {
        let a = UIAlertController(title: "HandPoseLiDAR", message: msg, preferredStyle: .alert)
        a.addAction(UIAlertAction(title: "OK", style: .default))
        return a
    }
}
