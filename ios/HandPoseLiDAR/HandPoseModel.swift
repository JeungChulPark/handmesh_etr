import CoreML

/// Core ML wrapper for the dual-stream hand model (ds_lidarsim.mlpackage).
///
/// Xcode auto-generates a `ds_lidarsim` class from the .mlpackage, but we use the
/// generic MLModel API here so the file compiles regardless of the generated
/// class name — swap to the generated class for type-safety once it exists.
final class HandPoseModel {

    private let model: MLModel
    private let size = Preprocess.size

    init() throws {
        // The .mlpackage compiles to ds_lidarsim.mlmodelc in the bundle.
        guard let url = Bundle.main.url(forResource: "ds_lidarsim", withExtension: "mlmodelc") else {
            throw NSError(domain: "HandPoseModel", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "ds_lidarsim.mlmodelc not found in bundle"])
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all          // ANE + GPU + CPU
        self.model = try MLModel(contentsOf: url, configuration: cfg)
    }

    /// rgbPlanar: [3*256*256] (R,G,B planes, 0..1); depthChannel: [256*256] in [-1,1].
    /// Returns absolute camera-frame joints (21 x SIMD3, metres) from keypoints_abs.
    func infer(rgbPlanar: [Float], depthChannel: [Float], depthMed: Float) throws -> [SIMD3<Float>] {
        let plane = size * size
        let image = try MLMultiArray(shape: [1, 4, NSNumber(value: size), NSNumber(value: size)],
                                     dataType: .float32)
        let ptr = image.dataPointer.assumingMemoryBound(to: Float.self)
        // channels 0..2 = RGB, channel 3 = depth
        for i in 0..<(3 * plane) { ptr[i] = rgbPlanar[i] }
        for i in 0..<plane { ptr[3 * plane + i] = depthChannel[i] }

        let med = try MLMultiArray(shape: [1, 1], dataType: .float32)
        med[0] = NSNumber(value: depthMed)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: image),
            "depth_med": MLFeatureValue(multiArray: med),
        ])
        let out = try model.prediction(from: input)
        guard let absJ = out.featureValue(for: "keypoints_abs")?.multiArrayValue else {
            throw NSError(domain: "HandPoseModel", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "missing keypoints_abs output"])
        }
        var joints = [SIMD3<Float>](repeating: .zero, count: 21)
        for j in 0..<21 {
            joints[j] = SIMD3<Float>(absJ[[0, j, 0] as [NSNumber]].floatValue,
                                     absJ[[0, j, 1] as [NSNumber]].floatValue,
                                     absJ[[0, j, 2] as [NSNumber]].floatValue)
        }
        return joints
    }
}
