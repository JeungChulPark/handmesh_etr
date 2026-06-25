import UIKit

/// Transparent overlay that draws the 21-joint hand skeleton over the camera view.
final class SkeletonOverlayView: UIView {
    var joints: [CGPoint]?    // 21 screen-space points, or nil = no hand

    override func draw(_ rect: CGRect) {
        guard let pts = joints, pts.count == 21,
              let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.setLineWidth(3)
        for (a, b) in ARViewController.bones {
            ctx.setStrokeColor(UIColor.green.cgColor)
            ctx.move(to: pts[a]); ctx.addLine(to: pts[b]); ctx.strokePath()
        }
        ctx.setFillColor(UIColor.red.cgColor)
        for p in pts {
            ctx.fillEllipse(in: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6))
        }
    }
}
