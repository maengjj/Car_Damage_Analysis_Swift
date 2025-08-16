
import UIKit

// MARK: - UIImage Extension
extension UIImage {
    /// Returns a CGImage with orientation fixed (so Vision gets the right pixels)
    func fixedCGImage() -> CGImage? {
        if imageOrientation == .up, let cg = self.cgImage { return cg }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let drawn = renderer.image { _ in self.draw(in: CGRect(origin: .zero, size: size)) }
        return drawn.cgImage
    }
}
