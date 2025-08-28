
import SwiftUI
import CoreML
import Vision
import PhotosUI

// MARK: - ContentViewModel
@MainActor
class ContentViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var inputImage: UIImage?
    @Published var selectedItem: PhotosPickerItem? {
        didSet { Task { await loadImage() } }
    }
    @Published var maskScratch: UIImage?
    @Published var maskSeparated: UIImage?
    @Published var maskCrushed: UIImage?
    @Published var maskBreakage: UIImage?
    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var overlayOpacity: Double = 0.60
    @Published var selectedOverlays: Set<DamageOverlay> = []

    // MARK: - Private Properties
    private var damageClassIndex: Int = 1 // default: class 1 = damage

    // MARK: - Public Methods
    func runModel() async {
        guard let ui = inputImage, let cg = ui.fixedCGImage() else { return }
        isProcessing = true
        errorMessage = nil
        maskScratch = nil
        maskSeparated = nil
        maskCrushed = nil
        maskBreakage = nil
        defer { isProcessing = false }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all

            // ── 1) Scratch
            if let mask = try infer(makeWrapped: { try Damage_Scratch0_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.80, green: 0.10, blue: 0.10, alpha: 1.0), cgImage: cg) {
                self.maskScratch = mask
            }

            // ── 2) Separated
            if let mask = try infer(makeWrapped: { try Damage_Seperated1_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.10, green: 0.55, blue: 0.10, alpha: 1.0), cgImage: cg) {
                self.maskSeparated = mask
            }

            // ── 3) Crushed
            if let mask = try infer(makeWrapped: { try Damage_Crushed2_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.20, green: 0.35, blue: 0.80, alpha: 1.0), cgImage: cg) {
                self.maskCrushed = mask
            }

            // ── 4) Breakage
            if let mask = try infer(makeWrapped: { try Damage_Breakage3_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.75, green: 0.55, blue: 0.10, alpha: 1.0), cgImage: cg) {
                self.maskBreakage = mask
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func toggleOverlay(_ overlay: DamageOverlay) {
        if selectedOverlays.contains(overlay) {
            selectedOverlays.remove(overlay)
        } else {
            selectedOverlays.insert(overlay)
        }
    }

    func clearOverlays() {
        selectedOverlays.removeAll()
    }

    // MARK: - Private Methods
    private func loadImage() async {
        guard let item = selectedItem else { return }
        if let data = try? await item.loadTransferable(type: Data.self),
           let ui = UIImage(data: data) {
            self.inputImage = ui
        }
    }

    private func infer(makeWrapped: () throws -> MLModel, color: UIColor, cgImage: CGImage) throws -> UIImage? {
        let vnModel = try VNCoreMLModel(for: makeWrapped())
        var producedMask: UIImage?
        var producedError: Error?
        let request = VNCoreMLRequest(model: vnModel) { req, err in
            if let err { producedError = err }
            guard let obs = req.results?.first as? VNCoreMLFeatureValueObservation,
                  let multi = obs.featureValue.multiArrayValue else {
                return
            }
            if let mask = self.multiArrayToMask(multiArray: multi, positiveClass: self.damageClassIndex) {
                producedMask = self.colorizeMask(mask: mask, color: color)
            }
        }
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])
        if let producedError { throw producedError }
        return producedMask
    }

    private func multiArrayToMask(multiArray: MLMultiArray, positiveClass: Int = 1) -> UIImage? {
        let shape = multiArray.shape.map { $0.intValue }
        var C = 0, H = 0, W = 0
        switch shape.count {
        case 3: C = shape[0]; H = shape[1]; W = shape[2]
        case 4: C = shape[1]; H = shape[2]; W = shape[3]
        default: return nil
        }
        guard C >= 2 && H > 0 && W > 0 else { return nil }
        let positive = min(max(0, positiveClass), C - 1)

        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        let buffer = UnsafeBufferPointer(start: ptr, count: C * H * W)

        var mask = [UInt8](repeating: 0, count: H * W)
        for y in 0..<H {
            for x in 0..<W {
                var bestIdx: Int32 = 0
                var bestVal: Float32 = -.greatestFiniteMagnitude
                for cls in 0..<C {
                    let off = (cls * H + y) * W + x
                    let v = buffer[off]
                    if v > bestVal { bestVal = v; bestIdx = Int32(cls) }
                }
                let isPositive = (bestIdx == Int32(positive))
                mask[y * W + x] = isPositive ? 255 : 0
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: Data(mask) as CFData) else { return nil }
        guard let cg = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: W, space: colorSpace, bitmapInfo: CGBitmapInfo(), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cg)
    }

    private func colorizeMask(mask: UIImage, color: UIColor) -> UIImage? {
        guard let cg = mask.cgImage else { return nil }
        let W = cg.width, H = cg.height
        let bytesPerPixel = 4
        var rgba = [UInt8](repeating: 0, count: W * H * bytesPerPixel)

        guard let maskData = cg.dataProvider?.data as Data? else { return nil }
        maskData.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            let mptr = src.bindMemory(to: UInt8.self).baseAddress!
            let comps = color.cgColor.components ?? [1,0,0,1]
            let r = UInt8((comps.count > 0 ? comps[0] : 1) * 255)
            let g = UInt8((comps.count > 1 ? comps[1] : 0) * 255)
            let b = UInt8((comps.count > 2 ? comps[2] : 0) * 255)
            for i in 0..<(W*H) {
                let m = mptr[i]
                let base = i * 4
                rgba[base + 0] = r
                rgba[base + 1] = g
                rgba[base + 2] = b
                rgba[base + 3] = m
            }
        }

        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let outCG = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W * bytesPerPixel, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: outCG)
    }
}
