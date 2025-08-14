//
//  ContentView.swift
//  car_damage
//
//  Created by JiJooMaeng on 8/12/25.
//

import SwiftUI
import CoreML
import Vision
import PhotosUI

// MARK: - DamageOverlay enum
private enum DamageOverlay: String, CaseIterable {
    case none
    case scratch
    case separated
    case crushed
    case breakage
}

// MARK: - ContentView
struct ContentView: View {
    @State private var inputImage: UIImage?
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var maskScratch: UIImage?
    @State private var maskSeparated: UIImage?
    @State private var maskCrushed: UIImage?
    @State private var maskBreakage: UIImage?
    @State private var isProcessing = false
    @State private var errorMessage: String?
    // Overlay controls (multi selection)
    @State private var overlayOpacity: Double = 0.35
    @State private var selectedOverlays: Set<DamageOverlay> = []
    @State private var damageClassIndex: Int = 1 // default: class 1 = damage
    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                ZStack {
                    Rectangle()
                        .fill(Color(UIColor.secondarySystemBackground))
                        .aspectRatio(1, contentMode: .fit)
                        .overlay(
                            Group {
                                if let ui = inputImage {
                                    Image(uiImage: ui)
                                        .resizable()
                                        .scaledToFit()
                                } else {
                                    Text("사진을 선택하세요")
                                        .foregroundStyle(.secondary)
                                }
                            }
                        )

                    Group {
                        if selectedOverlays.contains(.scratch), let mask = maskScratch {
                            Image(uiImage: mask)
                                .resizable()
                                .scaledToFit()
                                .opacity(overlayOpacity)
                                .allowsHitTesting(false)
                                .blendMode(.multiply)
                        }
                        if selectedOverlays.contains(.separated), let mask = maskSeparated {
                            Image(uiImage: mask)
                                .resizable()
                                .scaledToFit()
                                .opacity(overlayOpacity)
                                .allowsHitTesting(false)
                                .blendMode(.multiply)
                        }
                        if selectedOverlays.contains(.crushed), let mask = maskCrushed {
                            Image(uiImage: mask)
                                .resizable()
                                .scaledToFit()
                                .opacity(overlayOpacity)
                                .allowsHitTesting(false)
                                .blendMode(.multiply)
                        }
                        if selectedOverlays.contains(.breakage), let mask = maskBreakage {
                            Image(uiImage: mask)
                                .resizable()
                                .scaledToFit()
                                .opacity(overlayOpacity)
                                .allowsHitTesting(false)
                                .blendMode(.multiply)
                        }
                    }
                }

                HStack {
                    PhotosPicker(selection: $selectedItem, matching: .images, photoLibrary: .shared()) {
                        Label("사진 선택", systemImage: "photo.on.rectangle")
                    }
                    .buttonStyle(.bordered)

                    Button {
                        Task { await runModel() }
                    } label: {
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(.circular)
                        } else {
                            Label("판별 실행", systemImage: "magnifyingglass")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(inputImage == nil || isProcessing)
                }

                // Checkbox-style selector (multi visible overlays)
                VStack(spacing: 8) {
                    HStack(spacing: 16) {
                        // None: clear all selections
                        CheckboxRow(title: "None", color: .secondary, checked: selectedOverlays.isEmpty) {
                            selectedOverlays.removeAll()
                        }
                        // Scratch
                        CheckboxRow(title: "Scratch", color: Color(red: 0.80, green: 0.10, blue: 0.10), checked: selectedOverlays.contains(.scratch)) {
                            if selectedOverlays.contains(.scratch) {
                                selectedOverlays.remove(.scratch)
                            } else {
                                selectedOverlays.insert(.scratch)
                            }
                        }
                        // Separated
                        CheckboxRow(title: "Separated", color: Color(red: 0.10, green: 0.55, blue: 0.10), checked: selectedOverlays.contains(.separated)) {
                            if selectedOverlays.contains(.separated) {
                                selectedOverlays.remove(.separated)
                            } else {
                                selectedOverlays.insert(.separated)
                            }
                        }
                        // Crushed
                        CheckboxRow(title: "Crushed", color: Color(red: 0.20, green: 0.35, blue: 0.80), checked: selectedOverlays.contains(.crushed)) {
                            if selectedOverlays.contains(.crushed) {
                                selectedOverlays.remove(.crushed)
                            } else {
                                selectedOverlays.insert(.crushed)
                            }
                        }
                        // Breakage
                        CheckboxRow(title: "Breakage", color: Color(red: 0.75, green: 0.55, blue: 0.10), checked: selectedOverlays.contains(.breakage)) {
                            if selectedOverlays.contains(.breakage) {
                                selectedOverlays.remove(.breakage)
                            } else {
                                selectedOverlays.insert(.breakage)
                            }
                        }
                    }
                    .font(.footnote)
                    .frame(maxWidth: .infinity, alignment: .leading)

                    HStack {
                        Text("투명도")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Slider(value: $overlayOpacity, in: 0...1)
                    }
                }

                if let err = errorMessage {
                    Text(err)
                        .foregroundColor(.red)
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("차량파손판별 AI")
            .onChange(of: selectedItem) { _, newValue in
                guard let item = newValue else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let ui = UIImage(data: data) {
                        self.inputImage = ui
                    }
                }
            }
        }
    }

    // MARK: - Inference
    @MainActor
    private func runModel() async {
        guard let ui = inputImage, let cg = ui.fixedCGImage() else { return }
        isProcessing = true
        errorMessage = nil
        maskScratch = nil
        maskSeparated = nil
        maskCrushed = nil
        maskBreakage = nil
        defer { isProcessing = false }

        do {
            // Common config
            let config = MLModelConfiguration()
            config.computeUnits = .all

            // Local helper to run a VNCoreMLRequest synchronously and return a colorized mask
            func infer(makeWrapped: () throws -> MLModel, color: UIColor) throws -> UIImage? {
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

                let handler = VNImageRequestHandler(cgImage: cg, options: [:])
                try handler.perform([request])
                if let producedError { throw producedError }
                return producedMask
            }

            // ── 1) Scratch
            if let mask = try infer(makeWrapped: { try Damage_Scratch0_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.80, green: 0.10, blue: 0.10, alpha: 1.0)) {
                self.maskScratch = mask
            }

            // ── 2) Separated
            if let mask = try infer(makeWrapped: { try Damage_Seperated1_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.10, green: 0.55, blue: 0.10, alpha: 1.0)) {
                self.maskSeparated = mask
            }

            // ── 3) Crushed
            if let mask = try infer(makeWrapped: { try Damage_Crushed2_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.20, green: 0.35, blue: 0.80, alpha: 1.0)) {
                self.maskCrushed = mask
            }

            // ── 4) Breakage
            if let mask = try infer(makeWrapped: { try Damage_Breakage3_Unet_256(configuration: config).model },
                                     color: UIColor(red: 0.75, green: 0.55, blue: 0.10, alpha: 1.0)) {
                self.maskBreakage = mask
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

// MARK: - Helpers
private extension ContentView {
    /// Convert MLMultiArray logits [1, C, H, W] or [C, H, W] to a single-channel 8-bit mask via argmax
    func multiArrayToMask(multiArray: MLMultiArray, positiveClass: Int = 1) -> UIImage? {
        // Infer shape flexibly
        let shape = multiArray.shape.map { $0.intValue }
        var C = 0, H = 0, W = 0
        switch shape.count {
        case 3: // [C,H,W]
            C = shape[0]; H = shape[1]; W = shape[2]
        case 4: // [N,C,H,W] with N=1
            C = shape[1]; H = shape[2]; W = shape[3]
        default:
            return nil
        }
        guard C >= 2 && H > 0 && W > 0 else { return nil }
        let positive = min(max(0, positiveClass), C - 1)

        // Read as Float32
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        let count = C * H * W
        let buffer = UnsafeBufferPointer(start: ptr, count: count)

        // Argmax along C
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
                // Mark pixels belonging to the selected class (default: class 1) as 255
                let isPositive = (bestIdx == Int32(positive))
                mask[y * W + x] = isPositive ? 255 : 0
            }
        }

        // Build CGImage (grayscale)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: Data(mask) as CFData) else { return nil }
        guard let cg = CGImage(
            width: W,
            height: H,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: W,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: cg)
    }

    /// Colorize a grayscale (0/255) mask to a solid color with alpha channel
    func colorizeMask(mask: UIImage, color: UIColor) -> UIImage? {
        guard let cg = mask.cgImage else { return nil }
        let W = cg.width, H = cg.height
        let bytesPerPixel = 4
        var rgba = [UInt8](repeating: 0, count: W * H * bytesPerPixel)

        // Read mask bytes
        guard let maskData = cg.dataProvider?.data as Data? else { return nil }
        maskData.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            let mptr = src.bindMemory(to: UInt8.self).baseAddress!
            let comps = color.cgColor.components ?? [1,0,0,1] // default red
            let r = UInt8((comps.count > 0 ? comps[0] : 1) * 255)
            let g = UInt8((comps.count > 1 ? comps[1] : 0) * 255)
            let b = UInt8((comps.count > 2 ? comps[2] : 0) * 255)
            for i in 0..<(W*H) {
                let m = mptr[i] // 0 or 255
                let base = i * 4
                rgba[base + 0] = r
                rgba[base + 1] = g
                rgba[base + 2] = b
                rgba[base + 3] = m // alpha from mask
            }
        }

        // Build CGImage directly from the RGBA buffer
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let outCG = CGImage(
            width: W,
            height: H,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: W * bytesPerPixel,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: outCG)
    }
}

// MARK: - Utilities
private extension UIImage {
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

private struct CheckboxRow: View {
    let title: String
    let color: Color
    let checked: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: checked ? "checkmark.square.fill" : "square")
                Text(title)
            }
            .foregroundStyle(color)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    ContentView()
}
