import SwiftUI
import PhotosUI

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var viewModel = ContentViewModel()

    private var hasAnyMask: Bool {
        viewModel.maskScratch != nil ||
        viewModel.maskSeparated != nil ||
        viewModel.maskCrushed != nil ||
        viewModel.maskBreakage != nil
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                imageDisplay
                controlButtons
                if hasAnyMask {
                    overlayControls
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                errorMessageDisplay
                Spacer()
            }
            .padding()
            .animation(.easeInOut(duration: 0.25), value: hasAnyMask)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text("자동차 파손 판별 AI")
                        .font(.title.bold()) // 원하는 크기 지정
                }
            }
        }
        
    }

    private var imageDisplay: some View {
        ZStack {
            Rectangle()
                .fill(Color(UIColor.secondarySystemBackground))
                .aspectRatio(1, contentMode: .fit)
                .overlay(
                    Group {
                        if let ui = viewModel.inputImage {
                            Image(uiImage: ui)
                                .resizable()
                                .scaledToFit()
                        } else {
                            Image("logoCar")
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: 300)
                        }
                    }
                )

            overlayViews
        }
    }

    private var overlayViews: some View {
        Group {
            if viewModel.selectedOverlays.contains(.scratch), let mask = viewModel.maskScratch {
                Image(uiImage: mask)
                    .resizable()
                    .scaledToFit()
                    .opacity(viewModel.overlayOpacity)
                    .allowsHitTesting(false)
                    .blendMode(.multiply)
            }
            if viewModel.selectedOverlays.contains(.separated), let mask = viewModel.maskSeparated {
                Image(uiImage: mask)
                    .resizable()
                    .scaledToFit()
                    .opacity(viewModel.overlayOpacity)
                    .allowsHitTesting(false)
                    .blendMode(.multiply)
            }
            if viewModel.selectedOverlays.contains(.crushed), let mask = viewModel.maskCrushed {
                Image(uiImage: mask)
                    .resizable()
                    .scaledToFit()
                    .opacity(viewModel.overlayOpacity)
                    .allowsHitTesting(false)
                    .blendMode(.multiply)
            }
            if viewModel.selectedOverlays.contains(.breakage), let mask = viewModel.maskBreakage {
                Image(uiImage: mask)
                    .resizable()
                    .scaledToFit()
                    .opacity(viewModel.overlayOpacity)
                    .allowsHitTesting(false)
                    .blendMode(.multiply)
            }
        }
    }

    private var controlButtons: some View {
        HStack {
            PhotosPicker(selection: $viewModel.selectedItem, matching: .images, photoLibrary: .shared()) {
                Label("사진 선택", systemImage: "photo.on.rectangle")
            }
            .buttonStyle(.bordered)

            Button {
                Task { await viewModel.runModel() }
            } label: {
                if viewModel.isProcessing {
                    ProgressView()
                        .progressViewStyle(.circular)
                } else {
                    Label("판별 실행", systemImage: "magnifyingglass")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.inputImage == nil || viewModel.isProcessing)
        }
    }

    private var overlayControls: some View {
        VStack(spacing: 12) {
            // 손상 유형 선택 영역 (Chip 스타일)
            VStack(alignment: .leading, spacing: 6) {
                Text("손상 유형")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        // None (전체 해제)
                        Chip(
                            title: "None",
                            isSelected: viewModel.selectedOverlays.isEmpty,
                            color: .secondary
                        ) {
                            viewModel.clearOverlays()
                        }

                        Chip(
                            title: "Scratch",
                            isSelected: viewModel.selectedOverlays.contains(.scratch),
                            color: Color(red: 0.80, green: 0.10, blue: 0.10)
                        ) {
                            viewModel.toggleOverlay(.scratch)
                        }

                        Chip(
                            title: "Separated",
                            isSelected: viewModel.selectedOverlays.contains(.separated),
                            color: Color(red: 0.10, green: 0.55, blue: 0.10)
                        ) {
                            viewModel.toggleOverlay(.separated)
                        }

                        Chip(
                            title: "Crushed",
                            isSelected: viewModel.selectedOverlays.contains(.crushed),
                            color: Color(red: 0.20, green: 0.35, blue: 0.80)
                        ) {
                            viewModel.toggleOverlay(.crushed)
                        }

                        Chip(
                            title: "Breakage",
                            isSelected: viewModel.selectedOverlays.contains(.breakage),
                            color: Color(red: 0.75, green: 0.55, blue: 0.10)
                        ) {
                            viewModel.toggleOverlay(.breakage)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }

            // 오버레이 투명도 조절
            VStack(spacing: 6) {
                HStack {
                    Text("오버레이 투명도")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(viewModel.overlayOpacity * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $viewModel.overlayOpacity, in: 0...1)
            }
        }
    }

    private var errorMessageDisplay: some View {
        Group {
            if let err = viewModel.errorMessage {
                Text(err)
                    .foregroundColor(.red)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
    }
}


struct Chip: View {
    let title: String
    let isSelected: Bool
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.bold())
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .frame(height: 34)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .background(
            Capsule()
                .fill(isSelected ? color.opacity(0.9) : Color(UIColor.systemGray6))
        )
        .foregroundStyle(isSelected ? Color.white : Color.primary)
        .overlay(
            Capsule()
                .stroke(isSelected ? Color.clear : Color(UIColor.separator), lineWidth: 1)
        )
        .shadow(color: isSelected ? .black.opacity(0.08) : .clear, radius: 2, x: 0, y: 1)
        .accessibilityLabel(Text(title + (isSelected ? " 선택됨" : "")))
    }
}

#Preview {
    ContentView()
}
