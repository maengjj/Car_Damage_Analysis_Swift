import SwiftUI
import PhotosUI

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var viewModel = ContentViewModel()

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                imageDisplay
                controlButtons
                overlayControls
                errorMessageDisplay
                Spacer()
            }
            .padding()
            .navigationTitle("차량파손판별 AI")
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
                            Text("사진을 선택하세요")
                                .foregroundStyle(.secondary)
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
        VStack(spacing: 8) {
            HStack(spacing: 16) {
                CheckboxRow(title: "None", color: .secondary, checked: viewModel.selectedOverlays.isEmpty) {
                    viewModel.clearOverlays()
                }
                CheckboxRow(title: "Scratch", color: Color(red: 0.80, green: 0.10, blue: 0.10), checked: viewModel.selectedOverlays.contains(.scratch)) {
                    viewModel.toggleOverlay(.scratch)
                }
                CheckboxRow(title: "Separated", color: Color(red: 0.10, green: 0.55, blue: 0.10), checked: viewModel.selectedOverlays.contains(.separated)) {
                    viewModel.toggleOverlay(.separated)
                }
                CheckboxRow(title: "Crushed", color: Color(red: 0.20, green: 0.35, blue: 0.80), checked: viewModel.selectedOverlays.contains(.crushed)) {
                    viewModel.toggleOverlay(.crushed)
                }
                CheckboxRow(title: "Breakage", color: Color(red: 0.75, green: 0.55, blue: 0.10), checked: viewModel.selectedOverlays.contains(.breakage)) {
                    viewModel.toggleOverlay(.breakage)
                }
            }
            .font(.footnote)
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack {
                Text("투명도")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
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

#Preview {
    ContentView()
}