
import SwiftUI

// MARK: - CheckboxRow
struct CheckboxRow: View {
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
