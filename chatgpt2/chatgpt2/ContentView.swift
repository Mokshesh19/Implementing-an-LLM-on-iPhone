import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = GPT2ViewModel()

    var body: some View {
        VStack(spacing: 20) {
            Text(String(format: "Time taken: %.2f seconds", viewModel.outputTime)) // Display generation time
                .font(.headline)
                .padding()

            TextField("Enter your text", text: $viewModel.inputText)
                .padding()
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .frame(height: 50)
            
            Text(viewModel.outputText)
                .padding()
                .frame(maxWidth: .infinity, minHeight: 200, alignment: .topLeading)
                .background(Color.gray.opacity(0.2))
                .cornerRadius(8)

            HStack(spacing: 20) {
                Button(action: {
                    if !viewModel.isGenerating { // Prevent multiple generations
                        viewModel.generateResponse()
                    }
//                    viewModel.generateResponse()
                }) {
                    Text("Generate")
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.teal)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(viewModel.isGenerating)

                Button(action: {
                    viewModel.stopResponseGeneration()
                }) {
                    Text("Stop")
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.teal)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
            .padding(.horizontal)

            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

