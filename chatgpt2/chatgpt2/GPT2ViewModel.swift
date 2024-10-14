//
//  GPT2ViewModel.swift
//  chatgpt2
//
//  Created by Mokshesh Jain on 31/08/2024.
//

import SwiftUI
import CoreML

//class GPT2ViewModel: ObservableObject {
//    @Published var inputText: String = ""
//    @Published var outputText: String = "Model Output"
//
//    private var model: GPT2?
//
//    init() {
//        loadModel()
//    }
//
//    func loadModel() {
////        let modelPath = Bundle.main.path(forResource: "compgpt2", ofType: "mlpackage")!
////        model = try? MLModel(contentsOf: URL(fileURLWithPath: modelPath))
//        let model = GPT2(strategy: .topK(40))
//    }
//
//    func generateResponse() {
//        guard let model = model else {
//            outputText = "Model not loaded."
//            return
//        }
//        let tokenizer = GPT2Tokenizer()
//        let inputIds = tokenizer.encode(text: inputText)
//
//        // Assuming the model is set up to take input_ids and produce a string output
//        if let output = try? self.model.(from: ["input_ids": inputIds]) as? String{
//            outputText = output
//        } else {
//            outputText = "Failed to generate text."
//        }
//    }
//}


//import SwiftUI
import Combine

class GPT2ViewModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var outputText: String = "Model Output"
    @Published var outputTime: TimeInterval = 0
    @Published var isGenerating: Bool = false


    private var model: GPT2?

    init() {
        loadModel()
    }

    func loadModel() {
        // Initialize the GPT2 model with the same strategy as in the ViewController
        model = GPT2(strategy: .topK(40))
    }

    func generateResponse() {
        guard let model = model else {
            outputText = "Model not loaded."
            return
        }
        guard !isGenerating else { return }
        isGenerating = true
    
        
        // Ensure UI updates happen on the main thread
        DispatchQueue.global(qos: .userInitiated).async {
            let response = model.generate(text: self.inputText, nTokens: 64) { completion, time in DispatchQueue.main.async {
//                    self.outputText = completion
                    self.outputText = self.inputText + completion
                    self.outputTime = time
                    self.isGenerating = false
//                    self.speedLabel.text = String(format: "%.2f", 1 / time)
                }
            }
        }
    }
    
    func stopResponseGeneration() {
            model?.stopGeneration()
            isGenerating = false
        }
}
