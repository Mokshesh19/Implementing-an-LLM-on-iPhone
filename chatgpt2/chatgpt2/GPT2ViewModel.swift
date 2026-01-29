//
//  GPT2ViewModel.swift
//  chatgpt2
//
//  Created by Mokshesh Jain on 31/08/2024.
//

import SwiftUI
import CoreML
import Combine

class GPT2ViewModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var outputText: String = "Model Output"
    @Published var outputTime: TimeInterval = 0
    @Published var isGenerating: Bool = false
    @Published var errorMessage: String?

    private var memoryWarningObserver: NSObjectProtocol?

    init() {
        setupMemoryWarningObserver()
    }

    deinit {
        if let observer = memoryWarningObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    /// Set up observer for memory warnings
    private func setupMemoryWarningObserver() {
        #if os(iOS)
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryWarning()
        }
        #endif
    }

    /// Handle memory warning by stopping generation
    private func handleMemoryWarning() {
        print("GPT2ViewModel: Received memory warning")
        if isGenerating {
            stopResponseGeneration()
            errorMessage = "Generation stopped due to low memory"
        }
    }

    /// Access the model through the singleton manager
    private var model: GPT2? {
        return ModelManager.shared.model
    }

    func generateResponse() {
        guard let model = model else {
            if let loadError = ModelManager.shared.loadError {
                outputText = "Model error: \(loadError.localizedDescription)"
            } else {
                outputText = "Model not loaded."
            }
            return
        }
        guard !isGenerating else { return }

        isGenerating = true
        errorMessage = nil

        // Use weak self to prevent retain cycles
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            let inputTextCopy = self.inputText
            let _ = model.generate(text: inputTextCopy, nTokens: GPT2Configuration.defaultGenerationTokens) { [weak self] completion, time in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.outputText = inputTextCopy + completion
                    self.outputTime = time
                    self.isGenerating = false
                }
            }
        }
    }

    func stopResponseGeneration() {
        model?.stopGeneration()
        isGenerating = false
    }

    /// Clear any error messages
    func clearError() {
        errorMessage = nil
    }

    /// Attempt to reload the model (useful after errors)
    func reloadModel() {
        ModelManager.shared.reloadModel()
        if ModelManager.shared.isModelLoaded {
            errorMessage = nil
            outputText = "Model reloaded successfully"
        } else if let error = ModelManager.shared.loadError {
            errorMessage = error.localizedDescription
        }
    }
}
